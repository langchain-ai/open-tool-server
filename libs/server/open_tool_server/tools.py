import uuid
from typing import Any, Awaitable, Callable, Dict, NotRequired, Tuple, Union, cast

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from jsonschema_rs import validator_for
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as tool_decorator
from langchain_core.utils.function_calling import convert_to_openai_function
from typing_extensions import TypedDict


class RegisteredTool(TypedDict):
    """A registered tool."""

    id: str
    """Unique identifier for the tool."""
    name: str
    """Name of the tool."""
    description: str
    """Description of the tool."""
    input_schema: Dict[str, Any]
    """Input schema of the tool."""
    output_schema: NotRequired[Dict[str, Any]]
    """Output schema of the tool."""
    fn: Callable[[Dict[str, Any]], Awaitable[Any]]
    """Function to call the tool."""
    permissions: set[str]
    """Scopes required to call the tool.

    If empty, not permissions are required and the tool is considered to be public.
    """
    accepts: list[tuple[str, Any]]
    """List of run time arguments that the fn accepts.

    For example, a signature like def foo(x: int, request: Request) -> str:`,

    would have an entry in the `accepts` list as ("request", Request).
    """
    version: Tuple[int, int, int]
    """Version of the tool. Allows for semver versioning of tools.

    The version is a tuple of three integers: (major, minor, patch).

    A version of 1 will be represented as (1, 0, 0).
    """


def _is_allowed(
    tool: RegisteredTool, request: Request | None, auth_enabled: bool
) -> bool:
    """Check if the tool is listable."""
    required_permissions = tool["permissions"]

    # If tool requests Request object, but one is not provided, then the tool is not
    # allowed.
    for _, type_ in tool["accepts"]:
        if type_ is Request and request is None:
            return False

    if not auth_enabled or not required_permissions:
        # Used to avoid request.auth attribute access raising an assertion errors
        # when no auth middleware is enabled..
        return True
    permissions = request.auth.scopes if hasattr(request, "auth") else set()
    return required_permissions.issubset(permissions)


class ExecuteToolRequest(TypedDict):
    """Request to call a tool."""

    tool_id: str
    """An unique identifier for the tool to call."""
    input: NotRequired[Dict[str, Any]]
    """The input to pass to the tool."""
    execution_id: NotRequired[str]
    """Execution ID."""
    trace_id: NotRequired[str]
    """Trace ID."""


class Error(TypedDict):
    """Error message from the tool."""

    message: str
    """Error message for the user or AI model."""
    developer_message: NotRequired[str]
    """Internal error message for logging/debugging."""
    can_retry: NotRequired[bool]
    """Indicates whether the tool call can be retried."""
    additional_prompt_content: NotRequired[str]
    """Extra content to include in a retry prompt."""
    retry_after_ms: NotRequired[int]
    """Time in milliseconds to wait before retrying."""


class ToolException(Exception):
    """An exception that can be raised by a tool."""

    def __init__(
        self,
        *,
        user_message: str = "",
        developer_message: str = "",
        can_retry: bool = False,
        additional_prompt_content: str = "",
        retry_after_ms: int = 0,
    ) -> None:
        """Initializes the tool exception."""
        self.message = user_message
        self.developer_message = developer_message
        self.can_retry = can_retry
        self.additional_prompt_content = additional_prompt_content
        self.retry_after_ms = retry_after_ms


class ToolError(TypedDict):
    """Error message from the tool."""

    error: Error


class Value(TypedDict):
    """A successful value from the tool invocation."""

    value: Any


ToolOutput = Union[ToolError, Value]
"""Output from a tool invocation.

The output will be of type Value if the tool invocation was successful.

Otherwise, the output should be of type ToolError.
"""


class ExecuteToolResponse(TypedDict):
    """Response from a tool execution."""

    execution_id: str
    """A unique ID for the execution"""

    success: bool
    """Whether the execution was successful."""

    output: ToolOutput
    """The output of the tool execution."""


class ToolDefinition(TypedDict):
    """Used in the response of the list tools endpoint."""

    id: str
    """Unique identifier for the tool."""

    name: str
    """The name of the tool."""

    description: str
    """A human-readable explanation of the tool's purpose."""

    input_schema: Dict[str, Any]
    """The input schema of the tool. This is a JSON schema."""

    output_schema: NotRequired[Dict[str, Any]]
    """The output schema of the tool. This is a JSON schema."""

    version: str
    """Version of the tool. Allows for semver versioning of tools."""


class ToolHandler:
    def __init__(self) -> None:
        """Initializes the tool handler."""
        self.catalog: Dict[str, RegisteredTool] = {}
        self.auth_enabled = False

    def add(
        self, tool: Union[BaseTool, Callable], *, permissions: list[str] | None = None
    ) -> None:
        """Registers a tool in the catalog."""
        # If not already a BaseTool, we'll convert it to one using
        # the tool decorator.
        if not isinstance(tool, BaseTool):
            tool = tool_decorator(tool)

        if isinstance(tool, BaseTool):
            accepts = []
            from pydantic import BaseModel

            if not issubclass(tool.args_schema, BaseModel):
                raise NotImplementedError(
                    "Expected args_schema to be a Pydantic model. "
                    f"Got {type(tool.args_schema)}."
                    "This is not yet supported."
                )

            for name, field in tool.args_schema.model_fields.items():
                if field.annotation is Request:
                    accepts.append((name, Request))

            registered_tool = {
                "id": f"default.{tool.name}",
                "name": tool.name,
                "description": tool.description,
                "input_schema": convert_to_openai_function(tool)["parameters"],
                "fn": cast(Callable[[Dict[str, Any]], Awaitable[Any]], tool.ainvoke),
                "permissions": cast(set[str], set(permissions or [])),
                "accepts": accepts,
                # Register everything as version 1.0.0 for now.
                "version": (1, 0, 0),
            }
        else:
            raise AssertionError("Reached unreachable code")

        if registered_tool["name"] in self.catalog:
            # Add unique ID to support duplicated tools?
            raise ValueError(f"Tool {registered_tool['name']} already exists")
        self.catalog[registered_tool["name"]] = registered_tool

    async def call_tool(
        self, execute_tool_request: ExecuteToolRequest, request: Request | None
    ) -> ExecuteToolResponse:
        """Calls a tool by name with the provided payload."""
        tool_id = execute_tool_request["tool_id"]
        args = execute_tool_request.get("input", {})
        execution_id = execute_tool_request.get("execution_id", uuid.uuid4())

        if tool_id not in self.catalog:
            if self.auth_enabled:
                raise HTTPException(
                    status_code=403,
                    detail="Tool either does not exist or insufficient permissions",
                )

            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")

        tool = self.catalog[tool_id]

        if not _is_allowed(tool, request, self.auth_enabled):
            raise HTTPException(
                status_code=403,
                detail="Tool either does not exist or insufficient permissions",
            )

        # Validate and parse the payload according to the tool's input schema.
        fn = tool["fn"]

        injected_arguments = {}

        accepts = tool["accepts"]

        for name, field in accepts:
            if field is Request:
                injected_arguments[name] = request

        if isinstance(fn, Callable):
            payload_schema_ = tool["input_schema"]
            validator = validator_for(payload_schema_)
            if not validator.is_valid(args):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid payload for tool call to tool {tool_id} "
                        f"with args {args} and schema {payload_schema_}",
                    ),
                )
            # Update the injected arguments post-validation
            args.update(injected_arguments)
            tool_output = await fn(args)
        else:
            # This is an internal error
            raise AssertionError(f"Invalid tool implementation: {type(fn)}")

        return {
            "success": True,
            "execution_id": str(execution_id),
            "output": {"value": tool_output},
        }

    async def list_tools(self, request: Request | None) -> list[ToolDefinition]:
        """Lists all available tools in the catalog."""
        # Incorporate default permissions for the tools.
        tool_definitions = []

        for tool in self.catalog.values():
            if _is_allowed(tool, request, self.auth_enabled):
                tool_definition = {
                    "id": tool["name"],
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                    "version": ".".join(map(str, tool["version"])),
                }

                if "output_schema" in tool:
                    tool_definition["output_schema"] = tool["output_schema"]

                tool_definitions.append(tool_definition)

        return tool_definitions


def create_tools_router(tool_handler: ToolHandler) -> APIRouter:
    """Creates an API router for tools."""
    router = APIRouter()

    @router.get("", operation_id="list-tools")
    async def list_tools(request: Request) -> list[ToolDefinition]:
        """Lists available tools."""
        return await tool_handler.list_tools(request)

    @router.post("/execute", operation_id="execute-tool")
    async def execute_tool(
        execute_tool_request: ExecuteToolRequest, request: Request
    ) -> ExecuteToolResponse:
        """Call a tool by name with the provided payload."""
        return await tool_handler.call_tool(execute_tool_request, request)

    return router


class InjectedRequest(InjectedToolArg):
    """Annotation for injecting the starlette request object.

    Example:
        ..code-block:: python

            from typing import Annotated
            from open_tool_server.server.tools import InjectedRequest
            from starlette.requests import Request

            @app.tool(permissions=["group1"])
            async def who_am_i(request: Annotated[Request, InjectedRequest]) -> str:
                \"\"\"Return the user's identity\"\"\"
                # The `user` attribute can be used to retrieve the user object.
                # This object corresponds to the return value of the authentication
                # function.
                return request.user.identity
    """
