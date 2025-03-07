from __future__ import annotations

import asyncio
import logging
import sys
from importlib import metadata
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence
from typing import TYPE_CHECKING

import httpx
import orjson
from httpx._types import QueryParamTypes

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

logger = logging.getLogger(__name__)


def _get_headers(custom_headers: Optional[dict[str, str]]) -> dict[str, str]:
    """Combine api_key and custom user-provided headers."""
    custom_headers = custom_headers or {}
    headers = {
        "User-Agent": f"open-tool-sdk-py/{__version__}",
        **custom_headers,
    }
    return headers


def _decode_json(r: httpx.Response) -> Any:
    body = r.read()
    return orjson.loads(body if body else None)


def _encode_json(json: Any) -> tuple[dict[str, str], bytes]:
    body = orjson.dumps(
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


def _orjson_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def _aencode_json(json: Any) -> tuple[dict[str, str], bytes]:
    """Encode JSON."""
    if json is None:
        return {}, None
    body = await asyncio.get_running_loop().run_in_executor(
        None,
        orjson.dumps,
        json,
        _orjson_default,
        orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
    )
    content_length = str(len(body))
    content_type = "application/json"
    headers = {"Content-Length": content_length, "Content-Type": content_type}
    return headers, body


async def _adecode_json(r: httpx.Response) -> Any:
    """Decode JSON."""
    body = await r.aread()
    return (
        await asyncio.get_running_loop().run_in_executor(None, orjson.loads, body)
        if body
        else None
    )


class AsyncHttpClient:
    """Handle async requests to the LangGraph API.

    Adds additional error messaging & content handling above the
    provided httpx client.

    Attributes:
        client (httpx.AsyncClient): Underlying HTTPX async client.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def get(self, path: str, *, params: Optional[QueryParamTypes] = None) -> Any:
        """Send a GET request."""
        r = await self.client.get(path, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return await _adecode_json(r)

    async def post(self, path: str, *, json: Optional[dict]) -> Any:
        """Send a POST request."""
        if json is not None:
            headers, content = await _aencode_json(json)
        else:
            headers, content = {}, b""
        r = await self.client.post(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return await _adecode_json(r)

    async def put(self, path: str, *, json: dict) -> Any:
        """Send a PUT request."""
        headers, content = await _aencode_json(json)
        r = await self.client.put(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return await _adecode_json(r)

    async def patch(self, path: str, *, json: dict) -> Any:
        """Send a PATCH request."""
        headers, content = await _aencode_json(json)
        r = await self.client.patch(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return await _adecode_json(r)

    async def delete(self, path: str, *, json: Optional[Any] = None) -> None:
        """Send a DELETE request."""
        r = await self.client.request("DELETE", path, json=json)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (await r.aread()).decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e


class SyncHttpClient:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def get(self, path: str, *, params: Optional[QueryParamTypes] = None) -> Any:
        """Send a GET request."""
        r = self.client.get(path, params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return _decode_json(r)

    def post(self, path: str, *, json: Optional[dict]) -> Any:
        """Send a POST request."""
        if json is not None:
            headers, content = _encode_json(json)
        else:
            headers, content = {}, b""
        r = self.client.post(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return _decode_json(r)

    def put(self, path: str, *, json: dict) -> Any:
        """Send a PUT request."""
        headers, content = _encode_json(json)
        r = self.client.put(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return _decode_json(r)

    def patch(self, path: str, *, json: dict) -> Any:
        """Send a PATCH request."""
        headers, content = _encode_json(json)
        r = self.client.patch(path, headers=headers, content=content)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e
        return _decode_json(r)

    def delete(self, path: str, *, json: Optional[Any] = None) -> None:
        """Send a DELETE request."""
        r = self.client.request("DELETE", path, json=json)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = r.read().decode()
            if sys.version_info >= (3, 11):
                e.add_note(body)
            else:
                logger.error(f"Error from open-tool-server: {body}", exc_info=e)
            raise e


############
# PUBLIC API


def get_async_client(
    *,
    url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> AsyncClient:
    """Get instance.

    Args:
        url: The URL of the tool server.
        headers: Optional custom headers
        transport: Optional transport to use.

    Returns:
        AsyncClient: The top-level client for accessing the tool server.
    """

    if url is None:
        url = "http://localhost:2424"

    if transport is None:
        transport = httpx.AsyncHTTPTransport(retries=5)

    client = httpx.AsyncClient(
        base_url=url,
        transport=transport,
        timeout=httpx.Timeout(connect=5, read=300, write=300, pool=5),
        headers=_get_headers(headers),
    )
    return AsyncClient(client)


def get_sync_client(
    *,
    url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> SyncClient:
    """Get instance.

    Args:
        url: The URL of the tool server.
        headers: Optional custom headers
        transport: Optional transport to use.

    Returns:
        AsyncClient: The top-level client for accessing the tool server.
    """

    if url is None:
        url = "http://localhost:2424"

    if transport is None:
        transport = httpx.HTTPTransport(retries=5)

    client = httpx.Client(
        base_url=url,
        transport=transport,
        timeout=httpx.Timeout(connect=5, read=300, write=300, pool=5),
        headers=_get_headers(headers),
    )
    return SyncClient(client)


class AsyncClient:
    """Top-level client for the tools server."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Initialize the client."""
        self.http = AsyncHttpClient(client)
        self.tools = AsyncToolsClient(self.http)

    async def info(self) -> Any:
        return await self.http.get("/info")

    async def health(self) -> Any:
        return await self.http.get("/health")


class SyncClient:
    """Top-level client for the tools server."""

    def __init__(self, client: httpx.Client) -> None:
        """Initialize the client."""
        self.http = SyncHttpClient(client)
        self.tools = SyncToolsClient(self.http)

    def info(self) -> Any:
        return self.http.get("/info")

    def health(self) -> Any:
        return self.http.get("/health")


class AsyncToolsClient:
    """Tools API."""

    def __init__(self, http: AsyncHttpClient) -> None:
        """Initialize the client."""
        self.http = http

    async def list(self) -> Any:
        """List tools."""
        return await self.http.get("/tools")

    async def call(self, tool_id: str, args: Dict[str, Any] | None = None) -> Any:
        """Call a tool."""
        payload = {"tool_id": tool_id}
        if args is not None:
            payload["input"] = args
        return await self.http.post("/tools/execute", json=payload)

    async def as_langchain_tools(
        self, *, select: Sequence[str] | None = None
    ) -> List[BaseTool]:
        """Load tools from the server.

        Args:
            select: If specified, will only load the selected tools.
                Otherwise, all tools will be loaded.

        Returns:
            a list of LangChain tools.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError(
                "To use this method, you must have langchain-core installed. "
                "You can install it with `pip install langchain-core`."
            ) from e

        tools = []

        available_tools = await self.list()

        available_tools_by_name = {tool["name"]: tool for tool in available_tools}

        if select is None:
            select = list(available_tools_by_name)

        if set(select) - set(available_tools_by_name):
            raise ValueError(
                f"Unknown tool names: {set(select) - set(available_tools_by_name)}"
            )

        # The code below will create LangChain style tools by binding
        # tool metadata and the tool implementation together in a StructuredTool.
        def create_tool_caller(tool_name_: str) -> Callable[..., Awaitable[Any]]:
            """Create a tool caller."""

            async def call_tool(**kwargs: Any) -> tuple[Any]:
                """Call a tool."""
                call_tool_result = await self.call(tool_name_, kwargs)
                return call_tool_result

            return call_tool

        for tool_name in select:
            tool_spec = available_tools_by_name[tool_name]

            tools.append(
                StructuredTool(
                    name=tool_name,
                    description=tool_spec["description"],
                    args_schema=tool_spec["input_schema"],
                    coroutine=create_tool_caller(tool_name),
                )
            )
        return tools


class SyncToolsClient:
    """Tools API."""

    def __init__(self, http: SyncHttpClient) -> None:
        """Initialize the client."""
        self.http = http

    def list(self) -> Any:
        """List tools."""
        return self.http.get("/tools")

    def call(self, tool_id: str, args: Dict[str, Any] | None = None) -> Any:
        """Call a tool."""

        payload = {"tool_id": tool_id}
        if args is not None:
            payload["input"] = args
        return self.http.post("/tools/execute", json=payload)

    def as_langchain_tools(
        self, *, select: Sequence[str] | None = None
    ) -> List[BaseTool]:
        """Load tools from the server.

        Args:
            select: If specified, will only load the selected tools.
                Otherwise, all tools will be loaded.

        Returns:
            a list of LangChain tools.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError(
                "To use this method, you must have langchain-core installed. "
                "You can install it with `pip install langchain-core`."
            ) from e

        tools = []

        available_tools = self.list()

        available_tools_by_name = {tool["name"]: tool for tool in available_tools}

        if select is None:
            select = list(available_tools_by_name)

        if set(select) - set(available_tools_by_name):
            raise ValueError(
                f"Unknown tool names: {set(select) - set(available_tools_by_name)}"
            )

        # The code below will create LangChain style tools by binding
        # tool metadata and the tool implementation together in a StructuredTool.

        def create_tool_caller(tool_name_: str) -> Callable[..., Any]:
            """Create a tool caller."""

            def call_tool(**kwargs: Any) -> tuple[Any]:
                """Call a tool."""
                call_tool_result = self.call(tool_name_, kwargs)
                return call_tool_result

            return call_tool

        for tool_name in select:
            tool_spec = available_tools_by_name[tool_name]

            tools.append(
                StructuredTool(
                    name=tool_name,
                    description=tool_spec["description"],
                    args_schema=tool_spec["inputSchema"],
                    func=create_tool_caller(tool_name),
                )
            )
        return tools


__all__ = [
    "get_async_client",
    "get_sync_client",
    "AsyncClient",
    "SyncClient",
]
