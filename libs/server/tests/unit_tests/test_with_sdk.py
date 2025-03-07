"""Test the server."""

from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Optional, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, HTTPStatusError
from open_tool_client import AsyncClient, get_async_client
from starlette.authentication import BaseUser
from starlette.requests import Request

from open_tool_server import Server
from open_tool_server._version import __version__
from open_tool_server.auth import Auth
from open_tool_server.tools import InjectedRequest


@asynccontextmanager
async def get_async_test_client(
    server: FastAPI,
    *,
    path: Optional[str] = None,
    raise_app_exceptions: bool = True,
    headers: dict | None = None,
) -> AsyncGenerator[AsyncClient, None]:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    transport = ASGITransport(
        app=server,
        raise_app_exceptions=raise_app_exceptions,
    )

    client = get_async_client(transport=transport, headers=headers)

    try:
        yield cast(AsyncClient, client)
    finally:
        del client


async def test_health() -> None:
    app = Server()
    async with get_async_test_client(app) as client:
        assert await client.health() == {"status": "OK"}


async def test_info() -> None:
    app = Server()
    async with get_async_test_client(app) as client:
        assert await client.info() == {
            "version": __version__,
        }


async def test_add_langchain_tool() -> None:
    """Test adding a tool that's defined using langchain tool decorator."""
    app = Server()

    # Test prior to adding any tools
    async with get_async_test_client(app) as client:
        tools = await client.tools.list()
        assert tools == []

    @app.tool
    async def say_hello() -> str:
        """Say hello."""
        return "Hello"

    @app.tool
    async def echo(msg: str) -> str:
        """Echo the message back."""
        return msg

    @app.tool
    async def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    async with get_async_test_client(app) as client:
        data = await client.tools.list()
        assert data == [
            {
                "description": "Say hello.",
                "id": "say_hello",
                "input_schema": {"properties": {}, "type": "object"},
                "name": "say_hello",
                "version": "1.0.0",
            },
            {
                "description": "Echo the message back.",
                "id": "echo",
                "input_schema": {
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                    "type": "object",
                },
                "name": "echo",
                "version": "1.0.0",
            },
            {
                "description": "Add two integers.",
                "id": "add",
                "input_schema": {
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                    "type": "object",
                },
                "name": "add",
                "version": "1.0.0",
            },
        ]


async def test_call_tool() -> None:
    """Test call parameterless tool."""
    app = Server()

    @app.tool
    async def say_hello() -> str:
        """Say hello."""
        return "Hello"

    @app.tool
    async def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    async with get_async_test_client(app) as client:
        response = await client.tools.execute(
            "say_hello",
            {},
        )

        assert "execution_id" in response
        del response["execution_id"]
        assert response == {
            "output": {
                "value": "Hello",
            },
            "success": True,
        }


async def test_create_langchain_tools_from_server() -> None:
    """Test create langchain tools from server."""
    app = Server()

    @app.tool
    async def say_hello() -> str:
        """Say hello."""
        return "Hello"

    @app.tool
    async def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    async with get_async_test_client(app) as client:
        tools = await client.tools.as_langchain_tools(tool_ids=["say_hello", "add"])
        say_hello_client_side = tools[0]
        add_client_side = tools[1]

        assert await say_hello_client_side.ainvoke({}) == "Hello"
        assert say_hello_client_side.args_schema == {"properties": {}, "type": "object"}

        assert await add_client_side.ainvoke({"x": 1, "y": 2}) == 3
        assert add_client_side.args == {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        }


class User(BaseUser):
    """User class."""

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return True

    @property
    def display_name(self) -> str:
        """Get user's display name."""
        return "Test User"

    @property
    def identity(self) -> str:
        """Get user's identity."""
        return "test-user"


async def test_auth_list_tools() -> None:
    """Test ability to list tools."""

    app = Server()
    auth = Auth()
    app.add_auth(auth)

    @app.tool(permissions=["group1"])
    async def say_hello() -> str:
        """Say hello."""
        return "Hello"

    @app.tool(permissions=["group2"])
    async def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    @auth.authenticate
    async def authenticate(headers: dict[bytes, bytes]) -> dict:
        """Authenticate incoming requests."""
        # Validate credentials (e.g., API key, JWT token)
        api_key = headers.get(b"x-api-key")
        if not api_key or api_key != b"123":
            raise auth.exceptions.HTTPException(detail="Not authorized")

        return {"permissions": ["group1"], "identity": "some-user"}

    async with get_async_test_client(app, headers={"x-api-key": "123"}) as client:
        tools = await client.tools.list()
        assert tools == [
            {
                "description": "Say hello.",
                "input_schema": {"properties": {}, "type": "object"},
                "name": "say_hello",
            }
        ]

        await client.tools.execute("say_hello", {})


async def test_call_tool_with_auth() -> None:
    """Test calling a tool with authentication provided."""
    app = Server()

    @app.tool(permissions=["group1"])
    async def say_hello(request: Annotated[Request, InjectedRequest]) -> str:
        """Say hello."""
        return "Hello"

    auth = Auth()

    @auth.authenticate
    async def authenticate(headers: dict[bytes, bytes]) -> dict:
        """Authenticate incoming requests."""
        api_key = headers.get(b"x-api-key")

        api_key_to_user = {
            b"1": {"permissions": ["group1"], "identity": "some-user"},
            b"2": {"permissions": ["group2"], "identity": "another-user"},
        }

        if not api_key or api_key not in api_key_to_user:
            raise auth.exceptions.HTTPException(detail="Not authorized")

        return api_key_to_user[api_key]

    app.add_auth(auth)

    async with get_async_test_client(app, headers={"x-api-key": "1"}) as client:
        assert await client.tools.execute("say_hello", {}) == "Hello"

    async with get_async_test_client(app, headers={"x-api-key": "2"}) as client:
        # `2` does not have permission to call `say_hello`
        with pytest.raises(HTTPStatusError) as exception_info:
            assert await client.tools.execute("say_hello", {}) == "Hello"
        assert exception_info.value.response.status_code == 403

    async with get_async_test_client(app, headers={"x-api-key": "3"}) as client:
        # `3` does not have permission to call `say_hello`
        with pytest.raises(HTTPStatusError) as exception_info:
            assert await client.tools.execute("say_hello", {}) == "Hello"

        assert exception_info.value.response.status_code == 401


async def test_call_tool_with_injected() -> None:
    """Test calling a tool with an injected request."""
    app = Server()

    @app.tool(permissions=["authorized"])
    async def get_user_identity(request: Annotated[Request, InjectedRequest]) -> str:
        """Get the user's identity."""
        return request.user.identity

    auth = Auth()

    @auth.authenticate
    async def authenticate(headers: dict[bytes, bytes]) -> dict:
        """Authenticate incoming requests."""
        # Validate credentials (e.g., API key, JWT token)
        api_key = headers.get(b"x-api-key")

        api_key_to_user = {
            b"1": {"permissions": ["authorized"], "identity": "some-user"},
            b"2": {"permissions": ["authorized"], "identity": "another-user"},
            b"3": {"permissions": ["not-authorized"], "identity": "not-authorized"},
        }

        if not api_key or api_key not in api_key_to_user:
            raise auth.exceptions.HTTPException(detail="Not authorized")

        return api_key_to_user[api_key]

    app.add_auth(auth)

    async with get_async_test_client(app, headers={"x-api-key": "1"}) as client:
        result = await client.tools.execute("get_user_identity")
        assert result["output"]["value"] == "some-user"

    async with get_async_test_client(app, headers={"x-api-key": "2"}) as client:
        result = await client.tools.execute("get_user_identity")
        assert result["output"]["value"] == "another-user"

    async with get_async_test_client(app, headers={"x-api-key": "3"}) as client:
        # Make sure this raises 401?
        with pytest.raises(HTTPStatusError) as exception_info:
            result = client.tools.execute("get_user_identity", {})

        assert exception_info.value.response.status_code == 403

    # Authenticated but tool does not exist
    async with get_async_test_client(app, headers={"x-api-key": "1"}) as client:
        # Make sure this raises 401?
        with pytest.raises(HTTPStatusError) as exception_info:
            await client.tools.execute("does_not_exist", {})

        assert exception_info.value.response.status_code == 404

    # Not authenticated
    async with get_async_test_client(app, headers={"x-api-key": "6"}) as client:
        # Make sure this raises 401?
        with pytest.raises(HTTPStatusError) as exception_info:
            await client.tools.execute("does_not_exist", {})

        assert exception_info.value.response.status_code == 401


async def test_exposing_existing_langchain_tools() -> None:
    """Test exposing existing langchain tools."""
    from langchain_core.tools import StructuredTool, tool

    @tool
    def say_hello_sync() -> str:
        """Say hello."""
        return "Hello"

    @tool
    async def say_hello_async() -> str:
        """Say hello."""
        return "Hello"

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async def amultiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

    server = Server()
    auth = Auth()
    server.add_auth(auth)

    @auth.authenticate
    async def authenticate(headers: dict) -> dict:
        """Authenticate incoming requests."""
        api_key = headers.get(b"x-api-key")

        api_key_to_user = {
            b"1": {"permissions": ["group1"], "identity": "some-user"},
        }

        if not api_key or api_key not in api_key_to_user:
            raise auth.exceptions.HTTPException(detail="Not authorized")

        return api_key_to_user[api_key]

    server.tool(say_hello_sync, permissions=["group1"])
    server.tool(say_hello_async, permissions=["group1"])
    server.tool(calculator, permissions=["group1"])

    async with get_async_test_client(server, headers={"x-api-key": "1"}) as client:
        tools = await client.tools.list()
        assert tools == [
            {
                "description": "Say hello.",
                "id": "say_hello_sync",
                "input_schema": {
                    "properties": {},
                    "type": "object",
                },
                "name": "say_hello_sync",
            },
            {
                "description": "Say hello.",
                "id": "say_hello_async",
                "input_schema": {
                    "properties": {},
                    "type": "object",
                },
                "name": "say_hello_async",
            },
            {
                "description": "Multiply two numbers.",
                "id": "multiply",
                "input_schema": {
                    "properties": {
                        "a": {
                            "type": "integer",
                        },
                        "b": {
                            "type": "integer",
                        },
                    },
                    "required": [
                        "a",
                        "b",
                    ],
                    "type": "object",
                },
                "name": "multiply",
            },
        ]

        result = await client.tools.execute("say_hello_sync", {})
        assert result["output"]["value"] == "Hello"

        result = await client.tools.execute("say_hello_async", {})
        assert result["output"]["value"] == "Hello"

        result = await client.tools.execute("multiply", {"a": 2, "b": 3})
        assert result["output"]["value"] == 6
