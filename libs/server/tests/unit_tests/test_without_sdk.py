"""Test the server."""

from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Optional

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from open_tool_server import Server
from open_tool_server._version import __version__
from open_tool_server.tools import InjectedRequest


@asynccontextmanager
async def get_async_test_client(
    server: FastAPI, *, path: Optional[str] = None, raise_app_exceptions: bool = True
) -> AsyncGenerator[AsyncClient, None]:
    """Get an async client."""
    url = "http://localhost:9999"
    if path:
        url += path
    transport = ASGITransport(
        app=server,
        raise_app_exceptions=raise_app_exceptions,
    )
    async_client = AsyncClient(base_url=url, transport=transport)
    try:
        yield async_client
    finally:
        await async_client.aclose()


async def test_health() -> None:
    app = Server()
    async with get_async_test_client(app) as client:
        response = await client.get("/health")
        response.raise_for_status()
        assert response.json() == {"status": "OK"}


async def test_info() -> None:
    """Test info end-point."""
    app = Server()
    async with get_async_test_client(app) as client:
        response = await client.get("/info")
        response.raise_for_status()
        json_data = response.json()
        assert json_data == {
            "version": __version__,
        }


async def test_list_tools() -> None:
    """Test list tools."""
    app = Server()
    async with get_async_test_client(app) as client:
        response = await client.get("/tools")
        response.raise_for_status()
        json_data = response.json()
        assert json_data == []


async def test_lifespan() -> None:
    import contextlib

    from starlette.testclient import TestClient

    calls = []

    @contextlib.asynccontextmanager
    async def lifespan(app):
        calls.append("startup")
        yield {"foo": "bar"}
        calls.append("shutdown")

    app = Server(lifespan=lifespan)

    @app.tool()
    def what_is_foo(request: Annotated[Request, InjectedRequest]) -> str:
        """Get foo"""
        return request.state.foo

    # Using Starlette's TestClient to make sure that the lifespan is used.
    # Seems to not be supported with httpx's ASGITransport.
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert calls == ["startup"]
        response = client.post(
            "/tools/execute", json={"tool_id": "what_is_foo", "input": {}}
        )
        response.raise_for_status()
        result = response.json()
        assert "execution_id" in result
        del result["execution_id"]
        assert result == {
            "output": {"value": "bar"},
            "success": True,
        }

    assert calls == ["startup", "shutdown"]
