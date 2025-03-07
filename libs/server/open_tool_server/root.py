from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from typing_extensions import TypedDict

from open_tool_server._version import __version__


class InfoResponse(TypedDict):
    """Get information about the server."""

    version: str


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_content = f"""
    <html>
        <head>
            <title>LangChain Tool Server</title>
        </head>
        <body>
            <h1>LangChain Tool Server</h1>
            <p>
                <ul>
                    <li><strong>Version</strong>: {__version__}</li>
                    <li><a href="docs">Docs</strong></li>
                </ul>
            </p>
        </body>
    </html>
    """
    return html_content


@router.get("/info")
def get_info() -> InfoResponse:
    """Get information about the server."""
    return {"version": __version__}


@router.get("/health")
def health() -> dict:
    """Are we OK?"""
    return {"status": "OK"}
