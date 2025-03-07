> [!IMPORTANT]  
> This is a work in progress. The API is expected to change.

# Open Tool Server

A dedicated tool server decouples the creation of specialized tools (e.g., for retrieving data from specific knowledge sources) from agent development. This separation enables different teams to contribute and manage tools independently. Agents can then be rapidly configured—by simply specifying a prompt and a set of accessible tools. This streamlined approach simplifies authentication and authorization and accelerates the deployment of agents into production.

Users working in a local environment that need MCP, [can enable MCP support](#MCP-SSE). In comparison to [MCP](https://modelcontextprotocol.io/introduction), this specification uses stateless connection which makes it suitable for web deployment. 

## Why

- 🖥️ **Stateless Web Deployment**: Deploy as a web server without the need for persistent connections, allowing easy autoscaling and load balancing.
- 📡 **Simple REST Protocol**: Leverage a straightforward REST API.
- 🔐 **Built-In Authentication**: Out-of-the-box auth support, ensuring only authorized users can access tools.
- 🛠️ **Decoupled Tool Creation**: In an enterprise setting, decouple the creation of specialized tools (like data retrieval from specific knowledge sources) from the agent configuration.

## Installation

```bash
pip install open-tool-server open-tool-client
```

## Example Usage

### Server 

Add a server.py file to your project and define your tools with type hints.

```python
from typing import Annotated
from starlette.requests import Request

from open_tool_server.tools import InjectedRequest
from open_tool_server import Server, Auth

app = Server()
auth = Auth()
app.add_auth(auth)

@auth.authenticate
async def authenticate(headers: dict[bytes, bytes]) -> dict:
    """Authenticate incoming requests."""
    api_key = headers.get(b"x-api-key")

    # Replace this with actual authentication logic.
    api_key_to_user = {
        b"1": {"permissions": ["authenticated", "group1"], "identity": "some-user"},
        b"2": {"permissions": ["authenticated", "group2"], "identity": "another-user"},
    }

    if not api_key or api_key not in api_key_to_user:
        raise auth.exceptions.HTTPException(detail="Not authorized")
    return api_key_to_user[api_key]



@app.tool(permissions=["group1"])
async def echo(msg: str) -> str:
    """Echo a message."""
    return msg + "!"


@app.tool(permissions=["group2"])
async def say_hello() -> str:
    """Say hello."""
    return "Hello"

@app.tool(permissions=["authenticated"])
async def who_am_i(request: Annotated[Request, InjectedRequest]) -> str:
    """Get the user identity."""
    return request.user.identity
```

### Client

Add a client.py file to your project and define your client.

```python
import asyncio

from open_tool_client import get_async_client


async def main():
    if len(sys.argv) < 2:
        print(
            "Usage: uv run client.py url of open-tool-server  (i.e. http://localhost:8080/)>"
        )
        sys.exit(1)

    url = sys.argv[1]
    client = get_async_client(url=url)
    # Check server status
    print(await client.ok())  # "OK"
    print(await client.info())  # Server version and other information

    # List tools
    print(await client.tools.list())  # List of tools
    # Call a tool
    print(await client.tools.execute("add", {"x": 1, "y": 2}))  # 3

    # Get as langchain tools
    select_tools = ["echo", "add"]
    tools = await client.tools.as_langchain_tools(select_tools)
    # Async
    print(await tools[0].ainvoke({"msg": "Hello"}))  # "Hello!"
    print(await tools[1].ainvoke({"x": 1, "y": 3}))  # 4


if __name__ == "__main__":
    import sys

    asyncio.run(main())
```

### Sync Client

If you need a synchronous client, you can use the `get_sync_client` function.

```python
from open_tool_client import get_sync_client
```

### React Agent

Here's an example of how you can use the Open Tool Server with a prebuilt LangGraph react agent.

```shell
pip install langchain-anthropic langgraph
```

```python
import os

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from open_tool_client import get_sync_client

if "ANTHROPIC_API_KEY" not in os.environ:
    raise ValueError("Please set ANTHROPIC_API_KEY in the environment.")

tool_server = get_sync_client(
    url=... # URL of the tool server
    # headers=... # If you enabled auth
)
# Get tool definitions from the server
tools = tool_server.tools.as_langchain_tools()
print("Loaded tools:", tools)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
agent = create_react_agent(model, tools=tools)
print()

user_message = "What is the temperature in Paris?"
messages = agent.invoke({"messages": [{"role": "user", "content": user_message}]})[
    "messages"
]

for message in messages:
    message.pretty_print()
```

### MCP SSE

You can enable support for the MCP SSE protocol by passing `enable_mcp=True` to the Server constructor.

> [!IMPORTANT]  
> Auth is not supported when using MCP SSE. So if you try to use auth and enable MCP, the server will raise an exception by design.

```python
from open_tool_server import Server

app = Server(enable_mcp=True)

@app.tool()
async def echo(msg: str) -> str:
    """Echo a message."""
    return msg + "!"
```

This will mount an MCP SSE app at /mcp/sse. You can use the MCP client to connect to the server.

Use MCP client to connect to the server. **The url should be the same as the server url with `/mcp/sse` appended.**

```python
from mcp import ClientSession

from mcp.client.sse import sse_client

async def main() -> None:
    # Please replace [host] with the actual host
    # IMPORTANT: Add /mcp/sse to the url!
    url = "[host]/mcp/sse" 
    async with sse_client(url=url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(tools)
            result = await session.call_tool("echo", {"msg": "Hello, world!"})
            print(result)
```

## Concepts

### Tool Definition

A tool is a function that can be called by the client. It can be a simple function or a coroutine. The function signature should have type hints. The server will use these type hints to validate the input and output of the tool.

```python
@app.tool()
async def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

#### Permissions

You can specify `permissions` for a tool. The client must have the required permissions to call the tool. If the client does not have the required permissions, the server will return a 403 Forbidden error.

```python
@app.tool(permissions=["group1"])
async def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

A client must have **all** the required permissions to call the tool rather than a subset of the permissions.

#### Injected Request

A tool can request access to Starlette's `Request` object by using the `InjectedRequest` type hint. This can be useful for getting information about the request, such as the user's identity.

```python
from typing import Annotated
from open_tool_server import InjectedRequest
from starlette.requests import Request

@app.tool(permissions=["group1"])
async def who_am_i(request: Annotated[Request, InjectedRequest]) -> str:
    """Return the user's identity"""
    # The `user` attribute can be used to retrieve the user object.
    # This object corresponds to the return value of the authentication function.
    return request.user.identity
```


### Tool Discovery

A client can list all available tools by calling the `tools.list` method. The server will return a list of tools with their names and descriptions.

The client will only see tools for which they have the required permissions.

```python
from open_tool_client import get_async_client

async def get_tools():
    # Headers are entirely dependent on how you implement your authentication
    # (see Auth section)
    client = get_async_client(url="http://localhost:8080/", headers={"x-api-key": "api key"})
    tools = await client.tools.list()
    # If you need langchain tools you can use the as_langchain_tools method
    langchain_tools = await client.tools.as_langchain_tools()
    # Do something
    ...
```

### Auth

You can add authentication to the server by defining an authentication function. 

**Tutorial**

If you want to add realistic authentication to your server, you can follow the 3rd tutorial in the [Connecting an Authentication Provider](https://langchain-ai.github.io/langgraph/tutorials/auth/add_auth_server/) series for 
LangGraph Platform. It's a separate project, but the tutorial has useful information for setting up authentication in your server.

#### Auth.authenticate

The authentication function is a coroutine that can request any of the following parameters:

| Parameter | Description                               |
|-----------|-------------------------------------------|
| `request` | The request object.                       |
| `headers` | A dictionary of headers from the request. |
| `body`    | The body of the request.                  |


The function should either:

1. Return a user object if the request is authenticated.
2. Raise an `auth.exceptions.HTTPException` if the request cannot be authenticated.

```python
from open_tool_server import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict[bytes, bytes]) -> dict:
    """Authenticate incoming requests."""
    is_authenticated = ... # Your authentication logic here
    if not is_authenticated:
        raise auth.exceptions.HTTPException(detail="Not authorized")
    
    return {
        "identity": "some-user",
        "permissions": ["authenticated", "group1"],
        # Add any other user information here
        "foo": "bar",
    } 
```

