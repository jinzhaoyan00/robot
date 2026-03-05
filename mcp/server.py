"""MCP Server with addition tool - supports remote connections via SSE."""

from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Math Server")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The difference of a and b
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: First number (dividend)
        b: Second number (divisor)
    
    Returns:
        The quotient of a and b
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    import os
    import uvicorn
    from pathlib import Path
    from dotenv import load_dotenv
    from starlette.applications import Starlette
    from starlette.responses import Response
    from mcp.server.sse import SseServerTransport
    from starlette.routing import Route, Mount

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8000"))

    # Create SSE transport with message endpoint
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        # Use connect_sse to establish SSE connection
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0],  # read_stream
                streams[1],  # write_stream
                mcp._mcp_server.create_initialization_options(),
            )
        # Must return Response to avoid NoneType error on client disconnect
        return Response()

    # Create Starlette app with routes
    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    print(f"Starting MCP Math Server on http://{host}:{port}")
    print(f"SSE endpoint: http://localhost:{port}/sse")
    print(f"Message endpoint: http://localhost:{port}/messages/")
    print("\nAvailable tools: add, subtract, multiply, divide")
    print(f"\nRemote clients can connect using: http://<server-ip>:{port}")
    uvicorn.run(app, host=host, port=port)
