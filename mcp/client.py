"""MCP Client - connects to remote MCP server and calls tools."""

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
_port = int(os.getenv("MCP_SERVER_PORT", "8000"))
_DEFAULT_SERVER_URL = f"http://localhost:{_port}"


async def call_remote_add(server_url: str, a: float, b: float) -> float:
    """Call the add function on a remote MCP server.
    
    Args:
        server_url: The URL of the MCP server (e.g., http://localhost:8000)
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b from the remote server
    """
    # Parse URL and construct SSE endpoint
    parsed = urlparse(server_url)
    if not parsed.scheme:
        server_url = f"http://{server_url}"
    
    sse_url = f"{server_url.rstrip('/')}/sse"
    
    print(f"Connecting to MCP server at {sse_url}...")
    
    async with sse_client(sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools_result = await session.list_tools()
            print(f"\nAvailable tools: {[tool.name for tool in tools_result.tools]}")
            
            # Call the add tool
            print(f"\nCalling add({a}, {b})...")
            result = await session.call_tool("add", {"a": a, "b": b})
            
            # Extract result
            if result.content and len(result.content) > 0:
                # Get the text content from the result
                value = result.content[0].text
                return float(value)
            else:
                raise ValueError("No result returned from server")


async def interactive_client(server_url: str):
    """Run an interactive client that can call multiple tools."""
    parsed = urlparse(server_url)
    if not parsed.scheme:
        server_url = f"http://{server_url}"
    
    sse_url = f"{server_url.rstrip('/')}/sse"
    
    print(f"Connecting to MCP server at {sse_url}...")
    
    async with sse_client(sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # List available tools
            tools_result = await session.list_tools()
            print(f"\nConnected! Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            print("\n" + "="*50)
            print("Interactive MCP Client")
            print("="*50)
            print("Commands:")
            print("  add <a> <b>       - Add two numbers")
            print("  subtract <a> <b>  - Subtract two numbers")
            print("  multiply <a> <b>  - Multiply two numbers")
            print("  divide <a> <b>    - Divide two numbers")
            print("  quit              - Exit the client")
            print("="*50)
            
            while True:
                try:
                    user_input = input("\n> ").strip()
                    if not user_input:
                        continue
                    
                    parts = user_input.split()
                    command = parts[0].lower()
                    
                    if command == "quit":
                        print("Goodbye!")
                        break
                    
                    if command in ["add", "subtract", "multiply", "divide"]:
                        if len(parts) != 3:
                            print(f"Usage: {command} <a> <b>")
                            continue
                        
                        try:
                            a = float(parts[1])
                            b = float(parts[2])
                        except ValueError:
                            print("Error: Arguments must be numbers")
                            continue
                        
                        print(f"Calling {command}({a}, {b})...")
                        result = await session.call_tool(command, {"a": a, "b": b})
                        
                        if result.content and len(result.content) > 0:
                            value = result.content[0].text
                            print(f"Result: {value}")
                        else:
                            print("Error: No result returned")
                    else:
                        print(f"Unknown command: {command}")
                        
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")


async def main():
    """Main entry point for the client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Math Client")
    parser.add_argument(
        "--server",
        default=_DEFAULT_SERVER_URL,
        help=f"MCP server URL (default: {_DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "single"],
        default="interactive",
        help="Run mode: interactive or single call"
    )
    parser.add_argument(
        "--a",
        type=float,
        default=5.0,
        help="First number (for single mode)"
    )
    parser.add_argument(
        "--b",
        type=float,
        default=3.0,
        help="Second number (for single mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        await interactive_client(args.server)
    else:
        try:
            result = await call_remote_add(args.server, args.a, args.b)
            print(f"Result: {args.a} + {args.b} = {result}")
        except Exception as e:
            print(f"Error calling remote server: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # 示例：直接调用远程加法函数
    try:
        res = asyncio.run(call_remote_add("http://192.168.43.120:8000", 1, 2))
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    