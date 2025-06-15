from mcp.server.fastmcp import FastMCP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

mcp = FastMCP("Demo")
logging.info("FastMCP instance created with name 'Demo'")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    logging.info(f"add() called with arguments: a={a}, b={b}")
    result = a + b
    logging.info(f"add() result: {result}")
    return result


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    greeting = f"Hello, {name}!"
    logging.info(f"get_greeting() result: {greeting}")
    return greeting


@mcp.resource("greeting://example")
def greeting_example() -> str:
    """Get a greeting"""
    return get_greeting("Claude")
