
uv venv env-mcp-server
uv add "mcp[cli]" --active

To Register Your Server in Claude Desktop App:
    uv run mcp install main.py

To Run the Server Locally (for testing/debugging):
    uv run mcp run main.py

To Launch with the Dev Inspector UI:
    uv run mcp dev main.py

    or 
    
    uv run mcp dev main.py --with-editable .