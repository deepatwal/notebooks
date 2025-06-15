
create a python virtual env:
    uv venv env-kg-mcp-server
   
uv pip install:
    uv pip install httpx

create a folder and run the command the following command in the folder:
    uv init

To Register Your Server in Claude Desktop App:
    uv run main.py

Add following to claude_desktop_config.json:
{
    "mcpServers": {
        "KnowledgeGraphMCP": {
        "command": "C:\\users\\deepa\\.local\\bin\\uv.EXE",
        "args": [
            "--directory",
            "C:\\Users\\deepa\\data\\workspace\\notebooks\\knowledge-graph-mcp-server",
            "run",
            "main.py"
        ]
        }
    }
    }

To Launch with the Dev Inspector UI:
    uv run --active mcp dev main.py
                or
    uv run --active mcp dev main.py --with-editable .