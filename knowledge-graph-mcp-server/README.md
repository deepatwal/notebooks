
create a python virtual env:
    uv venv env-mcp-server
    uv add "mcp[cli]" --active

create a folder and run the command the following command in the folder:
    uv init

To Register Your Server in Claude Desktop App:
    uv run main.py
    
    this will create a new virtual environment in the current folder
        uv run --active mcp install main.py

To Run the Server Locally (for testing/debugging):
    uv run --active mcp run main.py

To Launch with the Dev Inspector UI:
    uv run --active mcp dev main.py
                or
    uv run --active mcp dev main.py --with-editable .