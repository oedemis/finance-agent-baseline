"""
MCP Client for Purple Agent.

Connects to the green agent's MCP server to access financial research tools.
Auto-injects context_id for state isolation across concurrent tasks.
"""
import logging
import os
from typing import Optional

from fastmcp import Client

logger = logging.getLogger("finance_agent.mcp")


class FinanceToolsClient:
    """
    MCP client for accessing financial research tools from green agent.

    Features:
    - Auto-injects context_id for state isolation
    - Convenience methods for each tool
    - Connection pooling (reuses same client)
    - Error handling with fallback
    """

    def __init__(self, mcp_server_url: Optional[str] = None, context_id: Optional[str] = None):
        """
        Initialize MCP client.

        Args:
            mcp_server_url: URL of green agent's MCP server (e.g., "http://localhost:9020")
                          If None, reads from MCP_SERVER_URL environment variable
            context_id: A2A context ID for this conversation (for state isolation)
        """
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9020")
        self.context_id = context_id or "default"
        self._client: Optional[Client] = None
        self._connected = False

        logger.info(f"FinanceToolsClient initialized: server={self.mcp_server_url}, context={self.context_id}")

    async def connect(self):
        """Connect to MCP server and discover tools."""
        if self._connected:
            return

        try:
            # Append /mcp to URL if not present (FastMCP convention)
            url = self.mcp_server_url
            if not url.endswith("/mcp"):
                url = f"{url}/mcp"

            self._client = Client(url)
            await self._client.__aenter__()
            self._connected = True

            # Discover available tools
            tools = await self._client.list_tools()
            self._tools = tools  # Store for later access
            tool_names = [t.name for t in tools]
            logger.info(f"Connected to MCP server. Available tools: {tool_names}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server at {self.mcp_server_url}: {e}")
            raise RuntimeError(f"Cannot connect to MCP server: {e}")

    def get_tools_for_llm(self) -> list[dict]:
        """
        Get MCP tools in OpenAI format for LiteLLM function calling.

        Note: We convert manually (not using LiteLLM native MCP) because we need
        to control tool execution to inject context_id for state isolation.

        Returns:
            list[dict]: Tools in OpenAI format for LiteLLM
        """
        if not hasattr(self, '_tools'):
            raise RuntimeError("Not connected to MCP server. Call connect() first.")

        # Convert MCP Tool schemas to OpenAI function calling format
        # MCP already uses JSON Schema for inputSchema, so this is straightforward
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in self._tools
        ]

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._client and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
                self._connected = False
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call any tool discovered from MCP server (generic, tool-agnostic).

        Args:
            tool_name: Name of the tool (from LLM decision)
            arguments: Tool arguments (from LLM)

        Returns:
            dict: Tool result from MCP server
        """
        if not self._connected:
            await self.connect()

        try:
            # Auto-inject context_id for state isolation
            arguments["context_id"] = self.context_id

            logger.debug(f"Calling MCP tool '{tool_name}' with args: {list(arguments.keys())}")

            # FIX: Wrap the call in a timeout to break the FastMCP stream lock (Bug #691)
            # In high-concurrency environments, zombie tasks (never finish, never error) are
            # more dangerous than crashes. Fail-fast principle ensures retry logic can trigger.
            try:
                result = await asyncio.wait_for(
                    self._client.call_tool(tool_name, arguments),
                    timeout=30.0  # Adjust based on expected tool latency
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout: FastMCP client hung on tool '{tool_name}' (StreamableHTTP bug)")
                return {"success": False, "result": "Client-side timeout/deadlock"}

            # Extract data from MCP result
            # FastMCP returns CallToolResult with .data attribute
            if hasattr(result, 'data'):
                return result.data
            else:
                # Fallback: parse from content blocks
                if hasattr(result, 'content') and result.content:
                    return {"result": result.content[0].text}
                return {"result": str(result)}

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return {
                "success": False,
                "result": f"MCP tool call failed: {str(e)}"
            }

    # === Context Manager ===

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False
