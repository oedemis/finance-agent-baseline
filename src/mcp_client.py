"""
MCP Client for Purple Agent.

Connects to the green agent's MCP server to access financial research tools.
Auto-injects context_id for state isolation across concurrent tasks.
"""
import asyncio
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
    - Proper async context manager usage for FastMCP Client
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
        self._tools = None

        logger.info(f"FinanceToolsClient initialized: server={self.mcp_server_url}, context={self.context_id}")

    async def connect(self, force_reconnect=False):
        """Connect to MCP server and discover tools."""
        if self._client and not force_reconnect:
            return

        # Clean up old connection if forcing reconnect
        if force_reconnect and self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except:
                pass
            self._client = None

        try:
            # Append /mcp to URL if not present (FastMCP convention)
            url = self.mcp_server_url
            if not url.endswith("/mcp"):
                url = f"{url}/mcp"

            # Create Client - it will be used as context manager
            self._client = Client(url)

            # Enter the client context
            await self._client.__aenter__()

            # Discover available tools
            self._tools = await self._client.list_tools()
            tool_names = [t.name for t in self._tools]
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
        if not self._tools:
            raise RuntimeError("Not connected to MCP server. Call connect() first.")

        # Convert MCP Tool schemas to OpenAI function calling format
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
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
                self._client = None
                self._tools = None
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
        if not self._client:
            await self.connect()

        try:
            # Auto-inject context_id for state isolation
            arguments["context_id"] = self.context_id

            logger.debug(f"Calling MCP tool '{tool_name}' with args: {list(arguments.keys())}")

            # Call tool with timeout to prevent indefinite hangs
            # submit_answer runs LLM judges (3 judge calls) which can take 20-30s
            # Other tools complete in 2-3s typically
            timeout = 120.0 if tool_name == "submit_answer" else 60.0
            result = await asyncio.wait_for(
                self._client.call_tool(tool_name, arguments),
                timeout=timeout
            )

            # DEBUG: Log what we received
            logger.info(f"MCP tool '{tool_name}' RAW RESULT: type={type(result)}, has_data={hasattr(result, 'data')}, is_iterable={hasattr(result, '__iter__')}")

            # Extract data from result
            # FastMCP returns different formats depending on the tool
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                # If it's an iterable (like async generator), collect results
                logger.info(f"MCP tool '{tool_name}' is async iterable, collecting results...")
                results = []
                async for item in result:
                    logger.debug(f"Got item: type={type(item)}, has_data={hasattr(item, 'data')}")
                    if hasattr(item, 'data'):
                        results.append(item.data)
                    else:
                        results.append(item)
                logger.info(f"MCP tool '{tool_name}' collected {len(results)} items from async iterator")
                return {"success": True, "result": results[0] if len(results) == 1 else results}
            elif hasattr(result, 'data'):
                return result.data
            elif isinstance(result, dict):
                return result
            else:
                return {"result": str(result)}

        except asyncio.TimeoutError:
            timeout_val = 120.0 if tool_name == "submit_answer" else 60.0
            logger.error(f"Tool '{tool_name}' timed out after {timeout_val}s (client-side hang)")
            # Force reconnect after timeout
            try:
                await self.connect(force_reconnect=True)
                logger.info("Reconnected after timeout")
            except:
                pass
            return {
                "success": False,
                "error": f"Tool '{tool_name}' timed out - client did not receive response within 60s"
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool '{tool_name}' failed: {e}")

            # If client disconnected, try to reconnect
            if "not connected" in error_msg.lower() or "client is not" in error_msg.lower():
                try:
                    logger.info("Client disconnected, attempting reconnect...")
                    await self.connect(force_reconnect=True)
                    logger.info("Reconnected successfully")
                except Exception as reconnect_error:
                    logger.error(f"Reconnect failed: {reconnect_error}")

            return {
                "success": False,
                "error": f"MCP tool call failed: {error_msg}"
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
