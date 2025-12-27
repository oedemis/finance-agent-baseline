"""
Finance Agent Baseline - Purple agent for Finance Agent Benchmark.

This is a baseline agent that:
1. Receives financial research tasks from the Green Agent
2. Uses an LLM (via LiteLLM) to decide which tools to use
3. Calls tools directly via MCP (Model Context Protocol)
4. Submits final answers when ready

Uses LiteLLM for LLM calls to support multiple providers.
Uses MCP for tool access (no text parsing required).
"""
import argparse
import json
import logging
import os

import litellm
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

from mcp_client import FinanceToolsClient

from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils import new_agent_text_message, new_task, get_message_text
from a2a.utils.errors import ServerError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_agent")


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class FinanceAgent:
    """Baseline agent that uses LiteLLM and MCP to answer financial research questions."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, context_id: str = "default"):
        self.model = model
        self.temperature = temperature
        self.context_id = context_id
        self.conversation_history: list[dict] = []
        self.mcp_client: FinanceToolsClient | None = None

    async def _ensure_mcp_connected(self):
        """Connect to MCP server if not already connected."""
        if not self.mcp_client:
            self.mcp_client = FinanceToolsClient(context_id=self.context_id)
            await self.mcp_client.connect()
            logger.info(f"Connected to MCP server for context {self.context_id}")

    async def process_message(self, message: str, new_conversation: bool = False) -> tuple[str, dict | None]:
        """
        Process a message by looping internally with LLM and MCP tools.

        Returns:
            tuple: (status_message, final_answer_data)
                - status_message: Human-readable completion message
                - final_answer_data: Structured answer data if submit_answer was called, None otherwise
        """
        await self._ensure_mcp_connected()

        if new_conversation:
            self.conversation_history = []

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Loop until submit_answer is called
        max_iterations = 50  # Safety limit to prevent infinite loops

        for iteration in range(max_iterations):
            try:
                # Get LLM response with function calling (tools from MCP server)
                response = await litellm.acompletion(
                    model=self.model,
                    messages=self._get_system_messages() + self.conversation_history,
                    temperature=self.temperature,
                    max_tokens=512,
                    tools=self.mcp_client.get_tools_for_llm(),  # Dynamic from MCP!
                    tool_choice="auto",
                    parallel_tool_calls=False,  # Process one tool at a time
                )

                assistant_message = response.choices[0].message

                # Add assistant response to history (use model_dump() to preserve exact format)
                message_dict = {
                    "role": "assistant",
                    "content": assistant_message.content
                }

                # Add tool_calls if present (preserve exact LiteLLM format)
                if assistant_message.tool_calls:
                    message_dict["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, 'model_dump') else tc
                        for tc in assistant_message.tool_calls
                    ]

                self.conversation_history.append(message_dict)

                tool_calls = assistant_message.tool_calls

                logger.debug(f"Iteration {iteration + 1}: content={assistant_message.content[:100] if assistant_message.content else '(no content)'}, tool_calls={len(tool_calls) if tool_calls else 0}")

                # Check if LLM called any tools
                if not tool_calls:
                    # No tool call - LLM might be reasoning or stuck
                    logger.warning(f"No tool call found in iteration {iteration + 1}")
                    # Add a nudge to use tools
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Please use one of the available tools to continue your research, or call submit_answer if you're ready."
                    })
                    continue

                # Process first tool call (we don't support parallel calls yet)
                tool_call = tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"Calling tool: {tool_name}")

                # Call tool via MCP - catch errors to ensure we always add tool response
                # Add timeout to prevent FastMCP StreamableHTTP hanging bug (Issue #691)
                try:
                    result = await asyncio.wait_for(
                        self._call_tool_via_mcp(tool_name, tool_args),
                        timeout=30.0  # 30 second timeout per tool call
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Tool {tool_name} timed out after 30s (FastMCP hanging bug)")
                    result = {"success": False, "error": f"Tool {tool_name} timed out after 30 seconds"}
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    result = {"success": False, "error": str(e)}

                # Check if this was submit_answer (task complete)
                if tool_name == "submit_answer" and result.get("success", True):
                    logger.info("Task complete - submit_answer called")
                    # Return both status message and structured answer data
                    return (
                        f"Research complete. Final answer submitted.\n\nIterations: {iteration + 1}",
                        {
                            "answer": tool_args.get("answer", ""),
                            "sources": tool_args.get("sources", [])
                        }
                    )

                # ALWAYS add tool result to conversation (even if it failed)
                # This prevents tool_call_id mismatch errors
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })

            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                # Try to submit error as final answer
                try:
                    error_answer = f"Error during research: {str(e)}"
                    await self.mcp_client.submit_answer(
                        answer=error_answer,
                        sources=[]
                    )
                    return (
                        f"Error occurred, submitted error answer: {e}",
                        {"answer": error_answer, "sources": []}
                    )
                except:
                    return (f"Critical error: {e}", None)

        # Max iterations reached without submit_answer
        logger.warning(f"Max iterations ({max_iterations}) reached without submitting answer")
        try:
            incomplete_answer = "Unable to complete research within iteration limit."
            await self.mcp_client.submit_answer(
                answer=incomplete_answer,
                sources=[]
            )
            return (
                f"Max iterations reached. Submitted incomplete answer.",
                {"answer": incomplete_answer, "sources": []}
            )
        except:
            return ("Max iterations reached without submitting answer", None)

    async def _call_tool_via_mcp(self, tool_name: str, tool_args: dict) -> dict:
        """Call any tool generically via MCP (tool-agnostic)."""
        try:
            # Generic tool execution - works with ANY tool discovered from MCP
            return await self.mcp_client.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "success": False,
                "result": f"Tool execution error: {str(e)}"
            }

    def _get_system_messages(self) -> list[dict]:
        """Get system messages for the agent."""
        return [{
            "role": "system",
            "content": """You are a financial research agent. Today is December 27, 2025.

**Available Tools:**
- edgar_search: Search SEC filings
- google_web_search: Search the web
- parse_html_page: Parse and store webpage content
- retrieve_information: Analyze stored content with LLM
- submit_answer: Submit your final answer (REQUIRED to complete task)

**Workflow:**
1. Search for relevant information (edgar_search or google_web_search)
2. Parse important webpages (parse_html_page)
3. Analyze the content (retrieve_information)
4. **Always call submit_answer with your final answer and sources**

**Important:** You MUST call submit_answer when you have enough information to answer the question. Include all key facts, numbers, dates, and details. Be specific and factual."""
        }]


class FinanceAgentExecutor(AgentExecutor):
    """Executor that wraps the FinanceAgent for A2A protocol."""

    def __init__(self, model: str = "openai/gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.agents: dict[str, FinanceAgent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = FinanceAgent(
                model=self.model,
                temperature=self.temperature,
                context_id=context_id  # Pass context_id for MCP state isolation
            )
            self.agents[context_id] = agent

        # Check if this is a new conversation
        is_new = len(agent.conversation_history) == 0
        message_text = get_message_text(msg)

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            status_message, answer_data = await agent.process_message(message_text, new_conversation=is_new)

            # Create parts for the artifact
            parts = [Part(root=TextPart(text=status_message))]

            # If we have structured answer data, include it as DataPart
            if answer_data:
                from a2a.types import DataPart
                parts.append(Part(root=DataPart(data=answer_data)))

            await updater.add_artifact(
                parts=parts,
                name="Response",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"Agent error: {e}")
            await updater.failed(
                new_agent_text_message(f"Error: {e}", context_id=context_id, task_id=task.id)
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())


def create_agent_card(url: str) -> AgentCard:
    """Create the agent card for the finance agent."""
    skill = AgentSkill(
        id="finance_research",
        name="Financial Research",
        description="Researches financial questions using SEC filings and web search",
        tags=["finance", "research", "sec", "edgar"],
        examples=[
            "What was Apple's revenue in Q4 2024?",
            "Who is the CFO of Microsoft?",
        ],
    )
    return AgentCard(
        name="FinanceAgentBaseline",
        description="Baseline financial research agent for Finance Agent Benchmark",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "finance-agent-baseline",
        "version": "1.0.0"
    })


def main():
    parser = argparse.ArgumentParser(description="Run the Finance Agent Baseline.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    # Override from environment
    model = os.getenv("AGENT_MODEL", args.model)
    temperature = float(os.getenv("AGENT_TEMPERATURE", args.temperature))

    agent_card = create_agent_card(agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=FinanceAgentExecutor(model=model, temperature=temperature),
        task_store=InMemoryTaskStore(),
    )

    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    a2a_app = a2a_server.build()

    # Create wrapper app with health check
    routes = [
        Route("/health", health_check, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    app.mount("/", a2a_app)

    logger.info(f"Starting Finance Agent Baseline on {args.host}:{args.port}")
    logger.info(f"Using model: {model}, temperature: {temperature}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
