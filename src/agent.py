"""
Finance Agent Baseline - Purple agent for Finance Agent Benchmark.

This is a baseline agent that:
1. Receives financial research tasks from the Green Agent
2. Uses an LLM to decide which tools to use
3. Returns tool calls in the expected format
4. Submits final answers when ready
"""
import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any

import uvicorn
from dotenv import load_dotenv
from openai import AsyncOpenAI
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

load_dotenv()

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
)
from a2a.utils import new_agent_text_message, new_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_agent")


class FinanceAgent:
    """Baseline agent that uses OpenAI to answer financial research questions."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.conversation_history: list[dict] = []

    async def process_message(self, message: str, new_conversation: bool = False) -> str:
        """Process a message and return the response."""
        if new_conversation:
            self.conversation_history = []

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._get_system_messages() + self.conversation_history,
                temperature=self.temperature,
                max_tokens=2000,
            )

            assistant_message = response.choices[0].message.content

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling OpenAI: {e}")
            return f"<json>{{\"name\": \"submit_answer\", \"arguments\": {{\"answer\": \"Error: {str(e)}\"}}}}</json>"

    def _get_system_messages(self) -> list[dict]:
        """Get system messages for the agent."""
        return [{
            "role": "system",
            "content": """You are a financial research agent. You must respond with a JSON object wrapped in <json>...</json> tags.

IMPORTANT RULES:
1. Always respond with exactly ONE tool call at a time
2. Use the tools provided to gather information
3. When you have enough information, use submit_answer to provide your final answer
4. Be thorough but efficient - avoid redundant tool calls

RESPONSE FORMAT:
<json>
{
  "name": "tool_name",
  "arguments": {...}
}
</json>

AVAILABLE TOOLS:
- edgar_search: Search SEC EDGAR for filings
- google_web_search: Search the web
- parse_html_page: Parse and store webpage content
- retrieve_information: Extract info from stored documents (use {{key}} placeholders)
- submit_answer: Submit your final answer

EXAMPLE RESPONSES:
<json>
{"name": "edgar_search", "arguments": {"query": "quarterly revenue", "form_types": ["10-Q"], "ciks": [], "start_date": "2024-01-01", "end_date": "2024-12-31", "page": "1", "top_n_results": 5}}
</json>

<json>
{"name": "google_web_search", "arguments": {"search_query": "Apple Inc Q4 2024 earnings"}}
</json>

<json>
{"name": "submit_answer", "arguments": {"answer": "Based on my research..."}}
</json>

Always think step by step about what information you need and which tools will help you find it."""
        }]


class FinanceAgentExecutor(AgentExecutor):
    """Executor that wraps the FinanceAgent for A2A protocol."""

    def __init__(self, agent: FinanceAgent):
        self.agent = agent
        self.conversations: dict[str, bool] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.get_user_input()
        context_id = context.context_id

        # Check if this is a new conversation
        is_new = context_id not in self.conversations
        self.conversations[context_id] = True

        # Create task
        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            return

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing...", context_id=context_id)
        )

        try:
            response = await self.agent.process_message(message, new_conversation=is_new)

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response))],
                name="Response",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"Agent error: {e}")
            await updater.failed(
                new_agent_text_message(f"Error: {e}", context_id=context_id)
            )

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        return None


def finance_agent_card(name: str, url: str) -> AgentCard:
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
        name=name,
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


async def main():
    parser = argparse.ArgumentParser(description="Run the Finance Agent Baseline.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    # Override from environment
    model = os.getenv("AGENT_MODEL", args.model)
    temperature = float(os.getenv("AGENT_TEMPERATURE", args.temperature))

    agent = FinanceAgent(model=model, temperature=temperature)
    executor = FinanceAgentExecutor(agent)
    agent_card = finance_agent_card("FinanceAgentBaseline", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
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

    uvicorn_config = uvicorn.Config(app, host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)

    logger.info(f"Starting Finance Agent Baseline on {args.host}:{args.port}")
    logger.info(f"Using model: {model}, temperature: {temperature}")
    logger.info(f"Health check: http://{args.host}:{args.port}/health")

    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
