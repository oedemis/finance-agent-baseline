"""
Finance Agent Baseline - Purple agent for Finance Agent Benchmark.

This is a baseline agent that:
1. Receives financial research tasks from the Green Agent
2. Uses an LLM (via LiteLLM) to decide which tools to use
3. Returns tool calls in the expected format
4. Submits final answers when ready

Uses LiteLLM for LLM calls to support multiple providers.
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
    """Baseline agent that uses LiteLLM to answer financial research questions."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
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
            response = await litellm.acompletion(
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
            logger.error(f"Error calling LLM: {e}")
            return f'<json>{{"name": "submit_answer", "arguments": {{"answer": "Error: {str(e)}"}}}}</json>'

    def _get_system_messages(self) -> list[dict]:
        """Get system messages for the agent."""
        return [{
            "role": "system",
            "content": """You are a financial research agent. Today is December 24, 2025.

You will receive a question and available tools to research the answer.
You may not interact with the user directly.

RESPONSE FORMAT:

For tool calls, respond with <json>...</json> tags:
<json>
{
  "name": "tool_name",
  "arguments": {...}
}
</json>

When you have the final answer, respond with:

FINAL ANSWER: [Your comprehensive answer here, including all key facts, numbers, dates, and details]

{
    "sources": [
        {
            "url": "https://example.com",
            "name": "Name of the source"
        }
    ]
}

IMPORTANT:
- Follow the research requirements provided in the task description
- Use multiple sources to verify information
- Include specific numbers, dates, and details in your final answer
- Always provide sources with URLs and names"""
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
            agent = FinanceAgent(model=self.model, temperature=self.temperature)
            self.agents[context_id] = agent

        # Check if this is a new conversation
        is_new = len(agent.conversation_history) == 0
        message_text = get_message_text(msg)

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            response = await agent.process_message(message_text, new_conversation=is_new)

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response))],
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
