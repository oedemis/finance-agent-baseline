# Finance Agent Baseline (Purple Agent)

**Version:** 1.0.0
**Type:** Very dumy Purple Agent (Participant)
**Framework:** A2A Protocol, MCP

## Overview

The Finance Agent Baseline is a purple agent that participates in the Finance Agent Benchmark evaluation. It receives financial research questions from the green agent, uses provided tools to research information, and submits answers.

This baseline agent serves as:
- **Reference implementation** for other purple agents
- **Baseline for comparison** on the leaderboard
- **Example** of how to build agents for the benchmark

## Features

- **A2A Protocol**: Communicates with green agent via A2A
- **Tool Usage via MCP**: Calls EDGAR search, Google search, HTML parsing, etc.
- **LLM-Powered**: Uses LLM for decision-making
- **Simple Strategy**: Straightforward research approach for baseline performance

## Installation

### Prerequisites

- Python 3.10+
- uv (recommended) or pip

### Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# For development
uv sync --extra dev
```

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_key_here

# Optional: Model configuration
AGENT_MODEL=gpt-4o
AGENT_TEMPERATURE=0
```

## Usage

### Run Locally

```bash
uv run python src/agent.py --host 0.0.0.0 --port 9019
```

### Run with Docker

```bash
docker build -t finance-agent-baseline:v1.0 .
docker run -p 9019:9019 --env-file .env finance-agent-baseline:v1.0
```

### Test with Green Agent localy

See `../FAB/scenario.toml` for configuration.

```bash
cd ../FAB
docker-compose up
```

## How It Works

via MCP

## License

TBD

## References

- [AgentBeats Documentation](https://docs.agentbeats.dev/)
- [A2A Protocol](https://a2a-protocol.org/)
