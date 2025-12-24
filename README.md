# Finance Agent Baseline (Purple Agent)

**Version:** 1.0.0
**Type:** Purple Agent (Participant)
**Framework:** A2A Protocol

## Overview

The Finance Agent Baseline is a purple agent that participates in the Finance Agent Benchmark evaluation. It receives financial research questions from the green agent, uses provided tools to research information, and submits answers.

This baseline agent serves as:
- **Reference implementation** for other purple agents
- **Baseline for comparison** on the leaderboard
- **Example** of how to build agents for the benchmark

## Features

- **A2A Protocol**: Communicates with green agent via A2A
- **Tool Usage**: Calls EDGAR search, Google search, HTML parsing, etc.
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

### Test with Green Agent

See `../FAB/scenario.toml` for configuration.

```bash
cd ../FAB
docker-compose up
```

## How It Works

### 1. Receive Task

The green agent sends a question via A2A:

```
Question: What was RTX Corp's revenue in 2024?

Available tools:
- search_edgar
- search_google
- parse_html
- retrieve_information
- submit_answer

Respond with tool calls in JSON format wrapped in <json>...</json>
```

### 2. Research Strategy

The baseline agent:
1. Analyzes the question
2. Decides which tool to use
3. Calls tool by returning JSON:
   ```json
   <json>
   {
     "name": "search_edgar",
     "arguments": {
       "query": "RTX revenue",
       "form_types": ["10-K"],
       ...
     }
   }
   </json>
   ```
4. Receives tool result from green agent
5. Continues research or submits answer

### 3. Submit Answer

When ready:
```json
<json>
{
  "name": "submit_answer",
  "arguments": {
    "answer": "RTX Corp's revenue in 2024 was $70.3 billion..."
  }
}
</json>
```

## Architecture

```
Purple Agent Components:
├─ Agent Executor    # Handles A2A communication
├─ LLM Client        # Makes tool call decisions
└─ Message Parser    # Parses tool definitions and results
```

## Development

### Project Structure

```
finance-agent-baseline/
├── src/
│   ├── agent.py          # Main purple agent
│   ├── executor.py       # A2A executor
│   └── llm_client.py     # LLM interaction
├── tests/
│   └── test_agent.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format
uv run black src/

# Lint
uv run ruff check src/
```

## Performance

This baseline agent achieves:
- **~25-35% accuracy** (estimated)
- Simple strategy, room for improvement

## Improving the Agent

To build a better agent:

1. **Better Tool Selection**
   - Smarter decision making on which tools to use
   - Learn from previous tool calls

2. **Multi-Step Reasoning**
   - Plan research strategy upfront
   - Verify information from multiple sources

3. **Error Recovery**
   - Handle tool failures gracefully
   - Retry with different strategies

4. **Efficiency**
   - Minimize redundant tool calls
   - Optimize for cost and time

5. **Answer Quality**
   - Better information synthesis
   - Proper citation of sources
   - Clear, structured responses

## API Reference

### A2A Endpoint

**POST** `/agent`

Receives task from green agent, returns tool calls or final answer.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Test against benchmark
5. Submit pull request

## License

TBD

## References

- [Design Document](../FAB/docs/design.md)
- [Green Agent Repo](../finance-agent-evaluator/)
- [AgentBeats Documentation](https://docs.agentbeats.dev/)
- [A2A Protocol](https://a2a-protocol.org/)

## Contact

For questions or issues, please open an issue in the repository.
