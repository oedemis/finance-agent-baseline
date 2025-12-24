FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN uv pip install --system -e .

# Expose A2A port
EXPOSE 9019

# Set entrypoint
ENTRYPOINT ["python", "src/agent.py"]

# Default arguments
CMD ["--host", "0.0.0.0", "--port", "9019"]
