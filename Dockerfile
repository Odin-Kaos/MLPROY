# Use a lightweight Python base image
FROM python:3.11-slim

# Install uv (fast dependency manager)
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./
COPY models ./

# Install dependencies using uv
RUN uv sync --frozen

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uv", "run", "uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
