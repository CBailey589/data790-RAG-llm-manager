
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Copy code
COPY main.py ./
COPY static/ ./static/

# Copy corpus
COPY chroma_db_nintendo_power/ ./chroma_db_nintendo_power/

# FastAPI listens on port 8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
