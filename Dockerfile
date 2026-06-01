FROM python:3.13

WORKDIR /app

COPY . .

RUN pip install uv

RUN uv sync

EXPOSE 8000




CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]