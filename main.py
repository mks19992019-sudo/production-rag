import os
import time
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(send_to_logfire="if-token-present", service_name="agentic-rag")

from fastapi import FastAPI
from src.graph import get_workflow
from pydantic import BaseModel
from src.graph import initialize_resources , close_graph_resources , get_checkpointer
from src.dataInjection.data_injection import initialize_vectorstore


from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI):
    with logfire.span("FastAPI application startup",):
        logfire.info("Starting up FastAPI application and initializing resources...")
        await initialize_vectorstore()
        await initialize_resources()
    try:
        yield
    finally:
        with logfire.span("FastAPI application shutdown",):
            logfire.info("Shutting down FastAPI application and closing resources...")
            await close_graph_resources()


app = FastAPI(title='Rag_Agent',
              description='Production rag ',
              lifespan=lifespan)

logfire.instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your specific frontend URL like "http://localhost:5173"
    allow_credentials=False,
    allow_methods=["*"],   # This is what allows the OPTIONS preflight
    allow_headers=["*"],
)

class user_msg(BaseModel):
    msg: str
    thread_id : str

@app.get("/")
def check():
    return "Fast api is working"

@app.post("/chat")
async def chat(query:user_msg):

    user_msg= query.msg
    thread_id = query.thread_id

    try:
        workflow = await get_workflow()
        result = await workflow.ainvoke(
            {'msg': user_msg},
            {'configurable': {'thread_id': thread_id}},
        )

        # use result['msg'][-1].content
        if result['check'] is False:

            return {result['answer']}

        final_result = result['msg'][-1].content




        return final_result
    except Exception:
        
        logfire.exception("Chat request failed", thread_id=thread_id)

       




