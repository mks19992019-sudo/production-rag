from fastapi import FastAPI
from src.graph import get_workflow
from pydantic import BaseModel
from src.graph import initialize_resources , close_graph_resources , get_checkpointer
from contextlib import asynccontextmanager
from src.dataInjection.data_injection import initialize_vectorstore
from redis.asyncio import Redis
from dotenv import load_dotenv
load_dotenv()
import os
import time ,logfire

from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(_: FastAPI):
    await initialize_vectorstore()
    await initialize_resources()
    try:
        yield
    finally:
        await close_graph_resources()


app = FastAPI(title='Rag_Agent',
              description='Production rag ',
              lifespan=lifespan)

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
    start_time = time.time()
    user_msg= query.msg
    thread_id = query.thread_id

   
    
    workflow =  await get_workflow()
    result = await workflow.ainvoke({'msg':user_msg},{'configurable':{'thread_id':thread_id}})

    # use result['msg'][-1].content
    if result['check'] is False:
        logfire.info("No relevant documents found.")
        return {result['answer']}

    final_result = result['msg'][-1].content

    end_time = time.time()
    logfire.info(f"Response generated in {end_time - start_time:.2f} seconds.")

    return final_result




