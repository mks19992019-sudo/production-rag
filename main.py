from fastapi import FastAPI
from app.graph import get_workflow
from pydantic import BaseModel
from app.graph import initialize_resources , close_graph_resources , get_checkpointer
from contextlib import asynccontextmanager
from app.extractor import initialize_vectorstore
from redis.asyncio import Redis
from dotenv import load_dotenv
load_dotenv()
import os



redis_client = Redis(
    host=os.getenv("REDIS_URL")

)


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


class user_msg(BaseModel):
    msg: str
    thread_id : str



@app.get("/")
def check():
    return "Fast api is working"


# crypto.randomUUID() #thread id come from frontend 


# the main problem is about memoery of every thread_id 


@app.post("/chat")
async def chat(query:user_msg):
    user_msg= query.msg
    thread_id = query.thread_id

   
    
    workflow =  await get_workflow()
    result = await workflow.ainvoke({'msg':user_msg,'context':''},{'configurable':{'thread_id':thread_id}})


    final_result = result['msg']

    return final_result[-1].content




