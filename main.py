from fastapi import FastAPI
from app.graph import get_workflow
from pydantic import BaseModel

app = FastAPI()




class user_msg(BaseModel):
    msg: str




@app.get("/")
def check():
    return "Fast api is working"




@app.post("/chat")
async def chat(query:user_msg):

    user_msg= query.msg
    workflow =  await get_workflow()
    result = await workflow.ainvoke({'msg':user_msg,'context':''},{'configurable':{'thread_id':1}})


    final_result = result['msg']

    return final_result[-1].content




