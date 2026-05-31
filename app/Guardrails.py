from .llm import model
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from .state import AgentState
from pydantic import BaseModel
from .prompt import Guardrails_prompt
from typing import Optional

class t_f (BaseModel):
    check : bool
    answer : str



struture_model = model().with_structured_output(t_f)


react_agent = create_agent(
    system_prompt=Guardrails_prompt,
    model=struture_model,
    middleware=[
        PIIMiddleware(
            "email",
            strategy='redact',
            apply_to_input=True
        ),
        PIIMiddleware(
            'credit_card',
            apply_to_input=True,
            strategy='mask'

        ),
        PIIMiddleware(
            'ip',
            strategy='hash',
            apply_to_input=True
        ),PIIMiddleware(
            'mac_address',
            apply_to_input=True,
            strategy='mask'
        )
    ],

    
)



async def Guardrails_agent (state:AgentState):
    query = state['msg'][-1]
    result = await react_agent.ainvoke({'messages':query})
    return {'answer':result['answer'],
            'check':result['check']

            
            }
    













