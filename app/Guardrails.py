from .llm import model
from .state import AgentState
from pydantic import BaseModel
from .prompt import Guardrails_prompt


class t_f(BaseModel):
    check: bool
    answer: str


structured_model = model().with_structured_output(t_f)


async def Guardrails_agent(state: AgentState):

    query = state["msg"][-1]

    result = await structured_model.ainvoke(
        [
            ("system", Guardrails_prompt),
            ("human", query.content),
        ]
    )

    print(result)
    print(type(result))

    return {
        "answer": result.answer,
        "check": result.check,
    }