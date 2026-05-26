from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults



@tool
async def webserch(search)->str:
    ''' use it when u need more information or stuck in converstion '''

    return await DuckDuckGoSearchResults().ainvoke(search)