from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import observe
from langfuse.openai import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

_ = load_dotenv(dotenv_path=".env", override=True)
web_search_tool = TavilySearch(max_results=1)
langfuse_handler = CallbackHandler()   
openai_client = wrap_openai(openai.OpenAI())

prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand.
Your job is to answer the provided question so that even a 5 year old can understand it.
You have been provided with the relevant background context to answer the quesion.

Question: {question}

Context: {context}

Answer:"""

@traceable
@observe()
def search(question): 
    web_docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_docs["results"]])
    return web_results

@traceable
@observe()
def explain(question, context):
    formatted = prompt.format(question=question, context=context)
    
    completion = openai_client.chat.completions.create(
        messages = [
            {"role": "system", "content": formatted},
            {"role": "user", "content": question}
        ],
        model="x-ai/grok-4-fast:free"
    )
    return completion.choices[0].message.content
    
    
@traceable
@observe()
def eli5(question):
    context = search(question)
    answer = explain(question, context)
    return answer

print(eli5("What is trustcall?"))