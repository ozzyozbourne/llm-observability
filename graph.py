from dotenv import load_dotenv
from typing import List 
from typing_extensions import TypedDict

from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

from langfuse.langchain import CallbackHandler
from langfuse import observe

from langsmith import traceable

_ = load_dotenv(dotenv_path=".env", override=True)
web_search_tool = TavilySearch(max_retries=1)
langfuse_handler = CallbackHandler()
llm = ChatOpenAI(model_name="x-ai/grok-4-fast:free", temperature=0)

prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand.
Your job is to answer the provided question so that even a 5 year old can understand it.
You have been provided with the relevant background context to answer the quesion.

Question: {question}

Context: {context}

Answer:"""

class InputState(TypedDict): 
    question: str
    
class GraphState(TypedDict): 
    question: str
    documents: List[str]
    messages: List[str]

@traceable
@observe()  
def search(state):
   question = state["question"] 
   documents = state.get("documents", [])
   
   web_docs = web_search_tool.invoke({"query": question})
   web_results = "\n".join(d["content"] for d in web_docs["results"])
   web_results = Document(page_content=web_results)
   documents.append(web_results)
   
   return {"documents": documents, "question": question}

@traceable
@observe() 
def explain(state: GraphState):
    question = state["question"]
    documents = state.get("documents", [])
    formatted = prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}
    

graph = StateGraph(GraphState, input_schema=InputState)
graph.add_node("explain", explain)
graph.add_node("search", search)
graph.add_edge(START, "search")
graph.add_edge("search", "explain")
graph.add_edge("explain", END)

eli5_working = graph.compile()

buggy_prompt = """You are a professor and expert in complex technical communication.
Your job is to answer the provided question as precisely as possible, using technical language with maximal detail. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""

@traceable
@observe()
def buggy_explain(state: GraphState):

    question = state["question"]
    documents = state.get("documents", [])
    formatted = buggy_prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}

buggy_graph = StateGraph(GraphState, input_schema=InputState)
buggy_graph.add_node("explain", buggy_explain)
buggy_graph.add_node("search", search)
buggy_graph.add_edge(START, "search")
buggy_graph.add_edge("search", "explain")
buggy_graph.add_edge("explain", END)

eli5_buggy = buggy_graph.compile()

flaky_prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
You must use the provided context to answer the question. If no context is available, refuse to answer the question to avoid hallucination.

Question: {question} 

Context: {context}

Answer:"""

@traceable
@observe()
def flaky_explain(state: GraphState):
    
    question = state["question"]
    documents = state.get("documents", [])
    formatted = flaky_prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = llm.invoke([HumanMessage(content=formatted)])
    return {"question": question, "messages": [generation]}

@traceable
@observe()
def flaky_search(state):
    
    question = state["question"]
    documents = state.get("documents", [])

    if "economics" in question:
        web_results = "No results found."
    else:
        web_docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in web_docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

flaky_graph = StateGraph(GraphState, input_schema=InputState)
flaky_graph.add_node("explain", flaky_explain)
flaky_graph.add_node("search", flaky_search)
flaky_graph.add_edge(START, "search")
flaky_graph.add_edge("search", "explain")
flaky_graph.add_edge("explain", END)

eli5_flaky = flaky_graph.compile()

if __name__ == "__main__":
    # Test the working graph
    print("=== Working Graph ===")
    result = eli5_working.invoke({"question": "What is photosynthesis?"},
    config={"callbacks": [langfuse_handler]})
    print(result["messages"][0].content)
    print("\n")
    
    # Test the buggy graph (uses technical language)
    print("=== Buggy Graph ===")
    result = eli5_buggy.invoke({"question": "What is photosynthesis?"},
    config={"callbacks": [langfuse_handler]})
    print(result["messages"][0].content)
    print("\n")
    
    # Test the flaky graph (will fail for economics questions)
    print("=== Flaky Graph ===")
    result = eli5_flaky.invoke({"question": "What is supply and demand in economics?"},
    config={"callbacks": [langfuse_handler]})
    print(result["messages"][0].content)