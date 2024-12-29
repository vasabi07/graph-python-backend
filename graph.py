from dotenv import load_dotenv
load_dotenv()
import operator
from typing import Annotated,TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader

from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.graph import  END, START, StateGraph

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list,operator.add]

def web_search(state: State):
    """retrieve docs from web search"""
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state["question"])

    formatted_search_docs = []
    for doc in search_docs:
        if isinstance(doc, dict):  # Ensure the doc is a dictionary
            url = doc.get("url", "No URL available")
            content = doc.get("content", "No content available")
            formatted_search_docs.append(f'<Document href="{url}"/>\n{content}\n</Document>')
        else:
            formatted_search_docs.append(f"<Document>Error: Invalid document format</Document>")

    return {"context": ["\n\n----\n\n".join(formatted_search_docs)]}
    # return {"context": [search_docs]}


def wiki_search(state: State):
    """Retrieve docs from Wikipedia using LangChain's WikipediaLoader."""
    loader = WikipediaLoader(query=state["question"], load_max_docs=3)
    search_docs = loader.load()

    #Ensure response is handled correctly
    formatted_wiki_docs = []
    for doc in search_docs:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            title = doc.metadata.get("title", "No title available")
            content = doc.page_content or "No content available"
            formatted_wiki_docs.append(f'<Document href="{title}"/>\n{content}\n</Document>')
        else:
            formatted_wiki_docs.append("<Document>Error: Invalid document format</Document>")

    return {"context": ["\n\n----\n\n".join(formatted_wiki_docs)]}
    # return {"context": [search_docs]}

def generate_answer(state:State):
    context = state["context"]
    question = state["question"]

    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question,context=context)
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content="answer the question")])
    return {"answer": answer}

#graph
workflow = StateGraph(State)
workflow.add_node("web_search",web_search)
workflow.add_node("wiki_search",wiki_search)
workflow.add_node("generate_answer",generate_answer)

workflow.add_edge(START,"web_search")
workflow.add_edge(START,"wiki_search")
workflow.add_edge("web_search","generate_answer")
workflow.add_edge("wiki_search","generate_answer")
workflow.add_edge("generate_answer",END)

graph = workflow.compile()

# if __name__ == "__main__":
#     response = graph.invoke({"question": "who is dhoni? and give me his stats"})
#     print(response["answer"].content)
