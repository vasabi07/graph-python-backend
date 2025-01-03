from typing import Annotated, TypedDict, Literal

class InvalidUpdateError(Exception):
    """Custom exception for invalid updates in the supervisor node."""
    pass
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, START, END,MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel,Field
from langchain_community.tools.riza.command import ExecPython
import pprint
from langchain_groq import ChatGroq

tavily_tool = TavilySearchResults(max_results=3)
tool_code_interpreter = PythonREPL()
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
tools = [tavily_tool, tool_code_interpreter]
class Supervisor(BaseModel):
    next: Literal["enhancer","researcher","coder"] = Field(description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering, "
                    "'coder' for solving technical,math or code-related problems.")
    reason: str = Field(description="Reason for the decision, providing context on why a particular worker was chosen .")

system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and Coder Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focuse on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information.
3. Coder: Handles technical tasks related to calculation, coding, data analysis, and problem-solving, ensuring the correct implementation of solutions.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

Your goal is to maximize accuracy and effectiveness by leveraging each agent’s unique expertise while ensuring smooth workflow execution.
''')

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer","researcher","coder"]]:
   """
    Supervisor node for routing tasks based on the current state and LLM response.
    Args:
        state (MessagesState): The current state containing message history.
    Returns:
        Command: A command indicating the next state or action.
    """
   try:
       messages = [SystemMessage(content=system_prompt)] + state["messages"]
       response = llm.with_structured_output(Supervisor).invoke(messages)
       goto = response.next
       reason = response.reason
       return Command(
           update={
               "messages": [HumanMessage(content=reason,name = "supervisor")]
           },
           goto=goto
       )
   except Exception as e:
        raise InvalidUpdateError(f"Supervisor node encountered an error: {str(e)}")
   
def enhancer_node(state: MessagesState)-> Command[Literal["supervisor"]]:
    """
    Enhancer node for enhancing the user prompt
    Args:
        state (MessagesState): The current state of the workflow
    Returns:
        Command[Literal["supervisor"]]: The next worker to be assigned
    """
    system_prompt = (
        "You are an advanced query enhancer. Your task is to:\n"
        "Don't ask anything to the user, select the most appropriate prompt"
        "1. Clarify and refine user inputs.\n"
        "2. Identify any ambiguities in the query.\n"
        "3. Generate a more precise and actionable version of the original request.\n"
    )
    try:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        enhanced_query = llm.invoke(messages)

        return Command(
            update={
                "messages": [HumanMessage(content=enhanced_query.content, name="enhancer")]
            },
            goto="supervisor"

        )
    except Exception as e:      
        raise InvalidUpdateError(f"Enhancer node encountered an error: {str(e)}")
    
def researcher_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research node for leveraging a ReAct agent to process research-related tasks.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    research_agent = create_react_agent(llm,tools=[tavily_tool],state_modifier="You are a researcher. Do not worry about the code. Just focus on the research and do not do any math.")
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")]
        },
        goto="validator"
    )
def coder_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Coder node for leveraging a ReAct agent to process analyzing, solving math questions, and executing code.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    code_agent = create_react_agent(llm,tools=[],state_modifier="You are a coder and analyst. Focus on mathematical caluclations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks.")
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")]
        },
        goto="validator"
    )

# System prompt providing clear instructions to the validator agent
system_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

Routing Guidelines:
1. 'supervisor' Agent: For unclear or vague state messages and if someone says they cant do something
2. Respond with 'FINISH' to end the workflow.
'''

# Define a Validator class for structured output from the LLM
class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )


def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator node for checking if the question and the answer are appropriate.

    Args:
        state (MessagesState): The current state containing message history.

    Returns:
        Command: A command indicating whether to route back to the supervisor or end the workflow.
    """
    # Extract the first (user's question) and the last (agent's response) messages
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    # Prepare the message history with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    # Invoke the LLM with structured output using the Validator schema
    response = llm.with_structured_output(Validator).invoke(messages)

    # Extract the 'next' routing decision and the 'reason' from the response
    goto = response.next
    reason = response.reason



    # Determine the next node in the workflow
    if goto == "FINISH" or goto == END:
        goto = END  # Transition to the termination state
        print("Transitioning to END")  # Debug log to indicate process completion
    else:
        print(f"Current Node: Validator -> Goto: Supervisor")  # Log for routing back to supervisor
    # Debug logging to trace responses and transitions
    # print(f"Response: {response}")
    # Return a command with the updated state and the determined routing destination
    if goto == "supervisor":
        return Command(
            update={
                "messages": [
                    # Append the reason (validator's response) to the state, tagged with "validator"
                    HumanMessage(content=reason, name="validator")
                ]
            },
            goto="supervisor"
        )
    else:
        return Command(
            update={
                "messages": [
                    HumanMessage(content=agent_answer)
                ]
            },
            goto=END
        )

builder = StateGraph(MessagesState)
builder.add_node("supervisor",supervisor_node)
builder.add_node("enhancer",enhancer_node)
builder.add_node("researcher",researcher_node) 
builder.add_node("coder",coder_node)
builder.add_node("validator",validator_node)
builder.add_edge(START,"supervisor")
graph =builder.compile()

if __name__ == "__main__":
    
    inputs = {
    "messages": [
        ("user", "give me how many A's present in a string of AVYGABAAHKJHDAAAAUHBU  ?"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        print()









# repl = PythonREPL()

# def python_repl_tool(
#         code: Annotated[str, "Python code to run in the REPL"],
# ):
#     """use this to execute python code and do math. if you want to see the output of a value
#     you should print it out with `print(...)` This is visible to the user."""
#     try:
#         result = repl.run(code)
#     except Exception as e:
#         result = str(e)
#     result_str = f"successfully executed code: {code}\n\nresult: {result}"
#     return result_str

# members = ["researcher","coder"]
# options = members + ["FINISH"]

# system_prompt = (
#      "You are a supervisor tasked with managing a conversation between the"
#     f" following workers: {members}. Given the following user request,"
#     " respond with the worker to act next. Each worker will perform a"
#     " task and respond with their results and status. When finished,"
#     " respond with FINISH."
# )

# class Router(TypedDict):
#     next: Literal["researcher", "coder", "FINISH"]

# def supervisor_node(state: MessagesState) -> dict:
#     try:
#         systemMessage = SystemMessage(content=system_prompt)
#         messages = [systemMessage] + state["messages"]
#         response = llm.with_structured_output(Router).invoke(messages)
#         goto = response["next"]

#         if goto == "FINISH":
#             return {"status": "END"}
#         elif goto in ["researcher", "coder"]:
#             return {"status": "CONTINUE", "goto": goto}
#         else:
#             raise ValueError(f"Unexpected 'next' value: {goto}")
#     except Exception as e:
#         raise InvalidUpdateError(f"Supervisor node encountered an error: {str(e)}")



# research_agent = create_react_agent(llm,tools=[tavily_tool],state_modifier="You are a researcher. Do not worry about the code. Just focus on the research and do not do any math.")

# def researcher_node(state: MessagesState):
#     result = research_agent.invoke(state)
#     # Ensure result is properly handled and appended to messages
#     if "messages" in result and result["messages"]:
#         state["messages"].append(
#             HumanMessage(content=result["messages"][-1].content, name="researcher")
#         )
#     return {"next":"supervisor"}

# code_agent = create_react_agent(llm,tools=[python_repl_tool],state_modifier="You are a coder. You are tasked with writing code and doing math.")

# def coder_node(state: MessagesState):
#     result = code_agent.invoke(state)
#     # Ensure result is properly handled and appended to messages
#     if "messages" in result and result["messages"]:
#         state["messages"].append(
#             HumanMessage(content=result["messages"][-1].content, name="coder")
#         )
#     return {"next":"supervisor"}
# builder = StateGraph(MessagesState)
# builder.add_edge(START,"supervisor")
# builder.add_node("supervisor",supervisor_node)
# builder.add_node("researcher",researcher_node)
# builder.add_node("coder",coder_node)


# graph = builder.compile()

# if __name__ == "__main__":
#     # Initialize the MessagesState with a list of HumanMessage
#     initial_state = MessagesState(
#         messages=[HumanMessage(content="Find the latest GDP of New York and California, then calculate the average")]
#     )
    
#     # Invoke the graph with the correctly initialized state
#     response = graph.invoke(initial_state)
    
#     # Print the final response from the graph
#     print(response["messages"][-1].content)




