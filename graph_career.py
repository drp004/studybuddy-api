from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
from typing import TypedDict, Annotated


load_dotenv()

# === message model === #
class State(TypedDict):
    messages: Annotated[list, add_messages]

# === tavily search object === #
tavily_search = TavilySearch(max_results=2)

# === tool for web search === #
def search(query):
    """
    use this tool only for do web search for trending job role and career opportunity
    """
    return tavily_search.invoke(query)

# === list of tools === #
tools = [search]

# === Initialize llm instance for infrance === #
llm = ChatGroq(model="llama-3.3-70b-versatile")

# === tool binding with llm === #
llm_with_tool = llm.bind_tools(tools)

# === Node functionality === #
def chatbot(state: State):
    return {"messages" : llm_with_tool.invoke(state["messages"])}

# === Initialize Graph Builder === #
graph_builder = StateGraph(State)

# add nodes 
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# compile graph
career_graph = graph_builder.compile()

# === System prompt === #
career_system_prompt = f"""
    Role: You are a professional career guide specializing in personalized career path development. The goal is to provide comprehensive, strategic career guidance tailored to an individual's unique professional profile.

    Task: Conduct a thorough analysis of the user's professional background, skills, interests, and career aspirations to develop a precise, actionable career roadmap that maximizes their potential and aligns with their personal and professional goals.

    Objective:
    Create a holistic career development plan that,
        1. Identifies optimal career trajectories
        2. Bridges current skills with desired job roles
        3. Provides strategic recommendations for skill enhancement
        4. Outlines realistic timelines for career progression

    Provide a structured career path recommendation that includes:
        1. Short-term (1-2 years) career goals
        2. Long-term (3-5 years) career objectives
        3. Specific skill development recommendations
        4. Potential certifications or training programs
        5. Networking and professional development strategies

    Deliver recommendations with:
        1. Clear, actionable guidance
        2. Realistic and achievable milestones
        3. Consideration of the user's personal constraints and opportunities
"""

if __name__ == "__main__":
    messages = [SystemMessage(career_system_prompt)]

    while True:
        user = input("User: ")

        if user in {"exit", "q", "quit"}:
            break
        else:
            messages.append(HumanMessage(content=user))
            response = career_graph.invoke({"messages": messages})
            print("Agent: ", response["messages"][-1].content)