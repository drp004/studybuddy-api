import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

# youtube video process
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import assemblyai as aai
import yt_dlp
import os

# load environment variables
load_dotenv()

# get groq api key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY

# === message model === #
class State(TypedDict):
    messages: Annotated[list, add_messages]


# === tool to get youtube transcribe === #
def transcribe_yt(url):
    """ Use this tool only for getting youtube video transcribe """

    # # Parse the URL to extract its components
    # parsed_url = urlparse(url)

    # # Check if the URL contains a query string (e.g., ?v=abcd1234)
    # video_id = ""
    # if parsed_url.query:
    #     query_params = parse_qs(parsed_url.query)
    #     video_id = query_params.get("v", [None])[0]

    # # For shortened URLs (e.g., youtu.be/abcd1234)
    # elif parsed_url.netloc == "youtu.be":
    #     video_id = parsed_url.path[1:]  # Remove leading '/'
    #     print("Video id: ", video_id)

    # # Get Transcript   
    # if video_id: 
    #     transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])    
    #     text = ""
    #     for i in transcript:
    #         text += i['text']

    #     print("Transcribe: ", text)

    #     return text

    # else:
    #     return "Not getting video id video "
    # Step 1: Download audio from YouTube
    filename="audio.mp3"
    cookie = "cookie.txt"

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',  # downloads as .m4a (no conversion)
        'outtmpl': filename,
        "cookiefile": "cookies.txt",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Step 2: Transcribe with AssemblyAI
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filename)

    os.remove("audio.mp3")

    return transcript.text
    
# === define tools === #
tools = [transcribe_yt]

# initialize llm instance for chatbot
llm = ChatGroq(model="llama-3.3-70b-versatile")

# bind llm with tool
llm_with_tool = llm.bind_tools(tools)

# === Node Functionality === #
def chatbot(state: State):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}

# === Initialize graph builder === #
graph_builder = StateGraph(State)

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("tools", "chatbot")

# Compie graph
graph = graph_builder.compile()

# pre defined prompt
system_prompt = "You are a helpful AI assistant that answers clearly and politely. Be concise but informative."

if __name__ == "__main__":
    while True:
        user = str(input("You: "))

        if user in {"exit", "quit", "q"}:
            break

        else:
            response = graph.invoke({"messages": [HumanMessage(content=user)]})
            print("Agent: ", response["messages"][-1].content)