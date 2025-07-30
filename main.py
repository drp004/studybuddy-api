from fastapi import FastAPI, UploadFile, File, Form
import json
from pydantic import BaseModel
from typing import List, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage

import traceback
import logging

# logger for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# for pdf process
import fitz

# for image process
import cv2
import pytesseract
import numpy as np

# youtube video process
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

# audio process
from faster_whisper import WhisperModel
import tempfile

# import graph from graph_builder
from graph_builder import graph
from graph_career import career_graph, career_system_prompt 

# FastAPI app
app = FastAPI()

# your system prompt
system_prompt = """
    Role:- You are a Professional Note-Taking Specialist with expertise in transforming complex text into clear, structured, and easily digestible notes. Your core strengths include:

        Advanced information synthesis
        Ability to distill key concepts
        Precision in capturing essential details
        Creating organized and readable documentation

    Task:- Generate high-quality, standard notes from any provided text by:

        Extracting the most critical information
        Organizing content into a logical, hierarchical structure
        Using clear and concise language
        Ensuring notes are comprehensible and actionable
    
    Instructions:- Note Generation Process:

        Analyze the entire text for core themes and key points
        Use markdown formatting for clear visual hierarchy
        Structure notes with:
        Main headings
        Subheadings
        Bullet points
        Concise explanations

    Additional Guidelines:
        You are strickly not allowed to call any tool for this to extract the text from PDF or Image it will be handlede explicitly
"""

class ChatHistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem]

class Yt_link(BaseModel):
    yt_link: str


# === Pdf proccesing tool === #
async def extract_pdf_text(file: UploadFile) -> str:
    pdf_bytes = file.file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])

    return text

# === image processing tool === #
async def extract_image_text(image_bytes):
    """
    Process image bytes and extract text
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image. Make sure it's a valid image file.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}, \ntraceback: {traceback.format_exc()}")
    
# === YouTube video processor === #
async def transcribe_yt(url):
    # Parse the URL to extract its components
    parsed_url = urlparse(url)

    # Check if the URL contains a query string (e.g., ?v=abcd1234)
    video_id = ""
    if parsed_url.query:
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v", [None])[0]

    # For shortened URLs (e.g., youtu.be/abcd1234)
    elif parsed_url.netloc == "youtu.be":
        video_id = parsed_url.path[1:]  # Remove leading '/'

    # Get Transcript   
    if video_id: 
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])    
        text = ""
        for i in transcript:
            text += i['text']

        return text

    else:
        return "Not getting video id video "

# === Audio processing === #

# load the model once
model = WhisperModel("tiny", device="cpu", compute_type="int8")
async def transcribe_audio_file(audio_path: str) -> str:
    segments, info = model.transcribe(audio_path)

    transcript = ""
    for seg in segments:
        transcript += seg.text.strip() + " "
        
    return transcript.strip()


# === Routes === #

# === Route for pdf note generation === #
@app.post("/studybuddy/process-pdf")
async def process_pdf(req: str = Form(...), pdf: UploadFile = File(None)):

    # Parse JSON string
    req_dict = json.loads(req)
    req_obj = ChatRequest(**req_dict)

    # extract text from pdf
    if pdf:
        pdf_text = await extract_pdf_text(pdf)
        logging.info(f"Extracted text length: {len(pdf_text)}, \nTraceback: {traceback.format_exc()}")

    # Proceed as before…
    messages = [SystemMessage(content=system_prompt)]

    for msg in req_obj.history:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            messages.append(AIMessage(content=msg.content))

    # Add latest human message
    messages.append(HumanMessage(content= f"Here is text, generate notes from this: \n{pdf_text}"))

    # Run LangGraph
    state = {"messages": messages}
    result = graph.invoke(state)

    # Get bot reply
    bot_reply = result["messages"][-1].content

    # Convert full message history to dicts for response
    response_history = [
        {"role": msg.type, "content": msg.content}
        for msg in result["messages"]
    ]

    return {
        "reply": bot_reply,
        "history": response_history
    }

# === Route for image note generation === #
@app.post("/studybuddy/process-image")
async def process_image(req: str = Form(...) , image: UploadFile = File(...)):

    # Parse JSON string
    req_dict = json.loads(req)
    req_obj = ChatRequest(**req_dict)

    # Read the bytes
    image_bytes = await image.read()

    # Call processing
    image_text = await extract_image_text(image_bytes)

    if image_text:
        logging.info(f"Extracted text length: {len(image_text)}")

    # Proceed as before…
    messages = [SystemMessage(content=system_prompt)]

    for msg in req_obj.history:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            messages.append(AIMessage(content=msg.content))

    # Add latest human message
    messages.append(HumanMessage(content=  f"{req_obj.message}; Here is text extracted from image, generate notes from this: \n{image_text}"))

    # Run LangGraph
    state = {"messages": messages}
    result = graph.invoke(state)

    # Get bot reply
    bot_reply = result["messages"][-1].content

    # Convert full message history to dicts for response
    response_history = [
        {"role": msg.type, "content": msg.content}
        for msg in result["messages"]
    ]

    return {
        "reply": bot_reply,
        "history": response_history
    }

# === Route for yt note generation === #
@app.post("/studybuddy/process-yt")
async def process_yt(req: ChatRequest):

    system_prompt = """
        Situation: You are an advanced AI note-taking assistant specializing in generating comprehensive, structured notes from YouTube video content. Your primary function is to transform video transcripts into clear, organized, and insightful notes that capture the key information effectively.

        Task:
            1. Extract the full transcript from the provided YouTube video link using a reliable transcription tool
            2. Analyze the transcript thoroughly to identify:
                - Main topics and subtopics
                - Key insights and important points
                - Critical takeaways
            3. Generate well-structured notes with the following characteristics:
                - Clear hierarchical organization
                - Concise yet comprehensive content
                - Logical flow matching the video's progression
                - Markdown or outline format for easy readability

        Objective: Create high-quality, actionable notes that allow readers to quickly understand the video's core content, key learnings, and most significant insights without watching the entire video.

        Additional Guidelines:
            Use tool `transcribe_yt` to get youtube transcribe 
    """
    
    messages = [SystemMessage(content=system_prompt)]

    for msg in req.history:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            messages.append(AIMessage(content=msg.content))

    # Add latest human message
    messages.append(HumanMessage(content=req.message))

    # Run LangGraph
    state = {"messages": messages}
    result = graph.invoke(state)

    # Get bot reply
    bot_reply = result["messages"][-1].content

    # Convert full message history to dicts for response
    response_history = [
        {"role": msg.type, "content": msg.content}
        for msg in result["messages"]
    ]

    return {
        "reply": bot_reply,
        "history": response_history
    }

# === Route for audio summaries === #
@app.post("/studybuddy/process-audio")
async def process_audio(req: str = Form(...), audio: UploadFile = File(...)):
    """
    Accepts a JSON string in `req` + uploaded audio file.
    """
    # Parse req
    try:
        req = json.loads(req)
        print(req)
    except Exception as e:
        return {"error": f"Invalid JSON in req: {e}, \nTraceback: {traceback.format_exc()}"}

    # Save the uploaded audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    audio_text = await transcribe_audio_file(tmp_path)

    if audio_text:
        logging.info(f"Extract text length: {len(audio_text)}")

    # system prompt for conversation summary
    system_prompt = """
    Situation: You are a professional conversation summarizer working in a high-stakes communication environment where accuracy, brevity, and clarity are paramount.

    Task: Analyze and distill the provided conversation into a concise, comprehensive summary that captures the key points, main ideas, and critical insights without losing the essential context or nuance.

    Objective: Produce a summary that allows readers to quickly understand the core content, intent, and outcomes of the conversation with minimal time investment.

    Additional Guidelines:
        You don't need to call any tool for this to extract the text from audio.
    """

    # Proceed as before…
    messages = [SystemMessage(content=system_prompt)]

    if req.get("history", []):
        for msg in req.history:
            if msg.role == "human":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                messages.append(AIMessage(content=msg.content))

    # Add latest human message
    messages.append(HumanMessage(content= f"{req.get('message', '')}; Provide a summary for this text extracted from audio: \n{audio_text}"))

    # Run LangGraph
    state = {"messages": messages}
    result = graph.invoke(state)

    # Get bot reply
    bot_reply = result["messages"][-1].content

    # Convert full message history to dicts for response
    response_history = [
        {"role": msg.type, "content": msg.content}
        for msg in result["messages"]
    ]

    return {
        "reply": bot_reply,
        "history": response_history
    }

# === Route for roadmap guide === #
@app.post("/studybuddy/roadmap")
async def roadmap(req: ChatRequest):
    """
    Accepts a JSON string in `req` and suggest career roadmap
    """
    # add system prompt to messages
    messages = [SystemMessage(content=career_system_prompt)]

    # add older history to messages
    for msg in req.history:
        if msg.role == "human":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            messages.append(AIMessage(content=msg.content))

    # add latest user req to message
    messages.append(HumanMessage(content=req.message))

    # Run LangGraph
    state = {"messages": messages}
    result = career_graph.invoke(state)

    # Get bot reply
    bot_reply = result["messages"][-1].content

    # Convert full message history to dicts for response
    response_history = [
        {"role": msg.type, "content": msg.content}
        for msg in result["messages"]
    ]

    return {
        "reply": bot_reply,
        "history": response_history
    }