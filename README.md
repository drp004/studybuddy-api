# FastAPI Integration for NoteMate

This document describes the integration of FastAPI endpoints with the NoteMate application for AI-powered content processing.

## Overview

The integration connects the NoteMate frontend and backend with external FastAPI services to provide:
- Image processing and note generation
- PDF processing and note generation  
- Audio processing and note generation
- YouTube video processing and note generation
- Career roadmap generation

## FastAPI Endpoints

The following endpoints are integrated:

### 1. Image Processing
- **URL**: `https://studybuddy-api-t716.onrender.com/studybuddy/process-image`
- **Method**: POST
- **Input**: Form data with image file and request message
- **Output**: Generated notes from image content

### 2. PDF Processing
- **URL**: `https://studybuddy-api-t716.onrender.com/studybuddy/process-pdf`
- **Method**: POST
- **Input**: Form data with PDF file and request message
- **Output**: Generated notes from PDF content

### 3. Audio Processing
- **URL**: `https://studybuddy-api-t716.onrender.com/studybuddy/process-audio`
- **Method**: POST
- **Input**: Form data with audio file and request message
- **Output**: Generated notes from audio content

### 4. YouTube Processing
- **URL**: `https://studybuddy-api-t716.onrender.com/studybuddy/process-yt`
- **Method**: POST
- **Input**: JSON with YouTube link and request message
- **Output**: Generated notes from video content

### 5. Career Roadmap
- **URL**: `https://studybuddy-api-t716.onrender.com/studybuddy/roadmap`
- **Method**: POST
- **Input**: JSON with background description and interests
- **Output**: Generated career roadmap