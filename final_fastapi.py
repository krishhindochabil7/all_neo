from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
import os
import networkx as nx
# Set up logging
from logging_setup import logger
from all_in_one import initialize_neo, load_the_graph, all_final
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="/home/krishhindocha/Desktop/Neo/END_RESULT/GROQ_GOOGLE/static"), name="static")

# Mount the IMAGES_OUTPUT directory to serve images
app.mount("/images", StaticFiles(directory="/home/krishhindocha/Desktop/Neo/END_RESULT/GROQ_GOOGLE/IMAGES_OUTPUT"), name="images")

templates = Jinja2Templates(directory="/home/krishhindocha/Desktop/Neo/END_RESULT/GROQ_GOOGLE/Templates")
# Initialize with placeholder values

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_document(request: Request, file_path: str = Form(...)):
    graph = initialize_neo()
    logger.info(f"Received file path: {file_path}")
    load_the_graph(graph,file_path)
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/ask", response_class=HTMLResponse)
async def ask_question_page(request: Request):
    return templates.TemplateResponse("ask_question.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    logger.info(f"Received question: {question}")

    try:
        final_data,generated_query,full_graph_path,sub_graph_path = all_final(question)
        logger.info("Graphs generated and saved successfully.")
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "final_data": final_data,
            "cypher_query": generated_query,
            "full_graph_path": full_graph_path,
            "sub_graph_path": sub_graph_path,
        })
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return templates.TemplateResponse("ask_question.html", {"request": request, "error": "Error processing question. Please try again."})