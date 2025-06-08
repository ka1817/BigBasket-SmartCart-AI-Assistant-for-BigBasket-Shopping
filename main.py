# main.py
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from src.retrival_genaration import ingest_bigbasket_data, generate_bigbasket_chain
from src.query_rewritting import query_rewriting

# Initialize FastAPI and setup static/templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load vectorstore and RAG chain
vectorstore = ingest_bigbasket_data()
chain = generate_bigbasket_chain(vectorstore)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": None})

@app.post("/", response_class=HTMLResponse)
def get_response(request: Request, query: str = Form(...)):
    rewritten = query_rewriting(query)
    answer = chain.invoke(rewritten)
    return templates.TemplateResponse("index.html", {"request": request, "response": answer, "query": query})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
