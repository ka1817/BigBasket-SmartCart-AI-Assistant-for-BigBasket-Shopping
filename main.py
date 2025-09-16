from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from src.retrival_genaration import QueryRouter

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

router = QueryRouter()

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response": None, "query": None}
    )

@app.post("/", response_class=HTMLResponse)
async def get_response(request: Request, query: str = Form(...)):
    answer = router.route(query)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response": answer, "query": query}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
