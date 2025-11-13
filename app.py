from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
from service.pipeline import SemanticSearchEngine
import logging
from service.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware

setup_logging()  # настройка логирования
log = logging.getLogger(__name__)

searcher = SemanticSearchEngine()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно указать ["http://127.0.0.1:5500"] и т.п. для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await searcher.update()
    asyncio.create_task(searcher.background_updater())
    log.info("Application is start and ready to go")


@app.get("/Health")
async def root():
    return {"message": "Status OK"}


@app.get('/find/{query}')
async def return_formulation(query: str):
    """
        Поиск запросов по тексту
    """
    log.info(f"Requested: {query}")
    return searcher.search(query)


@app.post('/search')
async def search(request: Request):
    """Обработка POST-запроса /search"""
    data = await request.json()

    query = data.get("query")
    limit = data.get("limit", 5)
    alpha = data.get("alpha", 0.5)
    filters = data.get("filter", {})

    log.info(f"Request {query}, limit: {limit}, alpha: {alpha}")

    return JSONResponse(searcher.search(query, limit, alpha, filters))
