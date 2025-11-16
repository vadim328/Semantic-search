from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from service.pipeline import SemanticSearchEngine
from service.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from config import Config


setup_logging()  # настройка логирования
log = logging.getLogger(__name__)

searcher = SemanticSearchEngine()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно указать ["http://127.0.0.1:5000"] и т.п. для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    Config()  # Считываем файл на старте
    await searcher.update()
    asyncio.create_task(searcher.background_updater())
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
    log.info("Application is start and ready to go")


@app.get("/Health")
async def root():
    return {"Status": "OK"}


@app.get('/find/{query}')
async def return_formulation(query: str):
    """
        Поиск запросов по тексту
    """
    log.info(f"Requested: {query}")
    return searcher.search(query)


@app.get("/options")
def get_products():
    return searcher.get_metadata()


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
