from fastapi import FastAPI
from service.search_engine import SemanticSearchEngine
from service.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.search_routes import create_search_router
import asyncio
import logging
from config import Config


setup_logging()  # настройка логирования
log = logging.getLogger(__name__)

searcher = SemanticSearchEngine()
search_router = create_search_router(searcher)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно указать ["http://127.0.0.1:5000"] и т.п. для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)  # Подключаем маршрут /search


@app.on_event("startup")
async def startup_event():
    """
        Функция для первичного запуска приложения.
            Инициализирует классы, подключаетсяк БД,
                векторизует запросы и загружает их в БД
    """
    Config()  # Считываем файл на старте
    await searcher.update()
    asyncio.create_task(searcher.background_updater())
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/Health")
async def root():
    return {"Status": "OK"}


@app.get('/find/{query}')
async def return_formulation(query: str):
    """
        GET - метод. Поиск запросов по тексту
    """
    log.info(f"Requested: {query}")
    return searcher.search(query)
