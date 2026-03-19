from fastapi import FastAPI
from service.search_engine import SemanticSearchEngine
from service.updater import DataUpdater
from service.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import register_routes
import asyncio
import logging
from config import Config


setup_logging()  # настройка логирования
log = logging.getLogger(__name__)
app = FastAPI()

searcher = SemanticSearchEngine()
updater = DataUpdater()

# Подключаем маршруты
search_router = register_routes(searcher)
for router in register_routes(searcher):
    log.info(f"include api router - {router}")
    app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно указать ["http://127.0.0.1:5000"] и т.п. для безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
        Функция для первичного запуска приложения.
            Инициализирует классы, подключаетсяк БД,
                векторизует запросы и загружает их в БД
    """
    Config()  # Считываем файл на старте
    asyncio.create_task(updater.run())
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/Health")
async def root():
    return {"Status": "OK"}
