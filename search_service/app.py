from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import asyncio

from service.logging_config import setup_logging
from service.search_engine import SemanticSearchEngine
from service.updater import DataUpdater
from service.di import init_container
from api.routes import register_routes
from config import Config

setup_logging()
log = logging.getLogger(__name__)

app = FastAPI()

# Подключаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Функция для первичного запуска приложения:
    - создаёт контейнер
    - передаёт его searcher и updater
    - запускает updater
    """

    Config()  # Считываем файл на старте

    # создаём контейнер
    container = await init_container()

    searcher = SemanticSearchEngine(container)
    updater = DataUpdater(container)

    # Запускаем обновления в фоне
    asyncio.create_task(updater.run())

    search_routers = register_routes(
        searcher=searcher,
        container=container
    )
    for router in search_routers:
        log.info(f"Include api router - {router}")
        app.include_router(router)

    # Статика фронтенда
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/Health")
async def root():
    return {"Status": "OK"}
