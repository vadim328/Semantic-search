from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import asyncio

from search_service.service.config.logging_config import setup_logging
from search_service.service.core.search_engine import SemanticSearchEngine
from search_service.service.core.updater import DataUpdater
from search_service.service.config.di import init_container
from search_service.api.routes import register_routes
from search_service.config import Config

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

    routers = register_routes(
        searcher=searcher,
        container=container
    )
    for router in routers:
        log.info(f"Include api router - prefix={router.prefix}")
        app.include_router(router)

    # Статика фронтенда
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Функция для корректного завершения приложения:
    - закрывает gRPC клиент
    - останавливает все сервисы
    """
    container = getattr(app, "container", None)
    if container and hasattr(container, "model_client"):
        await container.model_client.close()
        log.info("ModelServiceClient correct closed")


@app.get("/Health")
async def root():
    return {"Status": "OK"}
