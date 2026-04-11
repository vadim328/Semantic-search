from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio

from search_service.service.core.updater import DataUpdater
from search_service.container.di import Container
from search_service.api.routes.summarize import router as search
from search_service.api.routes.search import router as summarize
from search_service.api.routes.health import router as health
from search_service.infrastructure.logging.config import setup_logging
from search_service.config import Config

import logging

setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    Config()

    container = await Container.create()
    app.state.container = container        # type: ignore

    updater = DataUpdater(container)

    # сервис НЕ готов
    app.state.ready = False                # type: ignore
    log.info("System initializing...")

    # Откладываем запуск на 10 секунд, пока нужные сервисы не поднимутся
    await asyncio.sleep(10)

    try:
        await updater.run()
    except Exception:
        log.exception("Fatal startup error in updater")
        raise RuntimeError("Application startup failed")

    app.state.ready = True                 # type: ignore
    log.info("System is READY")

    asyncio.create_task(updater.background_updater())

    yield

    log.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

# Подключаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health)
app.include_router(search)
app.include_router(summarize)

app.mount(
    "/",
    StaticFiles(directory="search_service/frontend", html=True),
    name="frontend"
)


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
