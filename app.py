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


@app.get("/options")
def get_products():
    """
        GET - метод для получения списка проудктов и клиентов
    """
    return searcher.get_metadata()


@app.post('/search')
async def search(request: Request):
    """
        POST - метод для поиска схожих запросов
            :input:
                query: Текст, по которобу будут искаться схожие запросы
                limit: Ограничение на количество найденых совпадений в порядке убывания
                alpha: коэффициент балансировки, принимающий значения в диапазоне от 0 до 1
                        - При α = 0 полностью используется поиск через алгоритм BM25
                        - При α = 1 полностью используется поиск по косинусной схожести
                exact: Включение быстрого поиска по индексированным векторам
                filter: Фильтры по датам, продукту и клиенту для сужения поиска
    """
    data = await request.json()

    query = data.get("query")
    limit = data.get("limit", 5)
    alpha = data.get("alpha", 0.5)
    exact = data.get("exact", False)
    filters = data.get("filter", {})

    log.info(f"Request: {query}, limit: {limit}, alpha: {alpha}, exact: {exact}")

    return JSONResponse(searcher.search(query, limit, alpha, exact, filters))
