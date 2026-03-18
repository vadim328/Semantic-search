# 📌 Semantic Search Engine (FastAPI + Qdrant + BERT + BM25)

Это сервис семантического поиска, который принимает текстовый запрос, преобразует его в эмбеддинг, и ищет похожие записи в Qdrant с использованием косинусной близости. 
Дополнительно для поиска используется алгоритм BM25, который комбинирует результат косинусной близости.
Сервис поддерживает фильтрацию по полям (client, product, диапазон дат), ускоренный поиск (HNSW) и точный поиск по всей базе.
***
🚀 Возможности

- Предварительная обработка текста для BERT и BM25
- Получение эмбеддингов текста через embedding-модель (cointegrated/rubert-tiny2, deepvk/USER-bge-m3)
- Суммаризация запросов с помощью LLM модели
- Хранение данных в Qdrant Vector DB 
- Поиск по косинусной близости
- Поиск с помощью алгоритма BM25
- Выбор режима поиска:
  - Быстрый (HNSW) — ограниченный лимит результатов, индексированный поиск 
  - Точный — перебор всех точек в коллекции
- Поддержка Docker + Docker Compose
***
🏗 Структура проекта

<pre>
project/
│
├── db/
│   └── relational_db
│       └── queries          # Директория с шаблонами запросов в реляционную БД
│       └── relational_db.py # Клиент для работы с реляционной БД
│   └── vector_db            # Логика работы с Qdrant (инициализация, поиск, скролл), логика работы с PostgreSQL
│       └── client.py        # Клиент для работы с векторной БД
│       └── collection.py    # Логика работы с коллекциями
│       └── filters.py       # Метод для конфигурации фильтров
│       └── metadata.py      # Датакласс для метаданных коллекции
│
├── Docker/                
│   └── Dockerfile           # Dockerfile для создания образа приложения
│   └── docker-compose.yaml  # yaml файл для быстрого развертывания приложения из готового образа + Qdrant
│
├── frontend/
│   └── static               
│       └── css
│           └── styles.css   # Стили 
│       └── js
│           └── api.js       # Методы для выполнения API запросов к бэкенду сервиса
│           └── form.js      # Формирование параметров для запроса поиска
│           └── main.js      # Основной скрипт для формирование веб-интерфейса и обработки запросов
│           └── table.js     # Формирование результатов запроса поиска
│           └── ui.js        # Отображение анимации
│   └── index.html           # HTML-файл веб-интерфейса (для демонстрации)
│
├── models/
│   └── model.py             # Класс для получения BERT-эмбеддингов
│   └── onnx/                # Директория с моделяги в формате onnx
│
├── routes/                
│   └── search_routes.py     # Скрипт инициализации маршрутов для /search
│
├── service/
│   └── search_engine.py     # Поисковой движок
│   └── logging_config.py    # Конфигурационный python-файл для логирования приложения
│   └── di.py                # Инициализация классов
│   └── scorer.py            # Рассчет схожести
│   └── updater.py           # Обновление данных
│   └── utils.py             # Вспомогательные методы
│
├── text_processing/                
│   └── text_preparation.py  # Обработка текста
│   └── TransformText.py     # Конструктор для обработки текста
│
├── app.py                   # Точка входа FastAPI
│
├── config.py                # Конфигурационный скрипт
│
├── config.yaml              # Конфигурационный файл
│ 
└── README.md
</pre>
***
⚙️ Требования
Python 3.10+
<pre>
pip install -r requirements.txt
</pre>

Основные библиотеки:

- FastAPI 
- Uvicorn 
- QdrantClient
- Transformers 
- PyTorch
- SQLAlchemy
- rank_bm25
***
🐳 Запуск

Собрать образ c помощью Dockerfile (Dockerfile должен быть в корне приложения):
<pre>
docker build -t semantic-search:1.0.0 .
</pre>

Создать директории

<pre>
mkdir /opt/searcher
mkdir /opt/searcher/configs
mkdir /opt/searcher/qdrant_storage
mkdir -p /opt/searcher/models/bert
</pre>

В директорию **/opt/searcher/** положить конфигурационные файлы приложения:

- config.yaml
- config.py

Cкачать модель и файлы модели из репозитория и переместить их в директорию:

<pre>
git clone https://huggingface.co/Vades/rubert-tiny2-onnx-optim
mv .../rubert-tiny2-onnx-optim/bert-onnx-optim/* /opt/searcher/models/bert
</pre>

Запустить сервисы через docker-compose:
<pre>
docker compose up -d --build
</pre>

Приложение будет доступно по адресу (порт можно изменить в compose-файле):
<pre>
http://localhost:5000
</pre>

Qdrant:
<pre>
http://localhost:6333
</pre>
***
🔍 Пример запроса поиска

HTTP POST
<pre>
POST http://host:port/search/

Headers: "Content-Type: application/json"

Тело запроса

{
  "query": "большая часть диалогов (примерно 2/3) перестала записываться в таблице",
  "product": "Erudite",
  "limit": 5,
  "alpha": 0.5,
  "exact": False
  "filters": {
    "client":  "NAUMEN НАУМЕН"
    "date_from": "2025-01-01"
    "date_to": "2025-02-01"
  }
}
</pre>

Пример ответа:

<pre>
[
  {
    "ID": "111111",
    "score": "90%",
    "responsible": "Петров Петр Петрович",
    "priority": "3",
    "date_end": "2025-12-29 14:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$11111"
  },
  {
    "ID": "222222",
    "score": "70%",
    "responsible": "Петров Петр Петрович",
    "priority": 4,
    "date_end": "2025-12-29 13:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$22222"
  },
  {
    "ID": "333333",
    "score": "50%",
    "responsible": "Петров Петр Петрович",
    "priority": 3,
    "date_end": "2025-12-29 10:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$33333"
  },
  {
    "ID": "444444",
    "score": "30%",
    "responsible": "Петров Петр Петрович",
    "priority": 4,
    "date_end": "2025-12-29 11:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$44444"
  },
  {
    "ID": "555555",
    "score": "10%",
    "responsible": "Петров Петр Петрович",
    "priority": 3,
    "date_end": "2025-12-29 12:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$55555"
  }
]
</pre>

Пример curl запроса:

<pre>
curl -X POST "192.168.211.244:5000/search/" \
-H "Content-Type: application/json" \
-d '{"query": "прошу прислать скрипт которым очищаются данные или сами запросы которые используются.", "limit": 6, "alpha": 0.6, "exact": false}'
</pre>

Описание параметров:
 - "query": Текст для которого требуется найти схожие по описанию запросы
 - "product": Название продукта,
 - "limit": Ограничение на количество найденых совпадений в порядке убывания
 - "alpha": коэффициент балансировки, принимающий значения в диапазоне от 0 до 1
   - При α = 0 полностью используется поиск по косинусной схожести
   - При α = 1 полностью используется поиск через алгоритм BM25 
 - "exact": Включение быстрого поиска по индексированным векторам
   - True: Полный поиск по всем точкам коллекции
   - False: Быстрый поиск по индексам
 - "filters": Фильтры по датам, продукту и клиенту для сужения поиска
   - "client":  Полное наименование клиента в SD
   - "date_from": Дата завершения запроса от
   - "date_to": Дата завершения запроса до
 
Получение метаданных коллекции:

HTTP GET
<pre>
GET http://host:port/search/options/metadata?product=Naumen Erudite
</pre>

Пример ответа:
<pre>
{   "points_count": 200000
    "clients":[
                "Магнит",
                "КРОК",
                "Top Contact Топ Контакт",
                "Промсвязьбанк", 
                "Глонасс",
                "Мосэнергосбыт"
              ],
    "date_last_record": 124235545
}
</pre>

Результат запроса - json-объект с данными:

- points_count - Количество точек в коллекции
- clients - Клиенты
- date_last_record - Дата завершения запроса последнего запроса в коллеции


***
⚙️ Описание конфигурационного файла

- database - блок с настройками подключения к БД
  - relational_db 
    - url - URL для подключения к базе PostgreSQL
  - vector_db
    - url - URL для подключения к базе Qdrant
    - date_from - дата, начиная с которой требуется брать данные из реляционной БД
    - params - блок с параметрами индексирования векторной БД
      - m_value - сколько k-ближайших соседей хранить, по умолчанию 128
      - ef_construct - сколько кандидатов анализируется при вставке точки, по умолчанию 600
      - full_scan_threshold - количество точек когда не нужен hnsw, по умолчанию 1000
      - max_indexing_threads - количество потоков, 0 - авто
      - on_disk - где хранить граф, False в RAM
- models - Блок с настройками использования моделей
  - llm
    - path - Путь хранения до LLM модели
  - embedding
    - path - Путь хранения до embedding модели
    - model_name - Название модели
- service - блок с настройками сервиса
  - threshold - Порог на отображение результатов, например результаты меннее 0.7 не будут возвращаться
  - logging_level - настройка уровня логирования приложения
  - products - Список продуктов с которыми будет работать сервис

Подробнее про индексирование Qdrant в [оффициальной документации](https://qdrant.tech/documentation/concepts/indexing/)
***
📌 Контакты / Авторы

    TG: @vades00777



