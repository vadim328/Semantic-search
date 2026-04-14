📌 Semantic Search Project

<table style="width:100%; border:none;">
  <tr>
    <td style="border:2px solid #000; border-radius:8px; padding:12px; text-align:center; background:#fff;">
      <b><span style="color:#000;">🔍 Search Service</span></b><br>
      <span style="color:#000;">⚡ FastAPI &nbsp;•&nbsp; 💾 Qdrant &nbsp;•&nbsp; 📊 BM25</span>
    </td>
    <td style="border:2px solid #000; border-radius:8px; padding:12px; text-align:center; background:#fff;">
      <b><span style="color:#000;">🧠 Model Service</span></b><br>
      <span style="color:#000;">🔗 gRPC &nbsp;•&nbsp; 🤖 Llama &nbsp;•&nbsp; 📐 BGE-M3</span>
    </td>
  </tr>
</table>

Сервис семантического поиска и суммаризации, предназначенный для поиска схожих запросов с высокой точностью за счёт комбинации векторных и классических методов ранжирования.

🚀 Основные возможности

    🔍 Семантический поиск по текстовым запросам
    🧠 Комбинация векторного поиска и алгоритма BM25
    💬 Поиск по комментариям
    🧾 Суммаризация текста с помощью LLM
    ⚡ Ускоренный поиск (HNSW) и точный (full scan) режим
    🎯 Фильтрация по:
        продукту
        клиенту
        диапазону дат
    🧠 Как это работает
        1) Входной запрос преобразуется в эмбеддинг с помощью эмбеддинг модели BGE-M3
        2) Выполняется поиск в векторной базе Qdrant по косинусной близости
        3) Для полученного результата применяется алгоритм BM25 для реранкинга
        4) Возвращаются наиболее релевантные записи

    🧩 Мультивекторное хранение
    
    Для повышения качества поиска сервис использует мультивекторный подход:
    Каждая запись хранится в виде нескольких представлений:
        📄 оригинальный текст
        🧾 суммаризированный текст
        💬 комментарии
    Это позволяет:
        - находить совпадения даже при разных формулировках
        - учитывать контекст из комментариев
        - улучшать полноту и точность результатов

    🔎 Режимы поиска
    Сервис поддерживает несколько режимов:
        - Base — поиск по оригинальному и суммаризированному тексту
        - Full — поиск по всем полям (включая комментарии)
        - Comments — поиск только по комментариям
***

🏗 Структура проекта

<pre>
project/
│
├── contracts/
│   └── generated/                             # gRPC - методы для интеграции с сервисом моделей
│   └── proto/                                 # Прото-файл gRPC
│
├── model_service/                             # Сервис моделей
│   └── Docker/                
│       └── Dockerfile                         # Dockerfile для создания образа приложения
│       └── docker-compose.yaml                # yaml файл для быстрого развертывания приложения из готового образа
│   └── models/
│       └── embedding/                         # Директория с Эмбеддинг моделью
│       └── llm/                               # Директория с llm моделью
│   └── service/
│       └── inference/                         # Классы для инференса эмбеддинг и llm моделей
│       └── config.py                          # Конфигурационный python-файл
│       └── grpc_server.py                     # Реализация grpc сервера
│       └── logging_config.py                  # Кофигурация логирования сервиса
│   └── utils/                                 # Утилиты для конвертации и проверки моделей
│   └── config.yaml                            # Конфигурационный файл
│   └── requirements.txt                       # Необходимые пакеты для работы сервиса
│
├── search_service/                            # Сервис для поиска
│   └── api/
│       └── deps/
│           └── container.py                   # Слой для получения di - контейнера
│           └── orchestrator.py                # Слой для получения оркестратора суммаризации
│           └── searcher.py                    # Слой для получения поискового движка
│       └── routes/
│           └── health.py                      # Маршрут health-check
│           └── search.py                      # Маршрут для поиска
│           └── summarize.py                   # Маршрут для суммаризации
│       └── schemas/
│           └── search.py                      # Pydantic-класс для поиска
│           └── summarization.py               # Pydantic-класс для суммаризации
│
│   └──container/                     
│       └── di.py                              # di-контейнер с необходимыми объектами
│
│   └── Docker/
│       └── Dockerfile                         # Dockerfile для создания образа приложения
│       └── docker-compose.yaml                # yaml файл для быстрого развертывания приложения из готового образа + Qdrant
│
│   └── frontend/                              # web-interface для сервиса поиска
│       └── static/               
│           └── css/                           # Стили 
│           └── js/
│               └── api.js                     # Методы для выполнения API запросов к бэкенду сервиса
│               └── customSelect.js            # Формирование селектов для полей "Продукт" и "Клиент"
│               └── form.js                    # Формирование параметров для запроса поиска
│               └── main.js                    # Основной скрипт для формирование веб-интерфейса и обработки запросов
│               └── table.js                   # Формирование результатов запроса поиска
│               └── ui.js                      # Отображение анимации
│           └── index.html                     # HTML-файл веб-интерфейса (для демонстрации)
│   
│   └── infrastructure/
│       └── clients/
│           └── llm_settings.py                # Датакласс с настройками суммаризации
│           └── model_client.py                # Клиент для взаимодействия с моделями по gRPC
│           └── summarization_builder.py       # Разделения текста на чанки
│           └── summarization_orchestrator.py  # Оркестратор для выполнения запросов на суммаризацию
│       └── db/
│           └── relational_db/
│               └── queries                    # Директория с шаблонами запросов в реляционную БД
│               └── relational_db.py           # Клиент для работы с реляционной БД
│           └── vector_db/                     # Логика работы с Qdrant (инициализация, поиск, скролл), логика работы с PostgreSQL
│               └── client.py                  # Клиент для работы с векторной БД
│               └── collection.py              # Логика работы с коллекциями
│               └── filters.py                 # Метод для конфигурации фильтров
│               └── metadata.py                # Датакласс для метаданных коллекции
│       └── logging/
│           └── config.py                      # Кофигурация логирования сервиса
│       └── retry/
│           └── base.py                        # Базовый метод для повторных попыток выполнения запросов
│           └── conditions.py                  # Правила обработки ошибок для gRPC
│           └── grpc.py                        # Метод для повторных попыток выполнения запросов gRPC
│           └── qdrant.py                      # Метод для повторных попыток выполнения запросов к qdrant
│
│   └── service/
│       └── core/
│           └── scorer.py                      # Рассчет схожести
│           └── search_engine.py               # Поисковой движок
│           └── search_mode.py                 # Enum-класс для режима поиска
│           └── updater.py                     # Сервис обновления данных
│       └── utils/                             # Утилиты для общих действий
│   └── text_processing/
│       └── text_preparation.py                # Конструкторы для обработки текста
│       └── TransformsText.py                  # Классы для обработки текста
│
│   └── app.py                                 # FastAPI приложение
│   └── config.py                              # Конфигурационный скрипт
│   └── config.yaml                            # Конфигурационный файл
│   └── requirements.txt                       # Конфигурационный файл
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
- AsyncQdrantClient
- Transformers 
- PyTorch
- SQLAlchemy
- rank_bm25
- llama-cpp-python
- grpc
***
🐳 Запуск

Директория для создания образа должна содержать:

1) contracts
2) search_service/model_service
3) Dockerfile
4) requirements.txt
5) .dockerignore
6) docker-compose.yaml - Опционально, если требуется запустить сервис отдельно

Собрать образы c помощью Dockerfile (Dockerfile должен быть в корне приложения):
<pre>
docker build -t "image-name" .
</pre>

Для сервиса поиска

1) Создать директории для сервиса поиска

<pre>
mkdir -p /opt/supportai/sherlock/config/
mkdir /opt/supportai/qdrant_storage/
</pre>

2) В директорию **/opt/supportai/sherlock/config/** положить конфигурационный файл приложения:
   - config.yaml

Для сервиса моделей

1) Создать директории

<pre>
mkdir /opt/supportai/athena/
mkdir -p /opt/supportai/athena/models/embedding
mkdir /opt/supportai/athena/models/llm
</pre>

2) В директорию **/opt/supportai/athena/** положить конфигурационный файл приложения:
   - config.yaml
3) В директорию **/opt/supportai/athena/models/embedding** положить эмбеддинг-модель
3) В директорию **/opt/supportai/athena/models/llm** положить llm-модель


Запустить сервисы через docker-compose:
<pre>
docker compose up -d
</pre>

Приложение будет доступно по адресу (порт можно изменить в compose-файле):

Sherlock
<pre>
http://localhost:5000
</pre>

Qdrant:
<pre>
http://localhost:6333
</pre>

Athena
<pre>
http://localhost:50051
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
  "product": "Naumen Erudite",
  "limit": 5,
  "alpha": 0.5,
  "mode": "base"
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
    "registry_date": "2025-12-29 14:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$11111"
  },
  {
    "ID": "222222",
    "score": "70%",
    "responsible": "Петров Петр Петрович",
    "priority": 4,
    "registry_date": "2025-12-29 13:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$22222"
  },
  {
    "ID": "333333",
    "score": "50%",
    "responsible": "Петров Петр Петрович",
    "priority": 3,
    "registry_date": "2025-12-29 10:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$33333"
  },
  {
    "ID": "444444",
    "score": "30%",
    "responsible": "Петров Петр Петрович",
    "priority": 4,
    "registry_date": "2025-12-29 11:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$44444"
  },
  {
    "ID": "555555",
    "score": "10%",
    "responsible": "Петров Петр Петрович",
    "priority": 3,
    "registry_date": "2025-12-29 12:35:11",
    "url": "https://support.naumen.ru/sd/operator/#uuid:serviceCall$55555"
  }
]
</pre>

Пример curl запроса:

<pre>
curl -X POST "localhost:5000/search/" \
-H "Content-Type: application/json" \
-d '{"query": "прошу прислать скрипт которым очищаются данные или сами запросы которые используются.", "product": "Naumen Erudite", "limit": 6, "alpha": 0.6, "exact": false}'
</pre>

Описание параметров:
 - "query" (Обязательный) - Текст для которого требуется найти схожие по описанию запросы
 - "product" (Обязательный) - Название продукта,
 - "limit" (По умолчанию 5) - Ограничение на количество найденых совпадений в порядке убывания
 - "alpha" (По умолчанию 0.5) - коэффициент балансировки, принимающий значения в диапазоне от 0 до 1
   - При α = 0 полностью используется поиск по косинусной схожести
   - При α = 1 полностью используется поиск через алгоритм BM25 
- "mode" (По умолчанию Base) - Режим поиска - Полный (Full), по описанию проблемы и суммаризаци (Base), комментариям (Comments)
 - "exact" (По умолчанию True) - Включение быстрого поиска по индексированным векторам
   - True - Полный поиск по всем точкам коллекции
   - False - Быстрый поиск по индексам
 - "filters" - Фильтры по датам, продукту и клиенту для сужения поиска
   - "client" -  Полное наименование клиента в SD
   - "date_from" - Дата завершения запроса от
   - "date_to" - Дата завершения запроса до

Получение списка продуктов:

HTTP GET
<pre>
GET http://host:port/search/options/products
</pre>

Результат запроса - список продуктов: ["NCC", "Naumen Erudite"]
 
Получение метаданных коллекции:

HTTP GET
<pre>
GET http://host:port/search/options/metadata?product='Naumen Erudite'
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
- date_last_record - Дата завершения последнего запроса в коллеции

🧾 Пример запроса суммаризации
HTTP POST
<pre> 
POST http://host:port/summarization/ 
Headers: "Content-Type: application/json" 
Тело запроса { 
              "text": "В системе наблюдается проблема: большая часть диалогов (примерно 2/3) перестала записываться в таблице. Ошибка проявляется периодически.", 
              "comments": "Пользователь уточнил, что проблема началась после обновления. Есть подозрение на некорректную работу БД." 
             } 
</pre>
📤 Пример ответа
<pre> 
{ "summary": "После обновления системы перестала записываться значительная часть диалогов. Возможная причина — некорректная работа базы данных." } 
</pre>
🧪 Пример curl запроса
<pre> 
curl -X POST "http://localhost:5000/summarization/" \ 
             -H "Content-Type: application/json" \ 
             -d '{ "text": "В системе наблюдается проблема: большая часть диалогов перестала записываться.", "comments": "Ошибка появилась после обновления системы." }' 
</pre>
⚙️ Описание параметров
- "text" (обязательный) - Текст, который необходимо суммаризировать
- "comments" (необязательный) - Дополнительные комментарии или контекст, который учитывается при суммаризации

***
⚙️ Описание конфигурационного файла сервиса поиска

- service - блок с настройками сервиса
  - logging_level - настройка уровня логирования приложения
  - products - Список продуктов с которыми будет работать сервис
  - searcher
    - threshold - Порог на отображение результатов, например результаты меннее 0.7 не будут возвращаться
  - updater
    - time_window - Максимальный размер временного окна (в днях) для разбивки периода.
    - max_concurrent - Количество одновременных потоков для обработки полученных строк из реляционной БД. Рекомендуемое значение - количество воркеров для сервиса моделей (не используется)

- database - блок с настройками подключения к БД
  - relational_db 
    - url - URL для подключения к базе PostgreSQL
  - vector_db
    - url - URL для подключения к базе Qdrant
    - date_from - дата, начиная с которой требуется брать данные из реляционной БД
    - vector_params - Параметры векторов, название и размер вектора
    - params - блок с параметрами индексирования векторной БД
      - m_value - сколько k-ближайших соседей хранить, по умолчанию 128
      - ef_construct - сколько кандидатов анализируется при вставке точки, по умолчанию 600
      - full_scan_threshold - количество точек когда не нужен hnsw, по умолчанию 1000
      - max_indexing_threads - количество потоков, 0 - авто
      - on_disk - где хранить граф, False в RAM
- model
  - url - URL для подключения к сервису моделей
  - chunking
    - max_content_tokens - Максимальная длина промпта
    - generation_tokens - Количество токенов для генерации результата
    - token_safety_ratio - Доля, которую необходимо брать от максимального количества токенов
    - chars_per_token - Количество символов в токене
  - timeouts:
    - timeout_generate - Таймаут ожидания результата LLM - модели
    - timeout_embed - Таймаут ожидания результата Embedding - модели

Подробнее про индексирование Qdrant в [оффициальной документации](https://qdrant.tech/documentation/concepts/indexing/)

⚙️ Описание конфигурационного файла сервиса моделей

- service
  - max_workers - Количество воркеров (реплик сервиса)
  - logging_level - Уровень логирования
- llm
  - path - Путь до LLM моделей в формате gguf
  - n_threads - Количество потоков
  - n_ctx - Длина контекста, в токенах
  - generate:
    - max_tokens - Количество токенов для генерации
    - temperature - Диапазон: 0–1. креативность / случайность модели, чем ниже тем детерминированней ответ
    - top_p - Диапазон: 0–1. Например top_p=0.9 → модель выбирает из 90% самых вероятных слов
    - top_k - Количество наиболее вероятных слов для выбора следующего слова. top_k = 1 - модель выбирает самое вероятное
    - repeat_penalty - Штраф за повторение уже использованных токенов
- embedding
  - path - Путь до embedding моделей в формате onnx
  - model_name - Название модели
***
📌 Контакты / Авторы

    TG: @vades00777



