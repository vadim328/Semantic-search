import yaml
from service.utils import load_file


class Config:
    """
    Singleton для конфигурации.
    Загружает config.yaml относительно PYTHONPATH через load_file.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # читаем конфиг через load_file
            config_text = load_file("search_service/config.yaml")
            cls._instance.data = yaml.safe_load(config_text)

        return cls._instance

    def __getattr__(self, item):
        return self.data.get(item)
