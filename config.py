import yaml


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            with open("config.yaml", "r", encoding="utf-8") as f:
                cls._instance.data = yaml.safe_load(f)
        return cls._instance
