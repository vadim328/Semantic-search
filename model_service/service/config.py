import yaml
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config.yaml"

            with open(config_path, "r", encoding="utf-8") as f:
                cls._instance.data = yaml.safe_load(f)

        return cls._instance

    def __getattr__(self, item):
        return self.data.get(item)
