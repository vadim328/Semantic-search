from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


def model_to_onnx(model_name: str, export="./onnx"):
    """Загружает, конвертирует модель в формат onnx и сохраняет
    Args:
        model_name: id модели на HuggingFace.
        export: Директория для сохранения
    """
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ort_model.save_pretrained(export)
    tokenizer.save_pretrained(export)


def model_optimizer(model_path: str):
    """Берет исходную модель в формате onnx и оптимизирует
    Args:
        model_path: Путь до модели, не включая саму модель
    """
    # Конфигурация оптимизации
    optimization_config = OptimizationConfig(optimization_level=99)

    # Создаём оптимизатор
    optimizer = ORTOptimizer.from_pretrained(model_path)

    # Применяем оптимизацию
    optimizer.optimize(
        save_dir=model_path+"/optim/",
        optimization_config=optimization_config,
    )


if __name__ == "__main__":
    export = "./onnx"
    model_to_onnx("deepvk/USER-bge-m3", export=export)
    model_optimizer(export)
