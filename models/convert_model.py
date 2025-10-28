from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


def model_to_onnx(model_name: str):
    """Загружает, конвертирует модель в формат onnx и сохраняет
    Args:
        model_name: id модели на HuggingFace.
    """
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ort_model.save_pretrained("./onnx")
    tokenizer.save_pretrained("./onnx")


def model_optimizer(model_path: str):
    """Берет исходную модель в формате onnx оптимизирует
    Args:
        model_path: Путь до модели.
    """
    # Конфигурация оптимизации
    optimization_config = OptimizationConfig(optimization_level=99)

    # Создаём оптимизатор
    optimizer = ORTOptimizer.from_pretrained(model_path)

    # Применяем оптимизацию
    optimizer.optimize(
        save_dir="./onnx/optim/",
        optimization_config=optimization_config,
    )


