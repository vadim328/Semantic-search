import os
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig
from optimum.onnxruntime.configuration import QuantizationMode, QuantFormat
from transformers import AutoTokenizer


def model_to_onnx(model_name: str, export_dir="./onnx"):
    """
    Загружает, конвертирует модель в формат onnx и сохраняет
    Args:
        model_name: id модели на HuggingFace.
        export_dir: Директория для сохранения
    """
    os.makedirs(export_dir, exist_ok=True)

    # Экспорт модели в ONNX
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    print(f"Модель экспортирована в {export_dir}")


def model_optimizer(model_dir: str):
    """
    Берет исходную модель в формате onnx и оптимизирует
    Args:
        model_dir: Путь до модели, не включая саму модель
    """
    optimizer = ORTOptimizer.from_pretrained(model_dir)

    # Оптимизация (уровень 2 — безопасно для embeddings)
    optimization_config = OptimizationConfig(optimization_level=2)
    optimizer.optimize(
        save_dir=os.path.join(model_dir, "optim"),
        optimization_config=optimization_config,
    )
    print(f"Оптимизированная модель сохранена в {model_dir}/optim")


def model_quantizer(model_dir: str):
    """
    Берет оптимизированную модель в формате onnx и квантизирует
    Args:
        model_dir: Путь до модели, не включая саму модель
    """

    # Квантование модели (для ускорения инференса)
    quantizer = ORTQuantizer.from_pretrained(os.path.join(model_dir))
    quantization_config = QuantizationConfig(
        is_static=False,    # динамическое квантование
        per_channel=True,
        format=QuantFormat.QDQ,
        mode=QuantizationMode.IntegerOps
    )
    quantizer.quantize(save_dir=os.path.join(model_dir, "quant"), quantization_config=quantization_config)
    print(f"Квантованная модель сохранена в {model_dir}/quant")


if __name__ == "__main__":
    export = "../models/embedding/e5/"
    #model_to_onnx("intfloat/multilingual-e5-large", export_dir=export)
    #model_optimizer(export)
    model_quantizer(export)

