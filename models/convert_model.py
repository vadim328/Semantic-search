from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer, AutoModel


def model_to_onnx(model_name: str):
    """Загружает, конвертирует модель в формат onnx и сохраняет
    Args:
        model_name: Список исходных текстов.
    """
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ort_model.save_pretrained("./onnx")
    tokenizer.save_pretrained("./onnx")


