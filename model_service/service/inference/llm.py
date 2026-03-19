import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


class LLMModel:
    """Класс LLM модели для суммаризации диалога
        и извлечения признаков"""
    def __init__(self, model_path):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cpu"},
            torch_dtype="bfloat16",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path + "tokenizer/")

    def generate(self, prompt, max_tokens=512):

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(device)

        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=max_tokens,
        )
        #return self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_tokens = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def infer(self, prompt):
        return self.generate(prompt)
