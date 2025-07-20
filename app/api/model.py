import logging
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RefactorModel:
    def __init__(self):
        model_path = os.getenv("MODEL_PATH")
        if not model_path:
            raise ValueError("MODEL_PATH environment variable not set.")

        logging.info(f"Loading model from local path: {model_path}...")
        
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logging.info(f"Using device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logging.info("Model and tokenizer loaded successfully.")

    def refactor(self, code: str) -> str:
        prompt = f"""
You are a coding assistant. Your task is to refactore a provided Python code.
Respond ONLY with a clean and efficient Python code and nothing else.
No comments or other text. No explanations. No markdown.<|EOT|>

User:
{code}
<|EOT|>

Assistant:
```python
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=32021
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        code_match = re.search(r"(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else response

        return code.strip()
