from typing import List, Union

import numpy as np
import torch

from embedding_generator.service.generator import GeneratorService


class GeneratorEngine:
    def __init__(self, model_path: str, tokenizer_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = GeneratorService(
            model_path=model_path, tokenizer_path=tokenizer_path, device=device
        )

    def get_embedding_from_sentence(
        self, sentences: Union[str, List[str]]
    ) -> np.ndarray:
        return self.generator.generate(sentences)
