from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer


class GeneratorService:
    def __init__(self, model_path: str, tokenizer_path: str, device: str):
        self.device = device
        self.model = self._load_model(model_path=model_path)
        self.tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)

    def _load_model(self, model_path: str):
        return AutoModel.from_pretrained(model_path)

    def _load_tokenizer(self, tokenizer_path: str):
        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _preprocess(self, sentences: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

    def _inference(
        self, encoded_output: Dict[str, torch.Tensor]
    ) -> transformers.modeling_outputs:
        with torch.no_grad():
            return self.model(**encoded_output.to(self.device))

    def _mean_pooling(
        self, model_output: transformers.modeling_outputs, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output[0]  # first element collect sentence embeeding
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _normalize(self, mean_pooling_tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(mean_pooling_tensor, p=2, dim=1)

    def _postprocess(
        self, model_output: transformers.modeling_outputs, attention_mask: torch.Tensor
    ) -> np.ndarray:
        mean_pooling_tensor = self._mean_pooling(
            model_output=model_output, attention_mask=attention_mask
        )
        return (
            self._normalize(mean_pooling_tensor=mean_pooling_tensor)
            .detach()
            .cpu()
            .numpy()
        )

    def generate(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        encoded_output = self._preprocess(sentences)
        inferenced_output = self._inference(encoded_output)

        return self._postprocess(inferenced_output, encoded_output["attention_mask"])
