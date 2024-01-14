import torch.nn as nn
import torch

from hufr.utils import argmax_with_threshold
from hufr.convert import convert_token_preds
from typing import Callable, List, Union
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TokenClassificationTransformer(nn.Module):
    def __init__(
        self,
        model: Union[Callable, str, None],
        tokenizer: Union[Callable, str, None],
        device="cpu",
        threshold=0.9,
    ):
        super().__init__()
        if isinstance(model, str):
            self.model = AutoModelForTokenClassification.from_pretrained(model)
        elif model is None:
            self.model = None
        else:
            self.model = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer
        if self.model is not None:
            self.model.to(device)
        self.threshold = threshold

    @property
    def device(self):
        return next(self.parameters()).device
    
    @classmethod
    def from_pretrained(cls, model_path):
        model = cls(model=model_path, tokenizer=model_path)
        return model
    
    def predict(self, texts: List[str]) -> List[float]:
        """Predicts the label for each text in the list.

        Args:
        ----
            texts (List[str]): List of text to predict.
        Returns:
        -------
            List[int]: List of predicted label probabilities.
        """
        # TO DO: Add offset to handle long input sequences
        if self.model is None:
            raise AttributeError("Please instantiate a model using the from_pretrained method.")
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.tokenizer.model_max_length,
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        logits = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).logits
        probs = torch.softmax(logits, dim=2)
        predictions = argmax_with_threshold(
            probs,
            threshold=self.threshold,
            default=self.model.config.label2id.get("O", 0),
            dim=2,
        )
        predictions = [
            [self.model.config.id2label[x] for x in pred]
            for pred in predictions.tolist()
        ]
        predictions = convert_token_preds(
            inputs, predictions, tokenizer=self.tokenizer, skip_punc=True
        )
        if isinstance(texts, str):
            return predictions[0]

        return predictions