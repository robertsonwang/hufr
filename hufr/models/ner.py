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
        """
        Wrapper for HuggingFace Token Classification models to predict entiteis in free text. This class is
        designed for token classification tasks using transformers' token classification models. It includes
        methods for loading a pre-trained model, predicting labels for input texts, and handling token-to-word mapping.

        Args:
            model (Union[Callable, str, None]): Pre-trained model or path to model weights (local or remote).
                If None, the model will not be loaded.
            tokenizer (Union[Callable, str, None]): Tokenizer for the model or path to tokenizer config (local or remote)
                If None, the tokenizer will not be loaded.
            device (str, optional): Device to move the model to. Defaults to "cpu".
            threshold (float, optional): Threshold for filtering predicted labels. Defaults to 0.9.

        Attributes:
            model: The pre-trained token classification model.
            tokenizer: The tokenizer associated with the model.
            device (str): The device on which the model is loaded.
            threshold (float): The threshold value for filtering predicted labels.

        Methods:
            from_pretrained(cls, model_path): Instantiate the class with a pre-trained model path.
            predict(self, texts: List[str]) -> List[List[str]]: Predicts labels for input texts.
        """
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
        """
        Property to get the device on which the model weights are loaded.

        Returns:
            str: The device on which the model weights are loaded.
        """
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(cls, model_path):
        """
        Instantiate the class with a pre-trained model path.

        Args:
            model_path (str): Path or identifier of the pre-trained model.

        Returns:
            TokenClassificationTransformer: An instance of the class with the pre-trained model.
        """
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
            inputs, predictions, tokenizer=self.tokenizer
        )
        if isinstance(texts, str):
            return predictions[0]

        return predictions
