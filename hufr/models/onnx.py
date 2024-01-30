import onnxruntime
import numpy as np
import torch

from hufr.models.utils import argmax_with_threshold
from typing import Callable, List, Union
from transformers import AutoTokenizer
from hufr.convert import convert_token_preds
from transformers.configuration_utils import PretrainedConfig


def softmax(x, axis=2):
    # Ensure numerical stability by subtracting the maximum value
    # along the second dimension before applying the softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))

    # Calculate softmax along the second dimension
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class TokenClassificationTransformerONNX:
    def __init__(
        self,
        onnx_model_path: Union[str, None],
        tokenizer: Union[Callable, str, None],
        execution_providers: List = ["CPUExecutionProvider"],
        threshold=0.9,
        config: Union[PretrainedConfig, None] = None,
    ):
        sess_option = onnxruntime.SessionOptions()
        sess_option.optimized_model_filepath = onnx_model_path
        sess_option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif tokenizer is None:
            self.tokenizer = None
        self.execution_providers = execution_providers
        self.session = None
        if onnx_model_path is not None:
            self.session = onnxruntime.InferenceSession(
                onnx_model_path, sess_option, providers=["CPUExecutionProvider"]
            )
        self.threshold = threshold
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        onnx_model_path: str,
        tokenizer: Union[Callable, str, None] = None,
        execution_providers: List[str] = ["CPUExecutionProvider"],
        threshold=0.9,
        config: Union[PretrainedConfig, None] = None,
    ):
        """
        Instantiate the class with a pre-trained model path.

        Args:
            model_path (str): Path or identifier of the pre-trained model.

        Returns:
            TokenClassificationTransformerONNX: An instance of the class with the pre-trained model.
        """
        model = cls(
            onnx_model_path=onnx_model_path,
            tokenizer=tokenizer,
            execution_providers=execution_providers,
            threshold=threshold,
            config=config,
        )
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
        if self.session is None:
            raise AttributeError(
                "Please instantiate an onnx model using the from_pretrained method."
            )
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.tokenizer.model_max_length,
        )
        inputs["input_ids"] = inputs["input_ids"].cpu().numpy()
        inputs["attention_mask"] = inputs["attention_mask"].cpu().numpy()
        inputs["token_type_ids"] = inputs["token_type_ids"].cpu().numpy()
        logits = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"],
            },
        )[0]
        probs = softmax(logits)
        predictions = argmax_with_threshold(
            torch.tensor(probs),
            threshold=self.threshold,
            default=self.config.label2id.get("O", 0),
            dim=2,
        )
        predictions = [
            [self.config.id2label[x] for x in pred] for pred in predictions.tolist()
        ]
        predictions = convert_token_preds(inputs, predictions, tokenizer=self.tokenizer)
        if isinstance(texts, str):
            return predictions[0]

        return predictions
