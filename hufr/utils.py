import onnxruntime as ort

from hufr.onnx import model2onnx
from hufr.constants import DEFAULT_MODEL
from hufr.models import TokenClassificationTransformerONNX
from typing import Union, List


def instantiate_onnx_model(
    model_path_or_name: str = DEFAULT_MODEL,
    onnx_output_path: str = ".tmp/model.onnx",
    tokenizer_path_or_name: Union[str, None] = None,
    execution_providers: Union[List, None] = None,
):
    """
    Instantiates a TokenClassificationTransformerONNX model from a pre-trained model path or name.

    Args:
        model_path_or_name (str, optional): Path or name of the pre-trained model. Defaults to DEFAULT_MODEL.
        onnx_output_path (str, optional): Path to save the ONNX model. Defaults to ".tmp/model.onnx".
        tokenizer_path_or_name (Union[str, None], optional): Path or name of the tokenizer. If None, the tokenizer
            will be loaded from the model. Defaults to None.
        execution_providers (Union[List, None], optional): List of execution providers for ONNXRuntime. If None, the
            available providers will be used. Defaults to None.
    Returns:
        TokenClassificationTransformerONNX: An instance of the class with the pre-trained model.
    """
    if tokenizer_path_or_name is None:
        tokenizer_path_or_name = model_path_or_name
    onnx_model_path, config = model2onnx(
        model_path_or_name,
        onnx_output_path=onnx_output_path,
        tokenizer_path_or_name=tokenizer_path_or_name,
    )
    onnx_model = TokenClassificationTransformerONNX.from_pretrained(
        onnx_model_path,
        tokenizer=tokenizer_path_or_name,
        execution_providers=ort.get_available_providers()
        if execution_providers is None
        else execution_providers,
        config=config,
    )
    return onnx_model
