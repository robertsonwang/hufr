import os
import logging

from pathlib import Path
from transformers.onnx import FeaturesManager, export
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.configuration_utils import PretrainedConfig


def model2onnx(
    model_path_or_name: str,
    onnx_output_path: str = None,
    tokenizer_path_or_name: str = None,
) -> tuple[str, PretrainedConfig]:
    """
    Export a model to ONNX format.
        Args:
            model_path_or_name (str): Path or identifier of the model.
            tokenizer_path_or_name (str, optional): Path or identifier of the tokenizer.
                If not provided, the model_path_or_name will be used. Defaults to None.
        Returns:
            path to model output or model as bytes depending on output parameter
    """
    model = AutoModelForTokenClassification.from_pretrained(model_path_or_name)
    if os.path.exists(onnx_output_path):
        logging.info(f"Found onnx weights at {onnx_output_path}")
        return onnx_output_path, model.config
    if tokenizer_path_or_name is None:
        tokenizer_path_or_name = model_path_or_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
    onnx_output_path = Path(
        f"{Path(model_path_or_name).path}.onnx"
        if onnx_output_path is None
        else onnx_output_path
    )
    if not onnx_output_path.parent.exists():
        logging.info(
            f"Created parent directory to save onnx weights at {onnx_output_path.parent}"
        )
        os.mkdir(onnx_output_path.parent)

    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
    onnx_config = model_onnx_config(model.config)
    export(
        model=model,
        output=Path(onnx_output_path),
        preprocessor=tokenizer,
        tokenizer=None,
        config=onnx_config,
        opset=12,
    )
    logging.info(f"Exported onnx weights to {onnx_output_path}")
    return str(onnx_output_path), model.config
