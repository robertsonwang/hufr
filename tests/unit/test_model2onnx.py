import pytest
import tempfile
import os

from hufr.onnx import model2onnx
from hufr.constants import DEFAULT_MODEL


@pytest.fixture
def temp_dir():
    # Create a temporary directory and return its path
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_model2onnx(temp_dir):
    # Call the function with the specified arguments
    onnx_output_path, model_config = model2onnx(
        model_path_or_name=DEFAULT_MODEL,
        onnx_output_path=os.path.join(temp_dir, "model.onnx"),
        tokenizer_path_or_name=DEFAULT_MODEL,
    )

    # Check for the presence of config attributes
    assert model_config.label2id is not None
    assert model_config.id2label is not None

    # Check if the ONNX file is created
    assert os.path.exists(onnx_output_path)
