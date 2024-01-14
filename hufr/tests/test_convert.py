import string
import pytest
from transformers.tokenization_utils_base import BatchEncoding

# Import your function
from hufr.convert import convert_token_preds  # Replace 'your_module' with the actual module name

# Mocking tokenized inputs and tokenizer
class MockTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return [str(i) for i in ids]

@pytest.fixture
def tokenized_inputs():
    return BatchEncoding(
        input_ids=[[1, 2, 3], [4, 5, 6]],
        word_ids=[[0, 0, 1], [2, 2, 3]],
    )

def test_convert_token_preds(tokenized_inputs):
    # Mocking preds and tokenizer
    preds = [['B-PER', 'I-PER', 'O'], ['B-LOC', 'I-LOC', 'O']]
    tokenizer = MockTokenizer

    # Test with skip_punc=True
    result_skip_punc = convert_token_preds(tokenized_inputs, preds, tokenizer, skip_punc=True)
    expected_skip_punc = [['PER', 'O'], ['LOC', 'O']]

    assert result_skip_punc == expected_skip_punc

    # Test with skip_punc=False
    result_not_skip_punc = convert_token_preds(tokenized_inputs, preds, tokenizer, skip_punc=False)
    expected_not_skip_punc = [['B-PER', 'I-PER', 'O'], ['B-LOC', 'I-LOC', 'O']]

    assert result_not_skip_punc == expected_not_skip_punc