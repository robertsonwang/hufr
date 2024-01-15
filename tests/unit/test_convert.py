import pytest

from transformers import BertTokenizerFast
from hufr.convert import convert_token_preds


@pytest.fixture
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


def test_convert_token_preds(tokenizer):
    sentences = ["This is a test sentence.", "Another example."]

    # Tokenize input sentences
    tokenized_inputs = tokenizer.batch_encode_plus(sentences)

    # Mock token-level predictions
    preds = [
        ["O", "O", "O", "B-PER", "O", "B-PER", "O", "O"],
        ["B-PER", "B-PER", "B-PER", "O", "O"],
    ]

    # Convert token-level predictions to word-level predictions
    word_preds = convert_token_preds(tokenized_inputs, preds, tokenizer)

    # Expected word-level entity predictions
    expected_word_preds = [["O", "O", "PER", "O", "PER"], ["PER", "PER"]]

    assert all(
        len(pred) == len(sentence.split())
        for pred, sentence in zip(expected_word_preds, sentences)
    )
    assert word_preds == expected_word_preds
