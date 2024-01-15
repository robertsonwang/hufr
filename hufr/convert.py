import string

from typing import Callable
from transformers.tokenization_utils_base import BatchEncoding

def convert_token_preds(
    tokenized_inputs: BatchEncoding,
    preds: list,
    tokenizer: Callable,
    skip_punc: bool = True,
):
    """
    Converts token-level predictions to word-level predictions based on tokenized inputs.

    Args:
        tokenized_inputs (BatchEncoding): Tokenized inputs using transformers' BatchEncoding.
        preds (list): List of token-level predictions.
        tokenizer (Callable): A callable tokenizer to convert token IDs to token strings.
        skip_punc (bool, optional): Flag to skip punctuation tokens when aggregating predictions.
            Defaults to True.

    Returns:
        list: List of word-level predictions, where each element is a list of labels corresponding to words.

    Note:
        This function assumes that the input token predictions follow the same order as the tokenized inputs.
        It aggregates token-level predictions into word-level predictions, considering token-to-word mapping.
        If `skip_punc` is True, it skips punctuation tokens when aggregating predictions.
    """
    labels = []
    for i, label in enumerate(preds):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_tags = []

        for k, word_idx in enumerate(word_ids):
            token_str = tokenizer.convert_ids_to_tokens(
                int(tokenized_inputs.input_ids[i][k])
            )
            if word_idx is None:
                continue
            if (
                word_idx != previous_word_idx and not skip_punc
            ):  # Only label the first token of a given word.
                label_tags.append(label[k].split("-")[-1])
            elif (
                word_idx != previous_word_idx
                and skip_punc
                and token_str not in string.punctuation
            ):
                # Remove BIO tags, keep only entity label
                label_tags.append(label[k].split("-")[-1])
            previous_word_idx = word_idx

        labels.append(label_tags)
    return labels
