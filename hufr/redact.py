import string

from typing import Union
from hufr.models.ner import TokenClassificationTransformer
from hufr.constants import DEFAULT_REDACTION_MAP, DEFAULT_MODEL

def redact_text(
    text: str,
    model: Union[TokenClassificationTransformer, None] = None,
    redaction_map: dict = DEFAULT_REDACTION_MAP,
    return_preds: bool = False,
):
    """
    Redacts sensitive information in the given text using a pre-trained HuggingFace Token Classification model.

    Args:
        text (str): The input text to be redacted.
        model (Union[TokenClassificationTransformer, None], optional): A pre-trained Token Classification model.
            If not provided, the default model will be used. Defaults to None.
        redaction_map (dict, optional): A mapping of predicted labels to redacted content.
            Defaults to DEFAULT_REDACTION_MAP.
        return_preds (bool, optional): Flag to return predictions along with the redacted text.
            Defaults to False.

    Returns:
        Union[str, Tuple[str, List[str]]]: Redacted text. If `return_preds` is True, also returns a tuple
        containing the redacted text and a list of predictions for each token.

    Note:
        This function uses a Token Classification model to predict labels for tokens in the input text
        and redacts sensitive information based on the provided redaction map.
        If `return_preds` is True, the function returns both the redacted text and the list of predictions.
    """
    if model is None:
        model = TokenClassificationTransformer.from_pretrained(DEFAULT_MODEL)

    predictions = model.predict(text)
    text_lines = text.split()
    redacted_text_lines = []
    if any(pred in redaction_map for pred in predictions):
        # Adjustment redaction for punctuation on the right
        for pred, line in zip(predictions, text_lines):
            is_last_char_punc = line[-1] in string.punctuation
            if is_last_char_punc and pred in redaction_map:
                redacted_text_lines.append(redaction_map.get(pred, line) + line[-1])
            else:
                redacted_text_lines.append(redaction_map.get(pred, line))
    else:
        redacted_text_lines = text_lines
    if return_preds:
        return " ".join(redacted_text_lines), predictions
    return " ".join(redacted_text_lines)
