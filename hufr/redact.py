from hufr.models.ner import TokenClassificationTransformer
from hufr.constants import DEFAULT_REDACTION_MAP, DEFAULT_MODEL
from typing import Union

def redact_text(
    text: str,
    model: Union[TokenClassificationTransformer, None] = None,
    redaction_map: dict = DEFAULT_REDACTION_MAP,
    return_preds: bool = False,
):
    if model is None:
        model = TokenClassificationTransformer.from_pretrained(DEFAULT_MODEL)
        
    predictions = model.predict(text)
    text_lines = text.split()
    redacted_text_lines = []
    if any(pred in redaction_map for pred in predictions):
        for pred, text in zip(predictions, text_lines):
            redacted_text_lines.append(redaction_map.get(pred, text))
    else:
        redacted_text_lines = text_lines
    if return_preds:
        return " ".join(redacted_text_lines), predictions
    return " ".join(redacted_text_lines)