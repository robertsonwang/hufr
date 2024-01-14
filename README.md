# 🤗 Redactions 

HuggingFace Redactions (`hufr`) is a Python wrapper for HuggingFace token classification models to help redact personal identifiable information from free text. This project is not associated with the official HuggingFace organization, just a fun side project for this individual contributor. 

# Installation
To install this package, first clone the repository and then run `pip install hufr/`

# Usage

See below for an example snippet to load a specific token classification library from the HuggingFace model zoo:

```python
from hufr.models import TokenClassificationTransformer
from hufr.redact import redact_text
from transformers.tokenization_utils_base import BatchEncoding

model_path = "dslim/bert-base-NER"
model = TokenClassificationTransformer(
    model=model_path,
    tokenizer=model_path
)

text = "Hello! My name is Rob"
redact_text(
    text,
    redaction_map={'PER': '<PERSON>'},
    model=model
)
```

This will output:

`"Hello! My name is \<PERSON\>"'

If you don't want to instantiate a model and supply a specific token classification model, then you can simply rely on the repository defaults for a quick and simple redaction:

```python
from hufr.redact import redact_text

text = "Hello! My name is Rob"
redact_text(text)
```

See the `constants.py` module for default model paths and default entity to redaction mapping.
