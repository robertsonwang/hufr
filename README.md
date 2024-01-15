# 🤗 Redactions

HuggingFace Redactions (`hufr`) is a Python wrapper for HuggingFace token classification models to streamline the redaction of personal identifiable information from free text. This project is not associated with the official HuggingFace organization, just a fun side project for this individual contributor.

# Installation
To install this package, run `pip install hufr`

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

`"Hello! My name is <PERSON>"`

If you don't want to instantiate a model and supply a specific token classification model, then you can simply rely on the repository defaults for a quick and simple redaction:

```python
from hufr.redact import redact_text

text = "Hello! My name is Rob"
redact_text(text)
```

To get the predicted entity for each word in the original text:

```python
from hufr.redact import redact_text

text = "Hello! My name is Rob"
redact_text(text, return_preds=True)
```

This will output:
`"Hello! My name is <PERSON>", ['O', 'O', 'O', 'O', 'PER']`

By default, personal identifiable information is predicted by the [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) model where entities are mapped to redactions using the following mapping table:


```python
'PER': '<PERSON>',
'MIS': '<OTHER>',
'ORG': '<ORGANIZATION>',
'LOC': '<LOCATION>'
```
