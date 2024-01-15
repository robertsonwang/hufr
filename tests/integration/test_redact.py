from hufr.redact import redact_text


def test_redact_text_default_model():
    """Test the redact_text loads a default model and default redaction scheme"""
    # Test input text
    input_text = "This is a test. My name is John."

    # Expected redacted text
    expected_redacted_text = "This is a test. My name is <PERSON>."

    # Expected model predictions
    expected_return_value = ["O", "O", "O", "O", "O", "O", "O", "PER"]

    # Call the redact_text function
    result, pred = redact_text(text=input_text, return_preds=True)

    # Assert the redacted text is as expected
    assert result == expected_redacted_text
    assert pred == expected_return_value
