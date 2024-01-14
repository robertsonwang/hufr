import torch
import pytest
from hufr.utils import argmax_with_threshold 

def test_argmax_with_threshold():
    # Test case 1
    tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    threshold1 = 4
    default1 = -1
    result1 = argmax_with_threshold(tensor1, threshold1, default1)
    expected1 = torch.tensor([default1, 1, 1])

    assert torch.equal(result1, expected1)

    # Test case 2
    tensor2 = torch.tensor([[5, 3, 1], [8, 6, 4]])
    threshold2 = 4
    default2 = 0
    result2 = argmax_with_threshold(tensor2, threshold2, default2, dim=1)
    expected2 = torch.tensor([0, default2])

    assert torch.equal(result2, expected2)