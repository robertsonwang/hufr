import numpy as np
from hufr.models.onnx import softmax


def test_softmax():
    # Test case 1: Basic test with a 1D array
    input_array_1d = np.array([1.0, 2.0, 3.0])
    result_1d = softmax(input_array_1d, axis=0)
    expected_result_1d = np.array([0.09003057, 0.24472847, 0.66524096])
    np.testing.assert_allclose(result_1d, expected_result_1d, rtol=1e-5)

    # Test case 2: Basic test with a 3D array
    input_array_2d = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    result_2d = softmax(input_array_2d, axis=2)
    expected_result_2d = np.array(
        [[[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]]
    )
    np.testing.assert_allclose(result_2d, expected_result_2d, rtol=1e-5)
