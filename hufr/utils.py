import torch

def argmax_with_threshold(tensor, threshold, default, dim=0):
    """
    Helper function to get the maximum indices of a tensor along a given dimension while using a
    threshold value. If all tensor elements are below the threshold value then return the value
    specified by the default.

    Args:
        tensor (torch.Tensor): Input tensor.
        threshold (float): Threshold value for filtering maximum values.
        default: Value to be used if all elements are below the threshold.
        dim (int, optional): Dimension along which to compute the argmax. Defaults to 0.

    Returns:
        torch.Tensor: Tensor containing the indices of maximum values. If all elements are below the
        threshold, the tensor is filled with the specified default value.

    Note:
        This function calculates the argmax and maximum values along the specified dimension.
        If all maximum values are below the threshold, the function replaces the indices with the
        specified default value.
    """
    max_indices = torch.argmax(tensor, dim=dim)
    max_values = torch.max(tensor, dim=dim).values

    below_threshold = max_values <= threshold
    max_indices[below_threshold] = default

    return max_indices
