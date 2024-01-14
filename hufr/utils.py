import torch

def argmax_with_threshold(tensor, threshold, default, dim=0):
    """
    Helper function to get the maximum indices of a tensor along a given dimension while using a
    threshold value. If all tensor elements are below the threshold value then return the value
    specified by the default.
    """
    max_indices = torch.argmax(tensor, dim=dim)
    max_values = torch.max(tensor, dim=dim).values

    below_threshold = max_values <= threshold
    max_indices[below_threshold] = default

    return max_indices