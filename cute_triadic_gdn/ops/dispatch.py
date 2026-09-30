import functools

import torch


@functools.lru_cache(maxsize=None)
def _cap(index):
    return torch.cuda.get_device_capability(index)


def sm_arch(device=None):
    """'sm90' on Hopper, 'sm100' on datacenter Blackwell (B200 sm_100, B300 sm_103), 'other' elsewhere."""
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    major, _ = _cap(idx if idx is not None else torch.cuda.current_device())
    return {9: "sm90", 10: "sm100"}.get(major, "other")
