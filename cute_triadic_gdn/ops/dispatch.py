import functools

import torch


@functools.lru_cache(maxsize=None)
def _cap(index):
    return torch.cuda.get_device_capability(index)


def sm_arch(device=None):
    """'sm90' on Hopper (the only architecture the kernels are built for), 'other' elsewhere."""
    idx = torch.cuda.current_device() if device is None else torch.device(device).index
    major, _ = _cap(idx if idx is not None else torch.cuda.current_device())
    return "sm90" if major == 9 else "other"
