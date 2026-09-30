"""cute_triadic_gdn: CuTe DSL kernels for Triadic linear attention (Triadic Gated DeltaNet) on Hopper.

    chunk_gdn_joint   Triadic GDN with autograd (torch reference off Hopper)
    gdn_joint_call    the same behind torch.library custom ops, for torch.compile
    conv_split_act_call   one causal depthwise conv with SiLU on the leading channels only
"""
from .ops.gdn_joint import chunk_gdn_joint
from .ops.joint_torch_ops import gdn_joint_call
from .ops.conv_split_act import conv_split_act_call

__all__ = ["chunk_gdn_joint", "gdn_joint_call", "conv_split_act_call"]
__version__ = "1.0.0"
