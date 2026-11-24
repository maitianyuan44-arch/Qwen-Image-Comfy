"""Auto-imported by Python at interpreter startup when this directory is
on PYTHONPATH.

handler.py launches the ComfyUI subprocess with this worker's src/
directory prepended to PYTHONPATH, so this file runs inside ComfyUI's
own interpreter before any of its code. It disables PyTorch's cuDNN
SDPA backend: on this stack (torch cu130 + cuDNN 9) that backend fails
inside Qwen-Image 2.1 attention with "cuDNN Frontend error: No valid
execution plans built" (seen on both a PRO 6000 MIG slice and a plain
RTX A5000), while PyTorch's flash / mem-efficient SDPA backends handle
the same shapes fine. Convolutions keep using cuDNN — only the SDPA
backend selection is opted out. Harmless if imported by the worker's
own interpreter too (it does no GPU attention work).
"""

try:
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
