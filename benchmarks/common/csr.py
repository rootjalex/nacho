"""Moving CSR buffers between the tensor classes the benchmarks call into.

Three classes carry the same three arrays: torch's sparse CSR, the hand-written kernels'
`BaselineCSR_*`, and the generated `CSR_*`. Nothing is copied here — each conversion
rewraps the buffers already on the device.
"""

import torch

import nacho


def to_baseline_csr(csr, device):
    """A torch sparse CSR as the class the hand-written comparison kernels take."""
    cls = nacho.BaselineCSR_cpu if device == "cpu" else nacho.BaselineCSR_gpu
    return cls(
        csr.crow_indices().to(dtype=torch.int32, device=device),
        csr.col_indices().to(dtype=torch.int32, device=device),
        csr.values().to(dtype=torch.float32, device=device),
        torch.tensor([csr.shape[0], csr.shape[1]], dtype=torch.int32),
    )


def as_nacho_csr(baseline, device="cuda"):
    """A baseline kernel's result as the class the generated kernels take.

    Nothing is copied, so the baseline result has to outlive whatever reads through it.
    """
    cls = nacho.CSR_cpu if device == "cpu" else nacho.CSR_gpu
    return cls(baseline.indptr, baseline.indices, baseline.values, baseline.shape)
