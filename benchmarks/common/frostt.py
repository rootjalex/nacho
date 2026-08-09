"""Loading FROSTT tensors and building operand pairs from them."""

import numpy as np
import torch

import nacho

import config


def load_frostt(filename, dims):
    """Read a .tns file into (coordinates, values), coordinates shaped (nnz, 3)."""
    path = config.FROSTT_DIR / f"{filename}.tns"
    if not path.is_file():
        raise FileNotFoundError(
            f"FROSTT tensor not found: {path}\n"
            f"Set NACHO_FROSTT_DIR to the directory holding the FROSTT .tns files, or "
            f"edit FROSTT_TENSORS in benchmarks/config.py.")
    parsed = nacho.parse3D_i32_f32(str(path), *dims)
    coordinates = np.stack([
        np.asarray(parsed.row), np.asarray(parsed.col), np.asarray(parsed.dep)
    ], axis=1)
    return coordinates, np.asarray(parsed.data)


def shift_coordinates(coordinates, values, shape):
    """A copy with every coordinate advanced by one, clamped to the tensor's extent.

    Clamping collapses entries at the far face onto their neighbours, so duplicates are
    summed and the result is re-sorted. Done with numpy rather than a Python loop: these
    tensors run to tens of millions of non-zeros.

    Shifting is what gives a second operand of the same size and shape as the first with a
    controlled amount of overlap, without needing a second file.
    """
    shifted = np.minimum(coordinates + 1, np.array(shape, dtype=coordinates.dtype) - 1)

    order = np.lexsort((shifted[:, 2], shifted[:, 1], shifted[:, 0]))
    shifted, ordered_values = shifted[order], values[order]

    starts = np.ones(len(shifted), dtype=bool)
    starts[1:] = np.any(shifted[1:] != shifted[:-1], axis=1)
    run_starts = np.flatnonzero(starts)
    return shifted[run_starts], np.add.reduceat(ordered_values, run_starts)


def to_torch(coordinates, values, shape, device):
    """A coalesced torch sparse COO tensor over the same coordinates."""
    indices = torch.from_numpy(np.ascontiguousarray(coordinates.T)).to(
        dtype=torch.int64, device=device)
    data = torch.from_numpy(np.ascontiguousarray(values)).to(
        dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, data, tuple(shape)).coalesce()


def iter_shifted_pairs(device):
    """Yield (name, dims, a, b) for each configured tensor, b a shifted copy of a.

    Each element is (coordinates, values). Fails rather than skipping when the dataset is
    missing, so a mis-set NACHO_FROSTT_DIR is obvious immediately.
    """
    config.require_dataset_dir(config.FROSTT_DIR, "FROSTT", "NACHO_FROSTT_DIR")
    for name, filename, dims in config.FROSTT_TENSORS:
        print(f"\n=== {name}  shape={dims} ===")
        a_coordinates, a_values = load_frostt(filename, dims)
        b_coordinates, b_values = shift_coordinates(a_coordinates, a_values, dims)
        print(f"  nnzA={len(a_values):,}  nnzB={len(b_values):,}")
        yield name, dims, (a_coordinates, a_values), (b_coordinates, b_values)
