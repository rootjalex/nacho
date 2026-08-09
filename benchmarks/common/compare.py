"""Comparing generated CSR results against PyTorch, and summarising runtimes."""

import numpy as np
import torch


def csr_equal(result, reference):
    """Whether a generated CSR result matches a torch sparse CSR tensor exactly."""
    return (torch.equal(reference.crow_indices(), result.dim_j_offsets) and
            torch.equal(reference.col_indices(), result.dim_j_indices) and
            torch.equal(reference.values(), result.values))


def csr_failure_reason(result, reference):
    """Print the first way a generated CSR result diverges from the reference."""
    if not torch.equal(result.dim_j_offsets, reference.crow_indices()):
        difference = result.dim_j_offsets != reference.crow_indices()
        if difference.any():
            index = torch.nonzero(difference, as_tuple=True)[0][0].item()
            print(f"  indptr mismatch at {index}: nacho={result.dim_j_offsets[index]}  "
                  f"ref={reference.crow_indices()[index]}")
    if len(result.dim_j_indices) != reference.col_indices().numel():
        print(f"  nnz mismatch: nacho={len(result.dim_j_indices)}  "
              f"ref={reference.col_indices().numel()}")
        return
    relative_error = ((reference.values() - result.values).abs() /
                      reference.values().abs().clamp_min(1))
    if relative_error.max() > 1e-4:
        index = torch.nonzero(relative_error > 1e-4, as_tuple=True)[0][0].item()
        print(f"  values mismatch at {index}: nacho={result.values[index]}  "
              f"ref={reference.values()[index]}")


def csr_result_to_coo(result):
    """A generated CSR result as (rows, cols, values), with explicit zeros dropped.

    An elementwise product stores a zero wherever both operands were non-zero but the
    product cancelled; torch's sparse multiply prunes those, so they have to go before
    the two can be compared.
    """
    empty = torch.tensor([], dtype=torch.int32, device=result.dim_j_indices.device)
    if result.dim_j_offsets is None or result.dim_j_offsets.numel() == 0:
        return empty, empty, empty.to(torch.float32)

    rows = torch.repeat_interleave(
        torch.arange(result.dim_j_offsets.numel() - 1,
                     dtype=result.dim_j_indices.dtype,
                     device=result.dim_j_indices.device),
        torch.diff(result.dim_j_offsets),
    )
    kept = result.values != 0
    return rows[kept], result.dim_j_indices[kept], result.values[kept]


def csr_allclose(result, reference, tolerance=1e-4):
    """Whether a generated CSR result matches a torch CSR up to floating point error.

    A contraction sums a coordinate's contributions in whatever order the reduction
    happens to visit them, so values agree only approximately even when the sparsity
    structure is identical.
    """
    if not (torch.equal(reference.crow_indices(), result.dim_j_offsets) and
            torch.equal(reference.col_indices(), result.dim_j_indices)):
        return False
    if reference.values().numel() == 0:
        return True
    relative = ((reference.values() - result.values).abs() /
                reference.values().abs().clamp_min(1))
    return bool(relative.max() < tolerance)


def csr_as_coo_equal(result, reference):
    """Whether a generated CSR result matches a coalesced torch sparse COO tensor."""
    rows, cols, values = csr_result_to_coo(result)
    return (torch.equal(reference.indices()[0], rows) and
            torch.equal(reference.indices()[1], cols) and
            torch.equal(reference.values(), values))


def coo_equal(result, reference, prune=False):
    """Whether a generated COO result matches a coalesced torch sparse COO tensor.

    prune drops stored zeros from the result first, which an elementwise product needs:
    nacho stores a position wherever both operands were non-zero, torch does not.
    """
    rows, cols, values = result.dim_i_indices, result.dim_j_indices, result.values
    if prune:
        kept = values != 0
        rows, cols, values = rows[kept], cols[kept], values[kept]
    return (torch.equal(reference.indices()[0], rows) and
            torch.equal(reference.indices()[1], cols) and
            torch.equal(reference.values(), values))


def prune_zeros(sparse):
    """A torch sparse tensor as coalesced COO with explicit zeros removed."""
    coo = sparse.to_sparse_coo()
    kept = coo.values() != 0
    return torch.sparse_coo_tensor(coo.indices()[:, kept], coo.values()[kept],
                                   coo.size()).coalesce()


def summarize(tag, nacho_runtimes, baseline_label, baseline_runtimes, failed):
    """Print runtime distributions and the speedup of nacho over one baseline."""
    if not nacho_runtimes:
        print("No results to summarise.")
        return

    from scipy.stats import gmean

    nacho = np.array(nacho_runtimes)
    baseline = np.array(baseline_runtimes)
    speedup = baseline / nacho

    rule = "-" * 60
    print(f"\n{rule}")
    print(f"  BENCHMARK SUMMARY  [{tag}]  ({len(nacho)} matrix pairs)")
    print(rule)
    print("\n  Runtimes (ms):")
    print(f"    Nacho:    mean={nacho.mean():.4f}  median={np.median(nacho):.4f}  "
          f"std={nacho.std():.4f}")
    print(f"    {baseline_label:<9} mean={baseline.mean():.4f}  "
          f"median={np.median(baseline):.4f}  std={baseline.std():.4f}")
    print(f"\n  Speedup over {baseline_label}:")
    print(f"    Mean:    {speedup.mean():.4f}x")
    print(f"    Geomean: {gmean(speedup):.4f}x")
    print(f"    Median:  {np.median(speedup):.4f}x")
    print(f"    Min:     {speedup.min():.4f}x  (idx {speedup.argmin()})")
    print(f"    Max:     {speedup.max():.4f}x  (idx {speedup.argmax()})")
    print(f"    % faster: {np.mean(speedup > 1.0) * 100:.1f}%")
    print(f"\n  Failures: {len(failed)} / {len(nacho)}  {failed if failed else ''}")
    print(rule)
