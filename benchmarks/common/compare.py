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


def csr_close(result, reference, atol=1e-5):
    """Whether two generated CSR results agree, values to within atol.

    An addition visits each output coordinate's two contributions in a fixed order, so
    unlike a contraction its values are comparable directly.
    """
    return (torch.equal(reference.dim_j_offsets, result.dim_j_offsets) and
            torch.equal(reference.dim_j_indices, result.dim_j_indices) and
            torch.allclose(result.values, reference.values, atol=atol))


def csr_structure_equal(result, reference):
    """Whether two generated CSR results have identical sparsity structure.

    Values are deliberately left out. A contraction sums each output coordinate's
    contributions in whatever order its partitioning visits them, and nacho's order
    differs from cuSPARSE's, so a product that cancels to near zero lands on a different
    float in each. The disagreement is in the last bits of a value whose magnitude is
    itself an accident of the summation order, which no fixed tolerance describes well.
    What is worth checking is that both agree on which coordinates the product produces.
    """
    return (torch.equal(reference.dim_j_offsets, result.dim_j_offsets) and
            torch.equal(reference.dim_j_indices, result.dim_j_indices))


def csr_structure_failure_reason(result, reference):
    """Print the first way two generated CSR results disagree on structure."""
    if not torch.equal(result.dim_j_offsets, reference.dim_j_offsets):
        if result.dim_j_offsets.numel() != reference.dim_j_offsets.numel():
            print(f"  row count mismatch: nacho={result.dim_j_offsets.numel() - 1}  "
                  f"ref={reference.dim_j_offsets.numel() - 1}")
            return
        difference = torch.nonzero(result.dim_j_offsets != reference.dim_j_offsets,
                                   as_tuple=True)[0]
        row = difference[0].item()
        print(f"  indptr mismatch at row {row}: nacho={result.dim_j_offsets[row]}  "
              f"ref={reference.dim_j_offsets[row]}")
        return
    if result.dim_j_indices.numel() != reference.dim_j_indices.numel():
        print(f"  nnz mismatch: nacho={result.dim_j_indices.numel()}  "
              f"ref={reference.dim_j_indices.numel()}")
        return
    difference = torch.nonzero(result.dim_j_indices != reference.dim_j_indices,
                               as_tuple=True)[0]
    if difference.numel():
        at = difference[0].item()
        print(f"  column mismatch at {at}: nacho={result.dim_j_indices[at]}  "
              f"ref={reference.dim_j_indices[at]}")


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


def csf3_result_to_coordinates(result):
    """A generated CSF3 result as ((3, nnz) coordinates, values), as torch indices are.

    Walks the level tree outwards: each i coordinate is repeated once per j fibre beneath
    it, and each of those once per stored k. Vectorised rather than looped, because these
    tensors run to tens of millions of non-zeros.
    """
    j_per_i = torch.diff(result.dim_j_offsets.to(torch.int64))
    i_of_j = torch.repeat_interleave(result.dim_i_indices, j_per_i)

    k_per_j = torch.diff(result.dim_k_offsets.to(torch.int64))
    i_of_k = torch.repeat_interleave(i_of_j, k_per_j)
    j_of_k = torch.repeat_interleave(result.dim_j_indices, k_per_j)

    return torch.stack([i_of_k, j_of_k, result.dim_k_indices], dim=0), result.values


def coo3_result_to_coordinates(result):
    """A generated COO3 result as ((3, nnz) coordinates, values)."""
    return torch.stack([result.dim_i_indices, result.dim_j_indices,
                        result.dim_k_indices], dim=0), result.values


def coordinates_equal(actual, expected, rtol=1e-4):
    """Whether two (coordinates, values) pairs agree; (ok, reason).

    Coordinates have to match exactly, values only to within rtol: an addition visits its
    operands in a different order in each implementation.
    """
    actual_coordinates, actual_values = actual
    expected_coordinates, expected_values = expected

    if actual_values.numel() != expected_values.numel():
        return False, (f"nnz mismatch: {actual_values.numel()} vs "
                       f"{expected_values.numel()}")

    for axis, name in enumerate("ijk"):
        if not torch.equal(actual_coordinates[axis].to(torch.int64),
                           expected_coordinates[axis].to(torch.int64)):
            differing = torch.nonzero(
                actual_coordinates[axis].to(torch.int64) !=
                expected_coordinates[axis].to(torch.int64), as_tuple=True)[0]
            at = differing[0].item()
            return False, (f"{name} mismatch at {at}: "
                           f"{actual_coordinates[axis][at]} vs {expected_coordinates[axis][at]}")

    if not torch.allclose(actual_values, expected_values, rtol=rtol):
        relative = ((actual_values - expected_values).abs() /
                    expected_values.abs().clamp_min(1))
        at = relative.argmax().item()
        return False, (f"value mismatch at {at}: {actual_values[at]} vs "
                       f"{expected_values[at]}")
    return True, "ok"


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
