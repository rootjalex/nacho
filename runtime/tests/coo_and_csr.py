import nacho_runtime
import torch
import random
from parser import parse_matrix, matrix_list , parse_vector
from plotter import plot, load_and_plot, plot_3, plot_2, plot_bar_graph, plot_bar_graph_2, load_and_plot2
import numpy as np

def _time_on_stream(op, iterations=14, trim=2):
    bench_stream = torch.cuda.Stream()
    bench_stream.wait_stream(torch.cuda.current_stream())
    times = []
    C = None

    with torch.cuda.stream(bench_stream):
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(bench_stream)
            C = op()
            end.record(bench_stream)
            end.synchronize()
            times.append(start.elapsed_time(end))

    # Hand off work completion so callers can safely consume C on default stream.
    torch.cuda.current_stream().wait_stream(bench_stream)

    arr = np.array(times)
    trimmed = np.sort(arr)[trim:-trim] if arr.size > 2 * trim else arr
    avg = trimmed.mean()
    return C, avg


def benchmark_coo_add(start, end, save_and_plot = True):
    df = matrix_list()
    

    nnz = []
    manual_runtime = []
    pytorch_runtime = []

    failed = []

    skip = []
    for i in range(start+1,end):
        if i in skip :
           continue
       
        print(f"Starting iteration {i}")
        A = parse_matrix(df.iloc[i-1]['name'], True).coalesce()
        B = parse_matrix(df.iloc[i]['name'], True).coalesce()

        #print(A,B)
        M = max(A.size(0),B.size(0)) 
        N = max(A.size(1),B.size(1))

        A_torch = torch.sparse_coo_tensor(A.indices(), A.values(), (M,N)).coalesce()
        B_torch = torch.sparse_coo_tensor(B.indices(), B.values(), (M,N)).coalesce()

        A_COO = nacho_runtime.COO(A.indices()[0], A.indices()[1], A.values(), torch.tensor([M,N], dtype=torch.int32))
        B_COO = nacho_runtime.COO(B.indices()[0], B.indices()[1], B.values(), torch.tensor([M,N], dtype=torch.int32))

        nnzA = A_COO.data.numel()
        nnzB = B_COO.data.numel()

        print(f"M {M}, N {N}, nnz {nnzA+nnzB}")

        C_pytorch, pytorch = coo_add(A_torch, B_torch, True)
        C_manual, manual = coo_add(A_COO, B_COO, False)
    
        
        ans = True

        ans = ans and torch.equal(C_pytorch.indices()[0],C_manual.row)
        ans = ans and torch.equal(C_pytorch.indices()[1],C_manual.col)
        ans = ans and torch.equal(C_pytorch.values(),C_manual.data)

        if not ans:
           print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']} {df.iloc[i]['name']}")
           failed.append(i)
           diff_mask_1 = C_pytorch.indices()[0] != C_manual.row
           diff_mask_2 = C_pytorch.indices()[1] != C_manual.col
           diff_mask_3 = C_pytorch.values() != C_manual.data
           idx = -1
           if diff_mask_1.any():
             idx = torch.nonzero(diff_mask_1, as_tuple=True)[0][0].item()
           elif diff_mask_2.any():
             idx = torch.nonzero(diff_mask_2, as_tuple=True)[0][0].item()
           elif diff_mask_3.any():
             idx = torch.nonzero(diff_mask_3, as_tuple=True)[0][0].item()
           print(f"Mismatch at Index: {idx} \n rows: manual {C_manual.row[idx]} , result {C_pytorch.indices()[0][idx]}")
           print(f"cols: manual {C_manual.col[idx]} , result {C_pytorch.indices()[1][idx]}")
           print(f"values: manual {C_manual.data[idx]} , result {C_pytorch.values()[idx]}")

           print(A_COO.row[:10], B_COO.row[:10])
           print(A_COO.col[:10], B_COO.col[:10])
           print(A_COO.data[:10], B_COO.data[:10])

           print(C_manual.row[:10], C_pytorch.indices()[0][:10])

           print(C_manual.col[:10], C_pytorch.indices()[1][:10])

           print(C_manual.data[:10], C_pytorch.values()[:10])
           return
       
        nnz.append(nnzA+nnzB)
        manual_runtime.append(manual)
        #cusparse_runtime.append(cusparse)
        pytorch_runtime.append(pytorch)

    if save_and_plot:
        plot(nnz, manual_runtime, [], pytorch_runtime, f"coo_rows_nnz_{start}-{end}")
    
    print(failed)

def benchmark_csr_add(start,end, save_and_plot=True):
    df = matrix_list()
    

    nnz = []
    manual_runtime = []
    cusparse_runtime = []
    pytorch_runtime = []

    failed = []

    skip = [611]
    for i in range(start+1,end):
        if i in skip :
           continue
       
        print(f"Starting iteration {i}")
        A = parse_matrix(df.iloc[i-1]['name'])
        B = parse_matrix(df.iloc[i]['name'])
       
        M = min(A.size(0),B.size(0)) 
        N = max(A.size(1),B.size(1))


       #print(A,B)
        A_torch = torch.sparse_csr_tensor(A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]], A.values()[:A.crow_indices()[M]], (M,N))
        B_torch = torch.sparse_csr_tensor(B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]], B.values()[:B.crow_indices()[M]], (M,N))

        plus_row = A_torch.crow_indices() + B_torch.crow_indices()


        A_CSR = nacho_runtime.CSR(A_torch.crow_indices(), A_torch.col_indices(), A_torch.values(), torch.tensor([M,N], dtype=torch.int32))
        B_CSR = nacho_runtime.CSR(B_torch.crow_indices(), B_torch.col_indices(), B_torch.values(), torch.tensor([M,N], dtype=torch.int32))


        nnzA = A_CSR.data.numel()
        nnzB = B_CSR.data.numel()
        print(f"M {M}, N {N}, nnz {plus_row.max()}")

        
        C_pytorch, pytorch = torch_add(A_torch, B_torch)
        C_cusparse, cusparse = csr_add(A_CSR, B_CSR, True)
        C_manual, manual = csr_add(A_CSR, B_CSR, False)
        

        ans = True

        ans = ans and torch.equal(C_cusparse.indptr,C_manual.indptr )
        ans = ans and torch.equal(C_cusparse.indices,C_manual.indices )
        ans = ans and torch.equal(C_cusparse.data,C_manual.data )

        ans = ans and torch.equal(C_pytorch.crow_indices(),C_manual.indptr )
        ans = ans and torch.equal(C_pytorch.col_indices(),C_manual.indices )
        ans = ans and torch.equal(C_pytorch.values(),C_manual.data )

        if not ans:
           print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']} {df.iloc[i]['name']}")
           failed.append(i)
           print(A_CSR.indptr[:10], B_CSR.indptr[:10])
           print(A_CSR.indices[:10], B_CSR.indices[:10])
           #C_expected =(A.to_dense() + B.to_dense()).to_sparse_csr()
           print(C_cusparse.indptr, C_manual.indptr)#, C_expected.crow_indices())
           print(C_cusparse.indices, C_manual.indices)
           print(torch.equal(C_cusparse.indptr,C_manual.indptr ))

           failure_reason(C_manual, C_cusparse)
       
        nnz.append(plus_row.max().item())
        manual_runtime.append(manual)
        cusparse_runtime.append(cusparse)
        pytorch_runtime.append(pytorch)

    if save_and_plot:
        plot(nnz, manual_runtime, cusparse_runtime, pytorch_runtime, f"torch_nnz_{start}-{end}")
    
    print(failed)

   
    


def csr_add(A_CSR, B_CSR, use_cusparse):
    return _time_on_stream(
        lambda: nacho_runtime.gpu_csr_add_f32(A_CSR, B_CSR, use_cusparse),
        iterations=14,
        trim=2,
    )

def coo_add(A_COO, B_COO, use_pytorch):
    if use_pytorch:
        op = lambda: (A_COO + B_COO).coalesce()
    else:
        op = lambda: nacho_runtime.gpu_coo_add_f32(A_COO, B_COO)
    return _time_on_stream(op, iterations=14, trim=2)

def torch_add(A,B):
    return _time_on_stream(lambda: A + B, iterations=14, trim=2)

def nacho_csr_add(A_CSR, B_CSR):
    """Time nacho-generated CSR add kernel."""
    return _time_on_stream(
        lambda: nacho_runtime.nacho_csr_add(A_CSR, B_CSR),
        iterations=14, trim=2,
    )

def nacho_csr_add_3(A_CSR, B_CSR, C_CSR):
    """Time nacho-generated fused CSR 3-way add kernel."""
    return _time_on_stream(
        lambda: nacho_runtime.nacho_csr_add_3(A_CSR, B_CSR, C_CSR),
        iterations=14, trim=2,
    )

def nacho_csr_add_3_unfused(A_CSR, B_CSR, C_CSR):
    """Time unfused nacho CSR 3-way add: two sequential nacho_csr_add calls."""
    def op():
        AB = nacho_runtime.nacho_csr_add(A_CSR, B_CSR)
        return nacho_runtime.nacho_csr_add(AB, C_CSR)
    return _time_on_stream(op, iterations=14, trim=2)

def csr_add_3_cusparse_unfused(A_CSR, B_CSR, C_CSR):
    """Time unfused cuSPARSE CSR 3-way add: two sequential cuSPARSE calls."""
    def op():
        AB = nacho_runtime.gpu_csr_add_f32(A_CSR, B_CSR, True)
        return nacho_runtime.gpu_csr_add_f32(AB, C_CSR, True)
    return _time_on_stream(op, iterations=14, trim=2)

def nacho_coo_add(A_COO, B_COO):
    """Time nacho-generated COO add kernel."""
    return _time_on_stream(
        lambda: nacho_runtime.nacho_coo_add(A_COO, B_COO),
        iterations=14, trim=2,
    )

def nacho_coo_mul(A_COO, B_COO):
    """Time nacho-generated COO element-wise mul kernel."""
    return _time_on_stream(
        lambda: nacho_runtime.nacho_coo_mul(A_COO, B_COO),
        iterations=14, trim=2,
    )

def _remove_zeros_coo(C_coo):
    """Remove explicit zeros from a coalesced COO tensor."""
    mask = C_coo.values() != 0
    indices = C_coo.indices()[:, mask]
    values = C_coo.values()[mask]
    return torch.sparse_coo_tensor(indices, values, C_coo.size()).coalesce()

def pytorch_coo_mul(A_coo, B_coo):
    """Time PyTorch sparse COO element-wise mul (including zero removal)."""
    return _time_on_stream(
        lambda: _remove_zeros_coo((A_coo * B_coo).coalesce()),
        iterations=14, trim=2,
    )



def failure_reason(C_manual, C_result):

    if torch.equal(C_manual.indptr, C_result.indptr) is False:
        diff_mask = C_manual.indptr != C_result.indptr
        if diff_mask.any():
            idx = torch.nonzero(diff_mask, as_tuple=True)[0][0].item()
            print(f"Mismatch at index: {idx} \n indptr: manual {C_manual.indptr[idx]} , result {C_result.indptr[idx]}")

    if (len(C_manual.indices) != len(C_result.indices)):
        print(f"Length Mismatch: manual {len(C_manual.indices)} , result {len(C_result.indices)}")
        for i in range(len(C_manual.indices)):
            if C_manual.indices[i] != C_result.indices[i]:
                print(f"Index {i}: manual {C_manual.indices[i]} , result {C_result.indices[i]}")
                return
    relative_error = (C_result.data - C_manual.data).abs() / C_result.data.abs().clamp_min(1)
    if relative_error.max() > 1e-4:
        if torch.equal(C_manual.indices, C_result.indices) is False:
            diff_mask = C_manual.indices != C_result.indices
        diff_mask = relative_error > 1e-4
        if diff_mask.any():
            idx = torch.nonzero(diff_mask, as_tuple=True)[0][0].item()
            print(f"Mismatch at Index: {idx} \n indices: manual {C_manual.indices[idx]} , result {C_result.indices[idx]}")
            print(f"values: manual {C_manual.data[idx]} , result {C_result.data[idx]}")
