import nacho_runtime
import torch
import random       
from parser import parse_matrix, matrix_list , parse_vector
from plotter import plot, load_and_plot, plot_3, plot_2, plot_bar_graph, plot_bar_graph_2, load_and_plot2
import numpy as np
import random
from coo_and_csr import failure_reason

def spgemm_benchmark(A_CSR, B_CSR, use_cusparse):
    torch.cuda.synchronize()
    iter = 14
    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
    
        C = nacho_runtime.spgemm(
            A_CSR,
            B_CSR,
            use_cusparse,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    np.std(trimmed)
    avg = trimmed.mean()

    return C, avg



def spgemm(start,end, save_and_plot=True):
    df = matrix_list()

    nnz = []
    manual_runtime = []
    cusparse_runtime = []

    failed = []

    skip = [611]
    for i in range(start+1,end):
        if i in skip :
           continue
       
        print(f"Starting iteration {i}")
        A = parse_matrix(df.iloc[i-1]['name'])
        B = parse_matrix(df.iloc[i]['name'])

        # if A is square also benchmark AxA 
        if(A.size(0)==A.size(1)):
            print(f"Also benchmarking A x A for matrix {df.iloc[i-1]['name']}")

            M = A.size(0)
            A_CSR = nacho_runtime.CSR(A.crow_indices(), A.col_indices(), A.values(), torch.tensor([M,M], dtype=torch.int32))
            
            print(f"M {M}, nnz {A.crow_indices().max().item() * 2}")
            
            C_cusparse, cusparse = spgemm_benchmark(A_CSR, A_CSR, True)
            C_manual, manual = spgemm_benchmark(A_CSR, A_CSR, False)

            ans = True
            ans = ans and torch.equal(C_cusparse.indptr,C_manual.indptr )
            ans = ans and torch.equal(C_cusparse.indices,C_manual.indices )
            # relative_error = (C_cusparse.data - C_manual.data).abs() / C_cusparse.data.abs().clamp_min(1)
            # absolute_error = (C_cusparse.data - C_manual.data).abs()
            # print(relative_error)
            # ans = ans and (torch.max(relative_error) < 1e-3 or torch.max(absolute_error) < 1e-1)
            # print(C_cusparse.data[torch.where(relative_error >= 1e-4)])
            # print(C_manual.data[torch.where(relative_error >= 1e-4)])

            print("Cusparse time:", cusparse)
            print("Manual time:", manual)

            if not ans:
                # print("A:")
                # print(A_CSR.indptr)
                # print(A_CSR.indices)
                # print(A_CSR.data)

                # print("C cusparse:")    
                # print(C_cusparse.indptr)
                # print(C_cusparse.indices)
                # print(C_cusparse.data)

                # print("C manual:")
                # print(C_manual.indptr)
                # print(C_manual.indices)
                # print(C_manual.data)

                print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']}")
                failed.append(i)
                # print(A_CSR.indptr[:10])
                # print(A_CSR.indices[:10])
                # #C_expected =(A.to_dense() + B.to_dense()).to_sparse_csr()
                # print(C_cusparse.indptr, C_manual.indptr)#, C_expected.crow_indices())
                # print(C_cusparse.indices, C_manual.indices)
                # print(torch.equal(C_cusparse.indptr,C_manual.indptr ))

                failure_reason(C_manual, C_cusparse)
                break

            nnz.append(A.crow_indices().max().item() * 2)
            manual_runtime.append(manual)
            cusparse_runtime.append(cusparse)

        M = A.size(0)
        K = min(A.size(1),B.size(0))
        if(K!=A.size(1)):
            print("Skipping since inner dimensions don't match")
            continue
        N = B.size(1)


       #print(A,B)
        A_torch = torch.sparse_csr_tensor(A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]], A.values()[:A.crow_indices()[M]], (M,K))
        B_torch = torch.sparse_csr_tensor(B.crow_indices()[:K+1], B.col_indices()[:B.crow_indices()[K]], B.values()[:B.crow_indices()[K]], (K,N))

        plus_row = A_torch.crow_indices().max() + B_torch.crow_indices().max()


        A_CSR = nacho_runtime.CSR(A_torch.crow_indices(), A_torch.col_indices(), A_torch.values(), torch.tensor([M,K], dtype=torch.int32))
        B_CSR = nacho_runtime.CSR(B_torch.crow_indices(), B_torch.col_indices(), B_torch.values(), torch.tensor([K,N], dtype=torch.int32))


        # nnzA = A_CSR.data.numel()
        # nnzB = B_CSR.data.numel()
        print(f"M {M}, K{K}, N {N}, nnz {plus_row}")

        
        C_cusparse, cusparse = spgemm_benchmark(A_CSR, B_CSR, True)
        C_manual, manual = spgemm_benchmark(A_CSR, B_CSR, False)

        print("Cusparse time:", cusparse)
        print("Manual time:", manual)

        ans = True

        ans = ans and torch.equal(C_cusparse.indptr,C_manual.indptr )
        ans = ans and torch.equal(C_cusparse.indices,C_manual.indices )
        # relative_error = (C_cusparse.data - C_manual.data).abs() / C_cusparse.data.abs().clamp_min(1)
        # absolute_error = (C_cusparse.data - C_manual.data).abs()
        # print(relative_error)
        # ans = ans and (torch.max(relative_error) < 1e-3 or torch.max(absolute_error) < 1e-1)

        if not ans:
           print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']} {df.iloc[i]['name']}")
           failed.append(i)
           print(A_CSR.indptr[:10], B_CSR.indptr[:10])
           print(A_CSR.indices[:10], B_CSR.indices[:10])
           #C_expected =(A.to_dense() + B.to_dense()).to_sparse_csr()
           print(C_cusparse.indptr[-10:], C_manual.indptr[-10:])#, C_expected.crow_indices())
           print(C_cusparse.indices[:10], C_manual.indices[:10])
           print(torch.equal(C_cusparse.indptr,C_manual.indptr ))

           failure_reason(C_manual, C_cusparse)
           break
       
        nnz.append(plus_row.max().item())
        manual_runtime.append(manual)
        cusparse_runtime.append(cusparse)

    if save_and_plot:
        plot(nnz, manual_runtime, cusparse_runtime, [], "spgemm") #f"spgemm_{start}-{end}")
    
    print(failed)


def spgemm_test():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    x_ind = torch.tensor([0,2,4, 7,8,9, 10, 15, 17], dtype=torch.int32, device="cuda")
    x_data = torch.tensor([1.0 for _ in range(x_ind.shape[0])], dtype=torch.float32, device="cuda")
    x = nacho_runtime.CVector(x_ind, x_data, 30)

    A_row  = torch.tensor([0,10,12,18,27], dtype=torch.int32, device="cuda")
    A_col  = torch.tensor([0,2,6,8,9,10,11,17,18,19,   0,23,   10, 11, 14, 15, 17, 18,   0,4,6,7,9,10,11,13,14], dtype=torch.int32, device="cuda")
    A_data = torch.tensor([1.0 for _ in range(A_col.shape[0])], dtype=torch.float32, device="cuda")
    A = nacho_runtime.CSR(
        torch.tensor(A_row, dtype=torch.int32, device="cuda"),
        torch.tensor(A_col, dtype=torch.int32, device="cuda"),
        torch.tensor(A_data, dtype=torch.float32, device="cuda"),
        torch.tensor([4,25], dtype=torch.int32)
    )

    B_row =[0]
    B_col = torch.tensor([], dtype=torch.int32, device="cuda")
    for i in range(25):
        r = random.randint(1, 10)
        B_row.append(B_row[-1] + r)
        for _ in range(r):
           col = torch.randperm(25, device="cuda")[:r]
           col,_ = col.sort()
        B_col = torch.cat((B_col, col))

    B_data = torch.tensor([1.0 for _ in range(B_col.shape[0])], dtype=torch.float32, device="cuda")
    B = nacho_runtime.CSR(
        torch.tensor(B_row, dtype=torch.int32, device="cuda"),
        torch.tensor(B_col, dtype=torch.int32, device="cuda"),
        torch.tensor(B_data, dtype=torch.float32, device="cuda"),
        torch.tensor([25,25], dtype=torch.int32)
    )
    print("A:")
    print(A.indptr)
    print(A.indices)
    print(A.data)

    print("B:")
    print(B.indptr)
    print(B.indices)
    print(B.data)

    C = nacho_runtime.spgemm(A, B, False)
    D = nacho_runtime.spgemm(A, B, True) 

    ans = True
    ans &= torch.equal(C.indptr, D.indptr)
    ans &= torch.equal(C.indices, D.indices)
    ans &= torch.equal(C.data, D.data)

    print(ans)
    
