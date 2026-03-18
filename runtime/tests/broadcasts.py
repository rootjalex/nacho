import nacho_runtime
import torch
import random       
from parser import parse_matrix, matrix_list , parse_vector
from plotter import plot, load_and_plot, plot_3, plot_2, plot_bar_graph, plot_bar_graph_2, load_and_plot2
import numpy as np


def csr_add(A_CSR, B_CSR, use_cusparse):
    torch.cuda.synchronize()
    iter = 14
    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
    
        C = nacho_runtime.gpu_csr_add_f32(
            A_CSR,
            B_CSR,
            use_cusparse,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    avg = trimmed.mean()

    return C, avg


def benchmark_broadcast():
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
        torch.tensor([4,30], dtype=torch.int32)
    )
    #print(torch.tensor(x_ind.repeat(4), dtype=torch.int32, device="cuda"))
    B = nacho_runtime.CSR(
        torch.tensor([x_ind.shape[0] *i for i in range(5)], dtype=torch.int32, device="cuda"),
        torch.tensor(x_ind.repeat(4), dtype=torch.int32, device="cuda"),
        torch.tensor(x_data.repeat(4), dtype=torch.float32, device="cuda"),
        torch.tensor([4,30], dtype=torch.int32)
    )
   

    iter = 14 

    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
        C = nacho_runtime.gpu_broadcast_xA(x, A)
        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        times.append(elapsed_time_ms)
    
    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    xA_time = trimmed.mean()


    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
        D = nacho_runtime.gpu_csr_add_f32(A, B, False)
        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)
        times.append(elapsed_time_ms)
    
    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    csr_time = trimmed.mean()



    ans = True
    ans = ans and torch.equal(D.indptr, C.indptr)
    if( not ans):
        print("INDPTR MISMATCH")
        print(D.indptr)
        print(C.indptr)
    ans = ans and torch.equal(D.indices, C.indices)
    if( not ans):
        print("INDICES MISMATCH")
        print(D.indices)
        print(C.indices)
    ans = ans and torch.equal(D.data, C.data )
    if( not ans):
        print("DATA MISMATCH")
        print(D.data)
        print(C.data)
    return ans, xA_time, csr_time

def broadcast_test_xA():
    
    
