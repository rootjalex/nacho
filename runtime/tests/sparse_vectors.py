
import nanobind_cuda_example
import torch
import random
from parser import parse_matrix, matrix_list , parse_vector
from plotter import plot, load_and_plot, plot_3, plot_2, plot_bar_graph, plot_bar_graph_2, load_and_plot2
import numpy as np



def test_mergepath(size, length, a = "", b="",c =""):

    expr = False

    
    A_index, A_data, size_A = parse_vector(a)
    B_index, B_data, size_B = parse_vector(b)
    C_index, C_data, size_C = parse_vector(c)

    size = max(size_A, size_B, size_C)
    print(f"Vector sizes: A {size_A}, B {size_B}, C {size_C}, unified size {size}")

    print(f"Vector nnz: A {A_index.shape[0]}, B {B_index.shape[0]}, C {C_index.shape[0]}")
    # Generate random sparse vectors A, B, C
    # print(length)
    # a_length = int(length)#int(size * sparsity_a)
    # b_length = int(length)#int(size * sparsity_b)
    # c_length = int(length)#int(size * sparsity_c)
    # print(a_length, b_length, c_length)
    

    # type = torch.int64
    # if 2*a_length > size:
    #     a_index = torch.randperm(size, dtype=type, device="cuda")[:a_length] + 1
        
    # else:
    #     a_index = torch.randint(1, size, (a_length,), dtype=type, device="cuda")
    #     a_index = torch.unique(a_index)
    
    # #a_index_cpu = torch.tensor(a_index, device="cpu")
    # #a_index_cpu,_ = torch.sort(a_index_cpu)
    # #a_index = a_index_cpu.to("cuda")
    # a_index,_ = torch.sort(a_index)

    # torch.cuda.empty_cache() 
    
    # if 2*b_length > size:
    #     b_index = torch.randperm(size, dtype=type, device="cuda")[:b_length] + 1
        
    # else:
    #     b_index = torch.randint(1, size, (b_length,), dtype=type, device="cuda")
    #     b_index = torch.unique(b_index)

    # # b_index_cpu = torch.tensor(b_index, device="cpu")
    # # b_index_cpu,_ = torch.sort(b_index_cpu)
    # # b_index = b_index_cpu.to("cuda")
    # b_index,_ = torch.sort(b_index)

    # torch.cuda.empty_cache()

    # if 2*c_length > size:
    #     c_index = torch.randperm(size, dtype=type, device="cuda")[:c_length] + 1
    # else:
    #     c_index = torch.randint(1, size, (c_length,), dtype=type, device="cuda")
    #     c_index = torch.unique(c_index)

    # # c_index_cpu = torch.tensor(c_index, device="cpu")
    # # c_index_cpu,_ = torch.sort(c_index_cpu)
    # # c_index = c_index_cpu.to("cuda")
    # c_index,_ = torch.sort(c_index)

    # torch.cuda.empty_cache()
    # A_index = a_index
    # A_data = torch.rand(a_index.shape[0], dtype=torch.float32, device="cuda")
    # torch.cuda.empty_cache()
    # B_index = b_index
    # B_data = torch.rand(b_index.shape[0], dtype=torch.float32, device="cuda")
    # torch.cuda.empty_cache()
    # C_index = c_index
    # C_data = torch.rand(c_index.shape[0], dtype=torch.float32, device="cuda")
    # torch.cuda.empty_cache()
    # A_index = torch.tensor([0,1,5, 7, 12, 13, 20, 21], dtype=torch.int32, device=torch.device("cuda"))
    # A_data = torch.tensor([3.0]*A_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))
    
    # B_index = torch.tensor([0,2,4,5 ,6,7,8,20,22,23], dtype=torch.int32, device=torch.device("cuda"))
    # B_data = torch.tensor([2.0]*B_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))

    # C_index =  torch.tensor([0,1,3,7,8,20,21,22,23], dtype=torch.int32, device=torch.device("cuda"))
    # C_data = torch.tensor([1.0]*C_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))

    # print("A:", A_index[-10:], A_data[-10:])
    # print("B:", B_index[-10:], B_data[-10:])
    # print("C:", C_index[-10:], C_data[-10:])
    # print(C_index.shape[0])
    A = nanobind_cuda_example.CVector64(
        A_index, 
        A_data,
        size)
    torch.cuda.empty_cache()
    B = nanobind_cuda_example.CVector64(
        B_index,
        B_data,
        size
        )
    torch.cuda.empty_cache()
    C = nanobind_cuda_example.CVector64(
        C_index,
        C_data,
        size
        )
    torch.cuda.empty_cache()
    print("Starting Benchmarks")

    iter = 14
    torch.cuda.synchronize()

    iter = 14
    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        
        D_full_fusion = nanobind_cuda_example.gpu_sss_mergepath_test64(A,B,C, 3, expr)
        #end.record()
        #torch.cuda.synchronize()
        #times.append(start.elapsed_time(end))
        times.append([D_full_fusion.time_1,D_full_fusion.time_2, D_full_fusion.time_3])
    # print(D_full_fusion.indices, D_full_fusion.data)
    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)
    # vals = np.sort(arr)
    # trimmed = vals[2: -2]  # remove lowest and highest `trim` values
    # mean_val = trimmed.mean()
    # result = mean_val
    

    full_lb = result
    print("-----------------full done-----------------------")
    torch.cuda.synchronize()

    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        D_partial_fusion = nanobind_cuda_example.gpu_sss_mergepath_test64(A,B,C, 2, expr)
        #end.record()
        #torch.cuda.synchronize()
        #times.append(start.elapsed_time(end))
        times.append([ D_partial_fusion.time_1, D_partial_fusion.time_2,  D_partial_fusion.time_3 ])

    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)
    # vals = np.sort(arr)
    # trimmed = vals[2: -2]  # remove lowest and highest `trim` values
    # mean_val = trimmed.mean()
    # result = mean_val

    partial_lb = result
    # print(D_partial_fusion.indices, D_partial_fusion.data)
    print("-----------------partial done------------------------")

    torch.cuda.synchronize()

    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        D_nofusion = nanobind_cuda_example.gpu_sss_mergepath_test64(A,B,C, 1, expr)
        #end.record()
        #torch.cuda.synchronize()
        #times.append(start.elapsed_time(end))
        times.append([ D_nofusion.time_1, D_nofusion.time_2,  D_nofusion.time_3 ])
    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)
    # vals = np.sort(arr)
    # trimmed = vals[2: -2]  # remove lowest and highest `trim` values
    # mean_val = trimmed.mean()
    # result = mean_val

    nofusion_lb = result
    # print(D_nofusion.indices, D_nofusion.data)
    print("------------------no done-----------------------")

    ans = torch.equal(D_full_fusion.indices, D_partial_fusion.indices)
    ans = ans and torch.equal(D_full_fusion.data, D_partial_fusion.data)
    ans = ans and torch.equal(D_full_fusion.indices, D_nofusion.indices)
    ans = ans and torch.equal(D_full_fusion.data, D_nofusion.data)
    # print(" full partial indices", torch.equal(D_full_fusion.indices, D_partial_fusion.indices))
    # print(" full nofusion indices", torch.equal(D_full_fusion.indices, D_nofusion.indices))
    # print(" full partial data", torch.equal(D_full_fusion.data, D_partial_fusion.data))
    # print(" full nofusion data", torch.equal(D_full_fusion.data, D_nofusion.data))
    #print(D_full_fusion.data != D_partial_fusion.data)
    if(not ans):
        print("Mismatch detected in results!")
        print(D_nofusion.indices.shape[0], D_partial_fusion.indices.shape[0])
        
        print(torch.equal(D_full_fusion.indices[:D_nofusion.indices.shape[0]], D_nofusion.indices))
        print(D_full_fusion.indices[-20:], D_nofusion.indices[-20:])
        for i in range(D_full_fusion.indices.shape[0]):
            if D_full_fusion.indices[i] != D_nofusion.indices[i]:
                print(f"Mismatch at index {i} in indices: full {D_full_fusion.indices[i]}, nofusion {D_nofusion.indices[i]}")
                break
        for i in range(D_full_fusion.indices.shape[0]):
            if D_full_fusion.indices[i] != D_partial_fusion.indices[i]:
                print(f"Mismatch at index {i} in indices: full {D_full_fusion.indices[i]}, partial {D_partial_fusion.indices[i]}")
                break
        return

    # print(D_full_fusion.indices, D_full_fusion.data)
    # print(D_partial_fusion.indices, D_partial_fusion.data)
    # print(D_nofusion.indices, D_nofusion.data)

    #print(D_nofusion.data != D_partial_fusion.data)



    return ans, full_lb, partial_lb, nofusion_lb, A_index.shape[0]+B_index.shape[0]+C_index.shape[0]
    
def sparse_vector_test():
    seed = 42517
    torch.manual_seed(seed)
    random.seed(seed)
    # benchmark_csr_add(0,100)
    #print(matrix_list())
    size = 2*1e9
    length = np.logspace(2.0, 8.49, num=1000)
    #print(length)
    full_time = []
    partial_time = []
    no_time = []
    lengths = []
    skip = []
    failed = []

    pts = 1713
    df = matrix_list()
    print(df)
    start = 1700
    for i in range(start,pts):
        if i in skip:
            continue
        
        print(f"iteration {i}")
        ans, full, partial, no , size= test_mergepath(int(size), length[0], df.iloc[i]['name'], df.iloc[i+1]['name'], df.iloc[i+2]['name'])
       
        if not ans:
            failed.append(i)

        print( ans, full, partial, no)
        if not ans:
            break
        full_time.append(full)
        partial_time.append(partial)
        no_time.append(no)
        lengths.append(size)
    #data = np.load("benchmark_results/" + "mergepath_test_5" + ".npz", allow_pickle=True)
    #print(full_time, partial_time, no_time)
    #plot_bar_graph(size, length[:pts],full_time, partial_time, no_time, "bar_graph_8")
    plot_bar_graph_2(size, lengths, full_time, partial_time, no_time, "1700-1712mergepath_ab+c_suitesparse_3")
    load_and_plot2(["mergepath_ab+c_suitesparse_3", "0-1422mergepath_ab+c_suitesparse_3", "1600-1700mergepath_ab+c_suitesparse_3", "1700-1712mergepath_ab+c_suitesparse_3"], "ab+c_combined")
    print(failed)
    #benchmark_csr_add(0,100)
    #benchmark_coo_add(0,100
    
   

