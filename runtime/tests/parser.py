import nacho_runtime
import pandas as pd
import torch
import os

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

def matrix_list():
    df = pd.read_csv(os.path.join(_TESTS_DIR, "suitesparse_stats.csv"))

    df.columns = ["name", "nnz", "percent_nnz", "total_elements", "rows", "columns"]

    # Convert numeric columns to integers/floats just in case they're strings
    df["nnz"] = df["nnz"].astype(int)
    df["total_elements"] = df["total_elements"].astype(int)
    df["rows"] = df["rows"].astype(int)
    df["columns"] = df["columns"].astype(int)
    df["percent_nnz"] = df["percent_nnz"].astype(float)

    matrix_dir = "/scratch/atharva/suitesparse/"

    df = df[df["name"].apply(lambda f: os.path.exists(os.path.join(matrix_dir, f)))]



    df_sorted = df.sort_values(by=["nnz"], ascending=[True])
    return df_sorted

def parse_matrix(matrix, return_coo = False):
    file = "/scratch/atharva/suitesparse/"

    matrix = file+matrix
    COO = nacho_runtime.parse2D(matrix)

    row = COO.row.to(dtype=torch.long, device="cuda")
    col = COO.col.to(dtype=torch.long, device="cuda")
    indices = torch.stack([row, col], dim=0)
    values = COO.data.to(dtype=torch.float32, device="cuda")
    COO = torch.sparse_coo_tensor(indices,values,(COO.N, COO.M))
    if return_coo:
        return COO
    CSR = COO.to_sparse_csr()

    return CSR

def parse_vector(matrix):
    file = "/scratch/atharva/suitesparse/"

    matrix = file+matrix
    COO = nacho_runtime.parse2D(matrix)

    row = COO.row.to(dtype=torch.long, device="cuda")
    col = COO.col.to(dtype=torch.long, device="cuda")
    indices = col*COO.N+row
    indices,_ = torch.sort(indices)
    values = COO.data.to(dtype=torch.float32, device="cuda")
    return indices, values, COO.N*COO.M