#pragma once
#include "../nb_utils.hpp"

// Sparse vector operations (CVector in, CVector out)
CVector<int, int, float> nacho_sparse_vec_mul_nb(CVector<int, int, float> A,
                                                  CVector<int, int, float> B);

CVector<int, int, float> nacho_sparse_vec_add_nb(CVector<int, int, float> A,
                                                  CVector<int, int, float> B);

CVector<int, int, float> nacho_sparse_vec_apb_c_nb(CVector<int, int, float> A,
                                                     CVector<int, int, float> B,
                                                     CVector<int, int, float> C);

CVector<int, int, float> nacho_sparse_vec_ab_pc_nb(CVector<int, int, float> A,
                                                     CVector<int, int, float> B,
                                                     CVector<int, int, float> C);

// DCSR operations (DCSR in, DCSR out)
DCSR<int, float> nacho_dcsr_mul_nb(DCSR<int, float> A, DCSR<int, float> B);

DCSR<int, float> nacho_dcsr_add_nb(DCSR<int, float> A, DCSR<int, float> B);

// CSR operations (CSR in, CSR out)
CSR<int, float> nacho_csr_add_nb(CSR<int, float> A, CSR<int, float> B);

// TCSF operations (3D all-sparse, TCSF in/out)
TCSF<int, float> nacho_tcsf_add_nb(TCSF<int, float> A, TCSF<int, float> B);

// COO operations (nacho-generated Coordinate format)
NachoCOO2D<int, float> nacho_coo2_add_nb(NachoCOO2D<int, float> A,
                                          NachoCOO2D<int, float> B);

NachoCOO3D<int, float> nacho_coo3_add_nb(NachoCOO3D<int, float> A,
                                          NachoCOO3D<int, float> B);

NachoCOO4D<int, float> nacho_coo4_add_nb(NachoCOO4D<int, float> A,
                                          NachoCOO4D<int, float> B);
