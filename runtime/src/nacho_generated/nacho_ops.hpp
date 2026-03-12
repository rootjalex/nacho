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
