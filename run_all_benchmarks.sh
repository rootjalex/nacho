numactl --physcpubind 0-15 python benchmarks/csr_add.py  --device both --start 0 --end 1600  # vs cuSPARSE/Taco/MKL
numactl --physcpubind 0-15 python benchmarks/csr_mul.py     --device both --start 0 --end 1600  # vs PyTorch
numactl --physcpubind 0-15 python benchmarks/coo_add.py     --device cuda --start 0 --end 1600  # vs PyTorch
numactl --physcpubind 0-15 python benchmarks/coo_mul.py     --device cuda --start 0 --end 1600  # vs PyTorch
numactl --physcpubind 0-15 python benchmarks/coo_csr_add.py --device both --start 0 --end 1600  # vs PyTorch
numactl --physcpubind 0-15 python benchmarks/csr_add_3.py   --device cuda --start 0 --end 1600  # fused vs unfused
numactl --physcpubind 0-15 python benchmarks/heatmap_dcsr_mul.py --device both                # compare partitioning schemes
numactl --physcpubind 0-15 python benchmarks/heatmap_csr_add.py --device both       # skew sweep, synthetic

numactl --physcpubind 0-15 python benchmarks/spgemm.py --start 0 --end 1300                    # vs cuSPARSE
numactl --physcpubind 0-15 python benchmarks/sssmm.py  --start 0 --end 1300                    # fused vs unfused vs cuSPARSE
numactl --physcpubind 0-15 python benchmarks/frostt_tensors_add.py --device both               # CSF3 vs COO3D vs torch
numactl --physcpubind 0-15 python benchmarks/inner_prod.py  --device both                       # FROSTT, CSF3 vs COO3 vs torch
