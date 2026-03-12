nacho_runtime
=============

```bash
module load cuda
pip install .
python -c "import nacho_runtime; print('nacho_runtime OK')"
pytest tests/
```

`pip install .` automatically generates nacho kernels during the CMake build.
Generated `.cu` files are emitted into the build directory (not the source tree).
