# ML-benchmarks

To add your inference backend/engine add it in backend folder and modify the benchmark_runs.py to add your backend.


To run the benchmarks follows:

Supported backend so far:
- torchscript
- ort

```
python benchmark_runs.py --model_path traced_model.pt --backend torchscript --output_path ./benchmark-output --duration 5

```
This will save the cvs file for each run with file_name of output_path/backend_batchsize_seq_length.csv
