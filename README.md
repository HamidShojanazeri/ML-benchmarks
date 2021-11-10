# ML-benchmarks

To add your inference backend/engine add it in backend folder and modify the benchmark_runs.py to add your backend.


To run the benchmarks follows:

Supported backend so far:
- torchscript
- ort

```
python benchmark_runs.py --model_path traced_model.pt --backend torchscript --output_path ./benchmark-output --duration 5 --batch_sizes 1 2 --sequence_lengths 10 20

```
This will save the cvs file for each run with file_name of output_path/backend_batchsize_seq_length.csv

To profile 

```
python benchmark_runs.py --model_path traced_model.pt --backend torchscript --output_path ./benchmark-output --duration 5 --batch_sizes 1  --sequence_lengths 10 --profile 1 

```
