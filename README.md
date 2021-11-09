# ML-benchmarks

To add your inference backend/engine add it in backend folder

To run the benchmarks follows:

Supported backend so far:
*torchscript
*ort

```
python benchmark_runs.py --model_path traced_model.pt --backend torchscript --output_path ./benchmark-output

```
