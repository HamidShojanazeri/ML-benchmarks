from pathlib import Path
from transformers.convert_graph_to_onnx import convert
from onnxruntime_tools import optimizer
from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
from transformers import BertTokenizerFast
import onnxruntime

from os import environ
from psutil import cpu_count
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange
from transformers import BertTokenizerFast, DistilBertTokenizer
from onnxruntime.transformers import optimizer
import onnx
import numpy as np
from time import perf_counter
import torch
from transformers import TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer, SEC_TO_MS_SCALE
import csv 

def benchmark_ORT(model_path, batch_size,sequence_length, backend, output_folder, duration):
    model = onnxruntime.InferenceSession(model_path)   

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.NUMPY,
    )
    inputs = {k: v.astype("i8") for k, v in inputs.items()}
    # inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    latencies = []
    # Warmup
    for _ in range(10):
        _ = model.run(None, inputs)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    while sum(latencies) < duration:
        start_time = perf_counter()
        _ = model.run(None, inputs)
        latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
        latencies.append(latency)
        # Compute run statistics
    bechmark_metrics={
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round((len(latencies)/duration)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    csv_writer(bechmark_metrics, backend, batch_size,sequence_length, output_folder)
   
        