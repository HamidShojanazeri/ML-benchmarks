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
import lightseq.inference as lsi
import numpy as np
from time import perf_counter
import torch
from transformers import TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer, SEC_TO_MS_SCALE
import csv 

def benchmark_LightSeq(model_path, batch_size,sequence_length, backend, output_folder, duration):
    model = lsi.Bert(model_path, 128)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # dummy_inputs = get_dummy_inputs(
    #         batch_size=batch_size,
    #         seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
    #     )
    dummy_inputs = [('UNK ' * (sequence_length - tokenizer.num_special_tokens_to_add(pair=False))).strip()] * batch_size
    inputs = tokenizer(dummy_inputs, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    latencies = []
    # Warmup
    for _ in range(10):
        _ = model.infer(inputs_id, attn_mask)

    duration = (int(duration) * SEC_TO_MS_SCALE)
    
    while sum(latencies) < duration:
        start_time = perf_counter()
        _ = model.infer(inputs_id, attn_mask)
        latency = (perf_counter() - start_time)*SEC_TO_MS_SCALE
        latencies.append(latency)
        # Compute run statistics
    bechmark_metrics={
        "batchsize":batch_size,
        "sequence_length": sequence_length,
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "throughput":round(((len(latencies)/duration)*batch_size)*SEC_TO_MS_SCALE,2),
        "latency_50": np.quantile(latencies, 0.5),
        "latency_90": np.quantile(latencies, 0.9),
        "latency_95": np.quantile(latencies, 0.95),
        "latency_99": np.quantile(latencies, 0.99),
        "latency_999": np.quantile(latencies, 0.999),
    }
    return bechmark_metrics
    

def profile_LightSeq(model_path, batch_size,sequence_length, output_folder):
    model = lsi.Bert(model_path, 128) 

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # dummy_inputs = get_dummy_inputs(
    #         batch_size=batch_size,
    #         seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
    #     )
    dummy_inputs = [('UNK ' * (sequence_length - tokenizer.num_special_tokens_to_add(pair=False))).strip()] * batch_size
    inputs = tokenizer(dummy_inputs, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
    wait=2,
    warmup=2,
    active=6,
    repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/{}'.format(output_folder)),
    with_stack=True) as profiler:
        for i in range(1000):
            _ = model.infer(inputs_id, attn_mask)
            profiler.step()

