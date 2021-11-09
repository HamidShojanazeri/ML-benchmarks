from pathlib import Path
from os import environ
from psutil import cpu_count
from transformers import BertTokenizerFast, DistilBertTokenizer
import numpy as np
from time import perf_counter
import torch
from backends.ort import benchmark_ORT
from transformers import AutoModel, TensorType
from utils.utils import get_dummy_inputs, get_dummy_inputs, csv_writer
import csv
def benchmark_Torchscript(model_path, batch_size,sequence_length, backend, output_folder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model_inputs = tokenizer("My name is Bert", return_tensors="pt")
    model = torch.jit.load(model_path, map_location=device)
    dummy_inputs = get_dummy_inputs(
            batch_size=batch_size,
            seq_len=(sequence_length - tokenizer.num_special_tokens_to_add(pair=False)),tokenizer=tokenizer
        )

    inputs = tokenizer(
        dummy_inputs,
        is_split_into_words=True,
        return_tensors=TensorType.PYTORCH,
    )
    model_inputs = inputs
    latencies = []
    # Warmup
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device) 
    for _ in range(10):
        _ = model(input_ids,attention_mask)
        
    for _ in range(100):
        start_time = perf_counter()
        _ = model(input_ids,attention_mask)
        latency = (perf_counter() - start_time )*1000
        latencies.append(latency)
    bechmark_metrics={
        
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_50": np.quantile(latencies, 0.5),
            "latency_90": np.quantile(latencies, 0.9),
            "latency_95": np.quantile(latencies, 0.95),
            "latency_99": np.quantile(latencies, 0.99),
            "latency_999": np.quantile(latencies, 0.999),
    }
    csv_writer(bechmark_metrics, backend, batch_size,sequence_length, output_folder)
    
