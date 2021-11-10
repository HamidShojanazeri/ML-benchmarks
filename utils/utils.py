from typing import Generic, TypeVar, ClassVar, List, Optional, Set, Tuple
import csv 
import os

SEC_TO_MS_SCALE = 1000

def get_dummy_token(tokenizer) -> str:
    if tokenizer.unk_token is not None:
        return tokenizer.unk_token
    else:
        return tokenizer.convert_tokens_to_string([1])

def get_dummy_inputs(batch_size: int, seq_len: int, tokenizer) -> List[List[str]]:
    return [[get_dummy_token(tokenizer)] * seq_len] * batch_size

def csv_writer(bechmark_metrics, backend, batch_size,sequence_length, output_folder):
    file_name = "resutls_{}_{}_{}.csv".format(backend,batch_size,sequence_length)
    output_dir = os.path.join(output_folder, file_name)
    with open(output_dir,'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in bechmark_metrics.items():
            writer.writerow([key, value])