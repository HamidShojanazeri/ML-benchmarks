from backends.ort import benchmark_ORT
from backends.torchscript import benchmark_Torchscript

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser("Model benchmarking")
    parser.add_argument("--model_path", type=str, help="The path to the trained/optimzied model")
    parser.add_argument("--backend", type=str, help="Backend, torchscript, ort")
    parser.add_argument("--output_path", type=str, help="Where the resulting report will be saved")
    
    # Parse command line arguments
    args = parser.parse_args()
    model_path = args.model_path
    backend = args.backend
    output_folder = args.output_path
    batch_size = 1
    sequence_length = 10
    batch_sizes = [1,2]
    sequence_lengths = [8,16]
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            if args.backend == 'ort':
                benchmark_ORT(model_path, batch_size,sequence_length, backend, output_folder)
            elif args.backend == 'torchscript':
                benchmark_Torchscript(model_path, batch_size,sequence_length, backend, output_folder)

