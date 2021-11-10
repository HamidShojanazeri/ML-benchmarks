from backends.ort import benchmark_ORT, profile_ORT
from backends.torchscript import benchmark_Torchscript, profile_torchscript

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser("Model benchmarking")
    parser.add_argument("--model_path", type=str, help="The path to the trained/optimzied model")
    parser.add_argument("--duration", type=str, help="duration of benchmark run")
    parser.add_argument("--backend", type=str, help="Backend, torchscript, ort")
    parser.add_argument("--output_path", type=str, help="Where the resulting report will be saved")
    parser.add_argument("--profile", type=bool, help="flag to profile the model")
    parser.add_argument('--batch_sizes', nargs='+', type=int)
    parser.add_argument('--sequence_lengths', nargs='+', type=int)
    # Parse command line arguments
    args = parser.parse_args()

    if args.profile and args.backend=='ort':
        profile_ORT(args.model_path, args.batch_sizes[0],args.sequence_lengths[0], args.output_path)
    elif args.profile and args.backend=='torchscript':
        profile_torchscript(args.model_path, args.batch_sizes[0],args.sequence_lengths[0], args.output_path)
    else:
        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                if args.backend == 'ort':
                    benchmark_ORT(args.model_path, batch_size,sequence_length, args.backend, args.output_path, args.duration)
                elif args.backend == 'torchscript':
                    benchmark_Torchscript(args.model_path, batch_size,sequence_length, args.backend, args.output_path, args.duration)

