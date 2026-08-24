import argparse
import torch

from pyedna.md import MDSimulation

# detect available GPUs 
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    raise RuntimeError("Error: Less than 1 GPU(s) detected! Check SLURM \
                       allocation and adjust accordingly.")


def main(config_file):
    md = MDSimulation.from_file(config_file)
    print(f"MD run directory: {md.output_dir}", flush=True)
    md.run()
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run Amber molecular dynamics")
    parser.add_argument(
        "config",
        nargs="?",
        default="md.toml",
        help="MD TOML file (default: md.toml)",
    )
    args = parser.parse_args()

    main(args.config)
