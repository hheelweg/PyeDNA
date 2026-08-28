import argparse
from pyedna.analysis.workflow import run_trajectory_analysis


def main(config_file):
    
    return run_trajectory_analysis(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MD trajectory")
    parser.add_argument("config_file", nargs="?", help="Path to traj.toml")
    parser.add_argument("--config", default=None, help="Path to traj.toml")
    args = parser.parse_args()
    main(args.config or args.config_file or "traj.toml")
