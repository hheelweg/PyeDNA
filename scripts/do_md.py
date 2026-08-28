import argparse

from pyedna.md import MDSimulation


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
