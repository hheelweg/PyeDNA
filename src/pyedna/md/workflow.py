from pyedna.md import MDSimulation


def run_md(config_file="md.toml"):
    md = MDSimulation.from_file(config_file)
    print(f"MD run directory: {md.output_dir}", flush=True)
    md.run()