import argparse
from pyedna.trajectory import analyze_trajectory, validate_frame_interval, build_groups


def main(config_file):
    traj, cfg = analyze_trajectory(config_file)
    traj_cfg = cfg["trajectory"]
    start, stop = validate_frame_interval(traj_cfg["frame_interval"], traj.num_frames)

    attachments = cfg.get("attachments", [])
    optimize_caps = traj_cfg.get("optimize_caps", False)
    basis = traj_cfg.get("basis", "6-31g")

    for frame in range(start, stop + 1):
        attachment_mols = {}

        for attachment in attachments:
            residue = attachment["residue"]
            attachment_mols[residue] = traj.get_capped_snapshot(
                frame=frame,
                initial_residue=residue,
                dye=attachment["dye"],
                cap_type=attachment.get("cap", "H"),
                optimize_caps=optimize_caps,
                basis=basis
            )

        groups = build_groups(cfg, attachment_mols, basis=basis)

        for name, mol in groups.items():
            print(
                f"Frame {frame}: group {name}, "
                f"{mol.natm} atoms, charge {mol.charge}"
            )

            # Eventually:
            # result = run_tddft(mol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MD trajectory")
    parser.add_argument("--config", default="traj.toml")
    args = parser.parse_args()
    main(args.config)