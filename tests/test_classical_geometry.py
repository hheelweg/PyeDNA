from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pyedna.analysis.classical.geometry import (
    angle_between_vectors,
    axis_angle,
    axis_from_named_atoms,
    orientation_factor,
    load_dye_geometry,
    plane_angle,
    plane_deviation_from_named_atoms,
    plane_normal_from_named_atoms,
)
from pyedna.analysis.config import validate_analysis_config
from pyedna.analysis.interactions import (
    run_axis_angle_interaction,
    run_distance_interaction,
    run_orientation_factor_interaction,
    run_plane_angle_interaction,
)
from pyedna.analysis.classical.jobs import classical_observables
from pyedna.trajectory.trajectory import validate_frame_interval


class DummyMol:
    def __init__(self, coords, masses=None):
        self.coords = np.asarray(coords, dtype=float)
        self.natm = len(self.coords)
        self.charge = 0
        self._masses = masses

    def atom_coord(self, index, unit="Angstrom"):
        return self.coords[index]

    def atom_mass_list(self):
        if self._masses is None:
            return [1.0 for _ in range(self.natm)]
        return self._masses


class ClassicalGeometryTests(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tempdir.name)

    def tearDown(self):
        self._tempdir.cleanup()

    def _dye_dir(self, dye="CY3", text=None):
        directory = self.tmpdir / dye
        directory.mkdir(parents=True, exist_ok=True)
        if text is not None:
            (directory / "geometry.toml").write_text(text)
        return self.tmpdir

    def _patch_config(self, dye_dir):
        cfg = SimpleNamespace(libraries=SimpleNamespace(dye_dir=dye_dir))
        return patch("pyedna.analysis.classical.geometry.get_config", return_value=cfg)

    def test_parses_valid_axis(self):
        dye_dir = self._dye_dir(text='[axis]\natoms = ["C4", "C12"]\n')

        with self._patch_config(dye_dir):
            geometry = load_dye_geometry("CY3", require_axis=True)

        self.assertEqual(geometry.axis_atoms, ("C4", "C12"))

    def test_parses_valid_plane(self):
        dye_dir = self._dye_dir(text='[plane]\natoms = ["C1", "C2", "C3", "C4"]\n')

        with self._patch_config(dye_dir):
            geometry = load_dye_geometry("CY3")

        self.assertEqual(geometry.plane_atoms, ("C1", "C2", "C3", "C4"))

    def test_missing_geometry_for_axis_is_explicit(self):
        dye_dir = self._dye_dir()

        with self._patch_config(dye_dir), self.assertRaisesRegex(FileNotFoundError, "CY3"):
            load_dye_geometry("CY3", require_axis=True)

    def test_missing_axis_is_explicit(self):
        dye_dir = self._dye_dir(text='[plane]\natoms = ["C1", "C2", "C3"]\n')

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "\\[axis\\]"):
            load_dye_geometry("CY3", require_axis=True)

    def test_invalid_axis_atom_count_is_rejected(self):
        dye_dir = self._dye_dir(text='[axis]\natoms = ["C1"]\n')

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "exactly 2"):
            load_dye_geometry("CY3", require_axis=True)

    def test_invalid_plane_atom_count_is_rejected(self):
        dye_dir = self._dye_dir(text='[plane]\natoms = ["C1", "C2"]\n')

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "at least 3"):
            load_dye_geometry("CY3")

    def test_requested_atom_absent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            axis_from_named_atoms(["C1", "C2"], [[0, 0, 0], [1, 0, 0]], ["C1", "C3"])

    def test_duplicate_atom_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            axis_from_named_atoms(["C1", "C1"], [[0, 0, 0], [1, 0, 0]], ["C1", "C2"])


    def test_angle_between_vectors_directed_parallel(self):
        self.assertAlmostEqual(angle_between_vectors([1, 0, 0], [2, 0, 0]), 0.0)

    def test_angle_between_vectors_directed_antiparallel(self):
        self.assertAlmostEqual(angle_between_vectors([1, 0, 0], [-1, 0, 0]), 180.0)

    def test_angle_between_vectors_undirected_antiparallel(self):
        self.assertAlmostEqual(
            angle_between_vectors([1, 0, 0], [-1, 0, 0], undirected=True),
            0.0,
        )

    def test_angle_between_vectors_perpendicular(self):
        self.assertAlmostEqual(angle_between_vectors([1, 0, 0], [0, 1, 0]), 90.0)

    def test_angle_between_vectors_known_sixty_degrees(self):
        self.assertAlmostEqual(
            angle_between_vectors([1, 0, 0], [0.5, np.sqrt(3) / 2, 0]),
            60.0,
        )

    def test_axis_angle_zero_degrees(self):
        self.assertAlmostEqual(axis_angle([1, 0, 0], [2, 0, 0]), 0.0)

    def test_axis_angle_ninety_degrees(self):
        self.assertAlmostEqual(axis_angle([1, 0, 0], [0, 1, 0]), 90.0)

    def test_axis_angle_antiparallel_is_undirected_zero_degrees(self):
        self.assertAlmostEqual(axis_angle([1, 0, 0], [-1, 0, 0]), 0.0)

    def test_plane_normal_uses_all_named_atoms(self):
        normal = plane_normal_from_named_atoms(
            ["A", "B", "C", "D"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            ["A", "B", "C", "D"],
        )

        self.assertAlmostEqual(abs(float(np.dot(normal, [0, 0, 1]))), 1.0)


    def test_plane_deviation_three_atoms_is_zero(self):
        values = plane_deviation_from_named_atoms(
            ["A", "B", "C"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            ["A", "B", "C"],
        )

        self.assertAlmostEqual(values["plane_rmsd"], 0.0)

    def test_plane_deviation_known_nonplanar_geometry(self):
        values = plane_deviation_from_named_atoms(
            ["A", "B", "C", "D"],
            [[1, 0, 0.25], [-1, 0, 0.25], [0, 1, -0.25], [0, -1, -0.25]],
            ["A", "B", "C", "D"],
        )

        self.assertAlmostEqual(values["plane_rmsd"], 0.25)

    def test_plane_deviation_rejects_collinear_atoms(self):
        with self.assertRaisesRegex(ValueError, "collinear"):
            plane_deviation_from_named_atoms(
                ["A", "B", "C"],
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                ["A", "B", "C"],
            )

    def test_plane_angle_parallel_planes(self):
        self.assertAlmostEqual(plane_angle([0, 0, 1], [0, 0, 2]), 0.0)

    def test_plane_angle_opposite_normals_are_undirected(self):
        self.assertAlmostEqual(plane_angle([0, 0, 1], [0, 0, -1]), 0.0)

    def test_plane_angle_perpendicular_planes(self):
        self.assertAlmostEqual(plane_angle([0, 0, 1], [1, 0, 0]), 90.0)

    def test_plane_deviation_observable_known_geometry(self):
        dye_dir = self._dye_dir("CY3", '[plane]\natoms = ["A", "B", "C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}]}
        snapshots = {
            10: SimpleNamespace(
                dye="CY3",
                atom_names=("A", "B", "C", "D"),
                coordinates=np.array(
                    [[1, 0, 0.25], [-1, 0, 0.25], [0, 1, -0.25], [0, -1, -0.25]],
                    dtype=float,
                ),
            ),
        }

        with self._patch_config(dye_dir):
            values = classical_observables(
                DummyMol([[0, 0, 0]]),
                ["plane_deviation"],
                group_name="donor",
                config=config,
                attachment_snapshots=snapshots,
            )

        self.assertAlmostEqual(values["plane_rmsd"], 0.25)

    def test_plane_deviation_missing_plane_mentions_requested_analysis(self):
        dye_dir = self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}]}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B", "C"), coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "plane_deviation"):
            classical_observables(
                DummyMol([[0, 0, 0]]),
                ["plane_deviation"],
                group_name="donor",
                config=config,
                attachment_snapshots=snapshots,
            )

    def test_plane_deviation_invalid_atom_names_are_rejected(self):
        dye_dir = self._dye_dir("CY3", '[plane]\natoms = ["A", "B", "C"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}]}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B", "X"), coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "found 0"):
            classical_observables(
                DummyMol([[0, 0, 0]]),
                ["plane_deviation"],
                group_name="donor",
                config=config,
                attachment_snapshots=snapshots,
            )

    def test_plane_deviation_rejects_multi_attachment_group(self):
        config = {"groups": [{"name": "donor", "attachments": [10, 11]}]}

        with self.assertRaisesRegex(ValueError, "exactly one attachment"):
            classical_observables(
                DummyMol([[0, 0, 0]]),
                ["plane_deviation"],
                group_name="donor",
                config=config,
                attachment_snapshots={},
            )

    def test_plane_angle_interaction_known_geometry(self):
        self._dye_dir("CY3", '[plane]\natoms = ["A", "B", "C"]\n')
        dye_dir = self._dye_dir("CY5", '[plane]\natoms = ["D", "E", "F"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B", "C"), coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("D", "E", "F"), coordinates=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)),
        }

        with self._patch_config(dye_dir):
            result = run_plane_angle_interaction(
                {"type": "plane_angle", "groups": ["donor", "acceptor"], "method": "plane"},
                config,
                snapshots,
                frame=5,
            )

        self.assertEqual(result.method, "plane")
        self.assertAlmostEqual(result.values["plane_angle"], 90.0)

    def test_plane_angle_rejects_multi_attachment_group(self):
        config = {"groups": [{"name": "donor", "attachments": [10, 11]}, {"name": "acceptor", "attachments": [12]}]}
        interaction = {"type": "plane_angle", "groups": ["donor", "acceptor"]}

        with self.assertRaisesRegex(ValueError, "exactly one attachment"):
            run_plane_angle_interaction(interaction, config, {}, frame=0)

    def test_axis_angle_interaction_rejects_multi_attachment_group(self):
        config = {"groups": [{"name": "donor", "attachments": [10, 11]}, {"name": "acceptor", "attachments": [12]}]}
        interaction = {"type": "axis_angle", "groups": ["donor", "acceptor"]}

        with self.assertRaisesRegex(ValueError, "exactly one attachment"):
            run_axis_angle_interaction(interaction, config, {}, frame=0)

    def test_axis_angle_interaction_known_geometry(self):
        self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [0, 1, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir):
            result = run_axis_angle_interaction(
                {"type": "axis_angle", "groups": ["donor", "acceptor"], "method": "axis"},
                config,
                snapshots,
                frame=5,
            )

        self.assertEqual(result.method, "axis")
        self.assertAlmostEqual(result.values["axis_angle"], 90.0)


    def test_orientation_factor_parallel_axes_perpendicular_to_r(self):
        values = orientation_factor([0, 1, 0], [0, 2, 0], [1, 0, 0])

        self.assertAlmostEqual(values["kappa"], 1.0)
        self.assertAlmostEqual(values["kappa_squared"], 1.0)

    def test_orientation_factor_parallel_axes_aligned_with_r(self):
        values = orientation_factor([1, 0, 0], [2, 0, 0], [1, 0, 0])

        self.assertAlmostEqual(values["kappa"], -2.0)
        self.assertAlmostEqual(values["kappa_squared"], 4.0)

    def test_orientation_factor_perpendicular_axes_zero(self):
        values = orientation_factor([0, 1, 0], [0, 0, 1], [1, 0, 0])

        self.assertAlmostEqual(values["kappa"], 0.0)
        self.assertAlmostEqual(values["kappa_squared"], 0.0)

    def test_orientation_factor_axis_reversal_behavior(self):
        forward = orientation_factor([0, 1, 0], [0, 1, 0], [1, 0, 0])
        one_reversed = orientation_factor([0, 1, 0], [0, -1, 0], [1, 0, 0])
        both_reversed = orientation_factor([0, -1, 0], [0, -1, 0], [1, 0, 0])

        self.assertAlmostEqual(one_reversed["kappa"], -forward["kappa"])
        self.assertAlmostEqual(one_reversed["kappa_squared"], forward["kappa_squared"])
        self.assertAlmostEqual(both_reversed["kappa"], forward["kappa"])
        self.assertAlmostEqual(both_reversed["kappa_squared"], forward["kappa_squared"])

    def test_orientation_factor_interaction_center_of_geometry(self):
        self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[1, 0, 0]])}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [0, 1, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [0, 1, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir):
            result = run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"], "method": "center_of_geometry"},
                config,
                groups,
                snapshots,
                frame=5,
            )

        self.assertEqual(result.method, "center_of_geometry")
        self.assertAlmostEqual(result.values["kappa"], 1.0)
        self.assertAlmostEqual(result.values["kappa_squared"], 1.0)

    def test_orientation_factor_interaction_center_of_mass(self):
        self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {
            "donor": DummyMol([[0, 0, 0], [10, 0, 0]], masses=[9, 1]),
            "acceptor": DummyMol([[0, 1, 0], [10, 1, 0]], masses=[9, 1]),
        }
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir):
            result = run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"], "method": "center_of_mass"},
                config,
                groups,
                snapshots,
                frame=5,
            )

        self.assertEqual(result.method, "center_of_mass")
        self.assertAlmostEqual(result.values["kappa_squared"], 1.0)

    def test_orientation_factor_missing_geometry_mentions_requested_analysis(self):
        dye_dir = self._dye_dir("CY3")
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[1, 0, 0]])}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(FileNotFoundError, "orientation_factor"):
            run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"]},
                config,
                groups,
                snapshots,
                frame=0,
            )

    def test_orientation_factor_missing_axis_mentions_requested_analysis(self):
        self._dye_dir("CY3", '[plane]\natoms = ["A", "B", "C"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[1, 0, 0]])}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "orientation_factor"):
            run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"]},
                config,
                groups,
                snapshots,
                frame=0,
            )

    def test_orientation_factor_invalid_axis_atom_names_are_rejected(self):
        self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[1, 0, 0]])}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "X"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "found 0"):
            run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"]},
                config,
                groups,
                snapshots,
                frame=0,
            )

    def test_orientation_factor_rejects_coincident_centers(self):
        self._dye_dir("CY3", '[axis]\natoms = ["A", "B"]\n')
        dye_dir = self._dye_dir("CY5", '[axis]\natoms = ["C", "D"]\n')
        config = {"groups": [{"name": "donor", "attachments": [10]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[0, 0, 0]])}
        snapshots = {
            10: SimpleNamespace(dye="CY3", atom_names=("A", "B"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
            12: SimpleNamespace(dye="CY5", atom_names=("C", "D"), coordinates=np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        }

        with self._patch_config(dye_dir), self.assertRaisesRegex(ValueError, "coincide"):
            run_orientation_factor_interaction(
                {"type": "orientation_factor", "groups": ["donor", "acceptor"]},
                config,
                groups,
                snapshots,
                frame=0,
            )

    def test_orientation_factor_rejects_multi_attachment_group(self):
        config = {"groups": [{"name": "donor", "attachments": [10, 11]}, {"name": "acceptor", "attachments": [12]}]}
        groups = {"donor": DummyMol([[0, 0, 0]]), "acceptor": DummyMol([[1, 0, 0]])}
        interaction = {"type": "orientation_factor", "groups": ["donor", "acceptor"]}

        with self.assertRaisesRegex(ValueError, "exactly one attachment"):
            run_orientation_factor_interaction(interaction, config, groups, {}, frame=0)

    def test_distance_interaction_still_works(self):
        groups = {
            "donor": DummyMol([[0, 0, 0]]),
            "acceptor": DummyMol([[3, 4, 0]]),
        }
        result = run_distance_interaction(
            {"type": "distance", "groups": ["donor", "acceptor"], "method": "center_of_geometry"},
            groups,
            frame=0,
        )

        self.assertAlmostEqual(result.values["distance"], 5.0)

    def test_existing_classical_observables_still_work(self):
        mol = DummyMol([[0, 0, 0], [2, 0, 0]], masses=[1.0, 3.0])
        values = classical_observables(
            mol,
            ["center_of_geometry", "center_of_mass", "radius_of_gyration"],
        )

        np.testing.assert_allclose(values["center_of_geometry"], [1, 0, 0])
        np.testing.assert_allclose(values["center_of_mass"], [1.5, 0, 0])
        self.assertAlmostEqual(values["radius_of_gyration"], 1.0)

    def test_config_accepts_axis_angle_interaction(self):
        config = _minimal_analysis_config()
        config["classical_interactions"].append({"type": "axis_angle", "groups": ["donor", "acceptor"]})

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["classical_interactions"][1]["method"], "axis")

    def test_config_accepts_plane_deviation_output(self):
        config = _minimal_analysis_config()
        config["classical"] = [{"group": "donor", "outputs": ["plane_deviation"]}]

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["classical"][0]["outputs"], ["plane_deviation"])

    def test_config_accepts_plane_angle_interaction(self):
        config = _minimal_analysis_config()
        config["classical_interactions"].append({"type": "plane_angle", "groups": ["donor", "acceptor"]})

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["classical_interactions"][1]["method"], "plane")

    def test_config_accepts_orientation_factor_interaction(self):
        config = _minimal_analysis_config()
        config["classical_interactions"].append({"type": "orientation_factor", "groups": ["donor", "acceptor"]})

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["classical_interactions"][1]["method"], "center_of_geometry")

    def test_config_accepts_orientation_factor_center_of_mass(self):
        config = _minimal_analysis_config()
        config["classical_interactions"].append({"type": "orientation_factor", "groups": ["donor", "acceptor"], "method": "center_of_mass"})

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["classical_interactions"][1]["method"], "center_of_mass")

    def test_config_rejects_axis_angle_classical_output(self):
        config = _minimal_analysis_config()
        config["classical"] = [{"group": "donor", "outputs": ["axis_angle"]}]

        with self.assertRaisesRegex(ValueError, "unsupported outputs"):
            validate_analysis_config(config)

    def test_config_accepts_all_frame_interval_and_default_stride(self):
        config = _minimal_analysis_config()
        config["trajectory"]["frame_interval"] = "all"

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["trajectory"]["frame_interval"], "all")
        self.assertEqual(validated["trajectory"]["frame_stride"], 1)

    def test_config_accepts_frame_stride(self):
        config = _minimal_analysis_config()
        config["trajectory"]["frame_stride"] = 10

        validated = validate_analysis_config(config).data

        self.assertEqual(validated["trajectory"]["frame_stride"], 10)

    def test_config_rejects_invalid_frame_stride(self):
        config = _minimal_analysis_config()
        config["trajectory"]["frame_stride"] = 0

        with self.assertRaisesRegex(TypeError, "frame_stride"):
            validate_analysis_config(config)

    def test_all_frame_interval_resolves_to_available_frames(self):
        self.assertEqual(validate_frame_interval("all", 41), (0, 40))

    def test_frame_interval_rejects_unknown_string(self):
        with self.assertRaisesRegex(ValueError, "all"):
            validate_frame_interval("everything", 41)


def _minimal_analysis_config():
    return {
        "trajectory": {
            "run_directory": "md/run",
            "topology_file": "system.prmtop",
            "trajectory_file": "system.nc",
            "frame_interval": [0, 0],
        },
        "attachments": [
            {"dye": "CY3", "residue": 10, "cap": "H"},
            {"dye": "CY5", "residue": 12, "cap": "H"},
        ],
        "groups": [
            {"name": "donor", "attachments": [10]},
            {"name": "acceptor", "attachments": [12]},
        ],
        "classical_interactions": [
            {"type": "distance", "groups": ["donor", "acceptor"], "method": "center_of_geometry"},
        ],
    }


if __name__ == "__main__":
    unittest.main()
