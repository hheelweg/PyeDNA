# Analysis Migration Audit

This audit records the old trajectory-analysis workflow pieces that were removed after the new `pyedna.analysis` / `pyedna.trajectory` workflow became the active path.

## Removed Old Workflow Files

The following files were deleted:

- `scripts/analyze_traj_old.py`
- `scripts/vibrational.py`
- `src/pyedna/geomtools.py`
- `src/pyedna/quanttools.py`
- `src/pyedna/pyscf_utils.py`
- `src/pyedna/qm_driver.py`
- `src/pyedna/trajectory.py`
- `src/pyedna/structure/legacy.py`

## Migrated Functionality

| Old functionality | New location |
| --- | --- |
| trajectory frame loading and snapshot extraction | `pyedna.trajectory.snapshot` |
| attachment molecule construction, caps, and groups | `pyedna.trajectory.structure` |
| trajectory-analysis orchestration | `pyedna.analysis.workflow` |
| TOML validation | `pyedna.analysis.config` |
| JSONL output, manifest, loaders, dataframe helpers | `pyedna.analysis.io` |
| PySCF DFT/TDDFT backend | `pyedna.analysis.quantum.pyscf` |
| Mulliken population helper | `pyedna.analysis.quantum.pyscf.mulliken_pop` |
| TDM couplings | `pyedna.analysis.quantum.couplings` |
| quantum job scheduling and GPU worker assignment | `pyedna.analysis.quantum.jobs` |
| classical distances and geometry summaries | `pyedna.analysis.classical` |

## Intentionally Dropped Functionality

These old workflow features were not carried forward:

- `traj.params` parsing
- old pandas/MultiIndex output formatting
- joblib cache exchange through `qm_driver.py`
- old ORCA/vibrational test script path
- spectrum helper
- orbital-energy window helper
- symmetry/alignment helpers from `geomtools.py`
- force-field/symmetry optimization helpers in `quanttools.py`

## Current Status

The current trajectory-analysis workflow is:

```bash
python "$PYEDNA_HOME/scripts/analyze_traj.py" traj.toml
```

The TOML-driven analysis path is the only workflow advertised in the main user documentation.
