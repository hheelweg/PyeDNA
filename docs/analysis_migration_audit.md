# Analysis Migration Audit

This audit records the OLD trajectory-analysis workflow pieces that were
removed after the new `pyedna.analysis` / `pyedna.trajectory` workflow became
the active path.

## Removed OLD Workflow Files

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

| OLD functionality | New location |
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

These OLD workflow features were not carried forward:

- `traj.params` parsing
- old pandas/MultiIndex output formatting
- joblib cache exchange through `qm_driver.py`
- old ORCA/vibrational test script path
- spectrum helper
- orbital-energy window helper
- symmetry/alignment helpers from `geomtools.py`
- force-field/symmetry optimization helpers in `quanttools.py`

## Current Status

The NEW trajectory-analysis workflow should now be the only supported analysis
entry point:

```bash
python -m analyze_traj traj.toml
```

The remaining cleanup work is documentation and example refresh: remove stale
references to the deleted OLD workflow and make the TOML-driven analysis path
the only advertised trajectory-analysis workflow.
