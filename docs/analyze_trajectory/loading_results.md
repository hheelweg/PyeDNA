# Loading Analysis Results

PyeDNA writes requested trajectory-analysis outputs as JSON Lines (`.jsonl`). In the current analysis format, each line is a compact JSON array for one analyzed frame, and the ordered column schema for that array is stored once in `manifest.json`. Result families that are not requested, such as quantum outputs in a classical-only run, do not create empty JSONL files. Older record-oriented runs are still loaded for compatibility.

The easiest way to load a completed run is with `load_analysis_run`.

```python
from pyedna.analysis.io import load_analysis_run

run = load_analysis_run("analysis/example_analysis")

classical_df = run.classical_dataframe()
quantum_df = run.quantum_dataframe()
classical_interactions_df = run.classical_interactions_dataframe()
quantum_interactions_df = run.quantum_interactions_dataframe()
```

For schema-based runs, dataframe columns come from `manifest.json`. Scalar cells become ordinary dataframe values. Array-valued or object-valued cells, such as `donor.center_of_geometry`, `donor.transition_dipoles`, or structured fragment summaries, remain Python lists or dictionaries in dataframe cells. Missing values are loaded as `None`. Pass `flatten=True` to a dataframe helper when you want fixed scalar lists expanded into numbered columns such as `donor.center_of_geometry.0`.

For fixed-shape array-valued columns, use `run.array(family, column)` to obtain a NumPy array:

```python
centers = run.array("classical", "donor.center_of_geometry")
dipoles = run.array("quantum", "donor.transition_dipoles")
```

If all non-missing cells have compatible shapes, PyeDNA stacks them into arrays such as `(n_frames, 3)` for a center vector or `(n_frames, nstates, 3)` for transition dipoles. If the shapes vary between frames, PyeDNA raises a clear error instead of silently creating an object array. Use the dataframe helpers when you want object-style access to variable-length cells.

For multi-structure analysis runs, `load_analysis_run` reads every `structure_<NNN>/` output listed in `manifest.json` and combines the rows. The JSONL rows themselves do not repeat `trajectory_index`, `structure`, or `structure_directory`; the loader adds those columns in memory from the manifest so you can filter or group the dataframe. If a result family is not listed because it was not requested, the corresponding loaded list and dataframe are empty.

Direct `pandas.read_json(..., lines=True)` can still read generated raw files, but it will only see unnamed row arrays. Prefer `load_analysis_run` when working with current PyeDNA output because it applies the manifest schema, restores column names, and handles omitted unrequested families.

For files that are generated, expected row counts are usually:

```text
classical rows              = structures x frames
quantum rows                = structures x frames
classical interaction rows  = structures x frames
quantum interaction rows    = structures x frames
```

Each row contains one value per schema column. For example, `classical_interactions.jsonl` may have one row per frame with columns such as:

```text
frame
donor.acceptor.distance
donor.acceptor.axis_angle
donor.acceptor.kappa
donor.acceptor.kappa_squared
```

Older analysis directories without `analysis_format_version = 2` in `manifest.json` are treated as legacy record-oriented output. For those runs, the dataframe helpers retain the previous recursive dictionary flattening behavior, so columns such as `values.distance` or `tddft.excited_state_energies.0` may appear.

For `frame_interval = [0, 10]` and the default `frame_stride = 1`, `frames = 11` because both endpoints are included. For `frame_interval = "all"`, PyeDNA uses every available frame in the trajectory, from `0` through `num_frames - 1`; `frame_stride = 10` keeps every tenth frame from that selection.
