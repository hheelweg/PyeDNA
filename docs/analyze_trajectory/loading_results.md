# Loading Analysis Results

PyeDNA writes analysis outputs as JSON Lines (`.jsonl`): one JSON object per line. The easiest way to load a completed run is with `load_analysis_run`.

```python
from pyedna.analysis.io import load_analysis_run

run = load_analysis_run("analysis/example_analysis")

classical_df = run.classical_dataframe()
quantum_df = run.quantum_dataframe()
classical_interactions_df = run.classical_interactions_dataframe()
quantum_interactions_df = run.quantum_interactions_dataframe()
```

The dataframe helpers flatten nested dictionaries into dotted column names. For example, `values.distance` or `tddft.excited_state_energies.0` may appear as columns. Non-scalar nested values, such as matrices, remain Python lists in dataframe cells.

For multi-structure analysis runs, `load_analysis_run` reads every `structure_<NNN>/` output listed in `manifest.json` and combines the records. Those records include `trajectory_index`, `structure`, and `structure_directory`, which can be used to filter or group the dataframe.

You can also load an individual JSONL file directly with pandas:

```python
import pandas as pd

classical_df = pd.read_json(
    "analysis/example_analysis/classical.jsonl",
    lines=True,
)
```

For multi-structure runs, include the structure directory:

```python
classical_df = pd.read_json(
    "analysis/example_analysis/structure_001/classical.jsonl",
    lines=True,
)
```

Expected row counts are usually:

```text
classical rows              = structures x frames x classical jobs
quantum rows                = structures x frames x quantum jobs
classical interaction rows  = structures x frames x classical interaction jobs
quantum interaction rows    = structures x frames x quantum interaction jobs x state pairs
```

For `frame_interval = [0, 10]`, `frames = 11` because both endpoints are included.
