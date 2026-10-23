# Running the notebooks

## Inspect saved results without a GPU

Open any notebook on GitHub to see its saved figures and tables. The notebooks include CPU analysis cells with embedded historical evidence. The replay command below selects those cells using the stable cell IDs recorded in `source_manifest.json`, so changing a presentation heading does not change what runs. Use a fresh process for historical analysis: live-run variables such as `OUT` deliberately switch the analysis to current-run evidence.

For local validation and CPU replay:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/validate_notebooks.py
python scripts/validate_notebooks.py --replay
~~~

Replay writes figures, tables, and logs under `outputs/cpu-replay/`. It executes only the archived analysis cells in separate processes, without model downloads or GPU inference. The tracked notebook outputs remain unchanged.

## Run new experiments on Kaggle

Download the desired notebook and import it into Kaggle. Enable a GPU accelerator and Internet. Accept access to [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it), then add an enabled Kaggle Secret named `HF_TOKEN` with permission to read that model. Use a fresh notebook session for each experiment and run its own installation/setup cells.

- Notebook 1 uses a discovery budget and a soft session limit; `RESUME_ZIP` continues matching saved runs.
- Notebook 2 defaults to `MODE='smoke'`; larger modes use independent run settings.
- Notebook 3 defaults to `MODE='pilot'`. Set `MODE='smoke'` for a smaller check. Its feature interventions require the reconstruction review gate and `RUN_FEATURE_INTERVENTIONS=True`.
- Notebook 4 defaults to `MODE='smoke'` and `GENERATE=False`: it initially captures activations. Enable generation when ready to collect answers.
- Notebook 5 contains the 256-answer intervention run. Inspect its settings before starting; `RESUME_SOURCE` can restore a matching checkpoint.
- Notebook 6 captures all 26 decoder-layer outputs for 24 fixed prompt variants. It starts with imports, so add this setup cell first in a fresh Kaggle session:

~~~python
%pip install "transformers==4.51.3" "accelerate==1.6.0" "huggingface_hub==0.36.2" safetensors numpy pandas scipy matplotlib
~~~

Download result ZIPs before ending a Kaggle session. Preserve your completed review CSVs separately before re-exporting blank review packets. Follow the settings and review instructions inside each notebook; the six experiments do not share one global configuration.

## Provenance

The scientific figures, numeric tables, and embedded evidence are retained in the notebooks. [source_manifest.json](source_manifest.json) records source filenames, hashes, presentation edits, and the start of each CPU analysis. GPU dependencies and model/dataset revisions are recorded inside each experiment.
