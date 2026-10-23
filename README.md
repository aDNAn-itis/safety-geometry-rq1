# Safety Geometry — Research Question 1

**How does persona framing change a language model's responses and measured internal representations?**

Six exploratory notebooks investigate this question using Gemma 3 1B IT, Gemma Scope 2 sparse autoencoders (SAEs), feature interventions, and residual-stream trajectories. They examine whether candidate activity persists when behavior changes and what the measurements can establish about refusal. The broader motivation is the robustness of safety alignment across framing.

## Notebook sequence

| # | Notebook | Question |
|---|---|---|
| 1 | [Feature coverage](notebooks/01_feature_coverage.ipynb) | Does the observed SAE feature inventory stabilize as more prompts are sampled? |
| 2 | [Candidate discovery](notebooks/02_candidate_discovery.ipynb) | Can a compact feature group distinguish harmful from benign request labels? |
| 3 | [Reconstruction and removal](notebooks/03_reconstruction_and_removal.ipynb) | Does SAE reconstruction preserve reviewed behavior, and does removing eight selected features change refusal? |
| 4 | [Persona activation mapping](notebooks/04_persona_activation_mapping.ipynb) | How do responses and candidate activity change under different framings of the same request? |
| 5 | [Five-feature interventions](notebooks/05_five_feature_interventions.ipynb) | Does restoring or removing a selected contribution change behavior relative to sham and random controls? |
| 6 | [Layer trajectories](notebooks/06_layer_trajectories.ipynb) | Do selected persona switches exhibit a distinctive late-layer trajectory change? |

Read in this order. The 20-feature candidate group, notebook 3's eight-feature group, and notebook 5's five-feature group have different selection histories; their IDs and conclusions are not interchangeable.

## Inspect saved results without a GPU

Open any notebook on GitHub to see its saved figures and tables. Each notebook also has a **Results** section with embedded historical evidence. In a fresh Python kernel, run from that heading onward to recompute those analyses on CPU. Skip the earlier GPU cells when inspecting the archive. Live-run variables such as `OUT` deliberately switch the analysis to current-run evidence.

For local validation and CPU replay:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python scripts/validate_notebooks.py
python scripts/validate_notebooks.py --replay
~~~

Replay writes figures, tables, and logs under `outputs/cpu-replay/`. It executes only the Results sections in separate processes, without model downloads or GPU inference. The tracked notebook outputs remain unchanged.

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

## Evidence and provenance

Request harmfulness, feature activity, and reviewed response behavior are separate measurements. Historical switch classifications remain provisional, and the saved five-feature intervention evidence still needs behavioral adjudication. These pilots do not establish invariant ethical principles, a complete safety map, or a downstream bypass mechanism.

The supplied notebooks' scientific figures, numeric tables, and embedded evidence are retained. Two prose cells were corrected from code to Markdown, and presentation copy was cleaned up to remove proposal-writing notes and external audit summaries. [source_manifest.json](source_manifest.json) records the original filenames, hashes, and edits. The validation workflow checks notebook structure and recomputes the archived CPU analyses; it does not rerun the GPU experiments.

Model and dictionary references: [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it), [Gemma Scope 2 1B IT](https://huggingface.co/google/gemma-scope-2-1b-it). Notebook 2 uses [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) or a user-supplied CSV; its recorded configuration identifies the data source and revision. Model and dataset terms remain applicable to those resources.
