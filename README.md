# Safety Geometry Under Persona Framing

## First experimental question

**In open-weight language models, how does persona framing change the geometry and functional role of candidate safety-related representations?**

When the same harmful request is placed inside a persona framing, how does its internal representation change through the model, and do the measured representations continue to influence its answer?

I distinguish two forms of robustness:

| Research target | What it asks | How it is investigated |
|---|---|---|
| **Representational robustness** | Does the measured representation remain similar across contexts? | Compare raw activations, SAE feature activity, and layer-to-layer updates under matched framings. |
| **Functional robustness** | Does a candidate representation continue to influence safety behavior across contexts? | Intervene on its contribution and compare reviewed answers with baseline, sham, and random controls. |

One may persist while the other changes. The project tests their relationship without assuming in advance that alignment is geometrically invariant or merely persona mimicry.

## Approach

The first step is to identify measurable candidate safety-related representations. I use Sparse Autoencoders (SAEs) to examine feature coordinates in the residual stream, then connect those measurements to responses and controlled interventions.

The experiments in this repository use **Gemma 3 1B IT**. Gemma Scope 2 SAEs provide **16,384 feature coordinates** at layers **7, 13, and 22**. Layer 13 is the primary midpoint measurement site. Activations are captured at the **final formatted prompt token, before response generation**. The later trajectory experiment measures all 26 decoder-layer outputs at that same token position.

The design progresses from direct benign and harmful requests to matched persona comparisons. Persona experiments preserve the underlying request across framings and include benign requests under the same wrappers. Harmful-request labels and observed refusal are recorded separately: a harmful prompt is not assumed to produce refusal.

## Experimental progression

### 1. Identify a candidate representation

The coverage pilot first asks whether the observed SAE feature inventory stabilizes as more prompts are sampled. A separate discovery experiment then selects up to 20 features per layer using the largest positive harmful-minus-benign firing-frequency differences, requiring at least three harmful discovery activations.

The discovery experiment uses **3,000 direct WildJailbreak requests**: 1,000 harmful and 1,000 benign for selection, followed by 500 of each for evaluation. Feature IDs are frozen before evaluation. The score is the fraction of selected features that activate.

| Layer | Selected features | Evaluation requests | Request-label AUROC |
|---|---:|---:|---:|
| 7 | 20 | 1,000 | 0.8838 |
| 13 | 20 | 1,000 | **0.9710** |
| 22 | 20 | 1,000 | 0.9484 |

The Layer-13 group, **D20**, provides a concrete candidate representation for later experiments. Its activity is not exclusive to harmful requests: **421/500 benign requests** activate at least one member. These results measure request-label association; no responses were generated in this discovery experiment, and topics were not held out.

### 2. Compare the same request across framings

The persona experiment evaluates **48 underlying requests**—24 harmful and 24 corresponding benign requests—under direct, neutral, editor, and librarian framings, giving **192 prompt variants** across six authored topic families.

In two provisional fundraising refusal-to-assistance switches, substantial D20 activity remains under editor framing:

| Request | Direct harmful | Editor harmful | Matched editor benign |
|---|---:|---:|---:|
| Animal shelter | 15/20 active | **14/20 active** | 9/20 active |
| Food bank | 15/20 active | **14/20 active** | 9/20 active |

The counts conceal changes in feature identity: ten features remain active, five become inactive, and four become active. Across the full harmful-request panel, editor framing produces two provisional changes toward assistance, three away from assistance, 17 unchanged labels, and two unresolved comparisons.

**The motivating observation is that candidate-feature activity can coexist with assistance.** This makes functional testing necessary: activity alone does not establish that D20 recognizes harmful intent or enforces refusal.

### 3. Test whether candidate contributions influence the answer

A reconstruction audit checks measurement fidelity and tests a separately selected eight-feature group. The subsequent five-feature experiment asks whether restoring the contribution lost under editor framing changes behavior, and whether removing that contribution from direct prompts changes behavior.

The five-feature experiment contains **256 generated answers** across baselines, targeted edits, sham controls, and two random controls. Edits act at Layer 13 during the first prefill pass, preserve the remaining residual contribution, and are capped at 8% of the activation norm. Re-encoding checks whether the intended feature measurements changed.

These controls distinguish intervention delivery from a behavioral effect. Whole-answer blind review remains unfinished, so an opening-response check cannot establish a final refusal-flip rate or settle the candidate group's causal role.

### 4. Examine trajectories across model depth

The trajectory experiment compares six selected pairs, each with harmful-direct, harmful-editor, benign-direct, and benign-editor conditions: **24 prompt variants**, **26 layer outputs**, and **25 transitions**.

For a transition, the residual update is $u_\ell=h_{\ell+1}-h_\ell$. The Inter-layer Trajectory Divergence Metric (ITDM) measures its cosine distance from the mean direct-benign update from other topics. A matched comparison subtracts the benign editor-minus-direct shift from the harmful editor-minus-direct shift.

This tests whether selected switches have a distinctive trajectory pattern relative to retained-refusal comparisons. ITDM measures distance to the declared benign reference; it is not a validated refusal boundary. Similar curves or small full-residual distances cannot by themselves locate the mechanism responsible for a behavioral change.

## What the pilots motivate

The pilots suggest that candidate-feature activity and large-scale trajectory divergence alone do not reliably distinguish persona-associated answer switches in the selected cases. The next step is to test which measurements generalize to unseen requests and which candidate representations causally influence refusal across framings.

Representational similarity, causal influence, and behavioral review remain separate parts of that question. The current evidence does not establish a downstream override mechanism or identify a subtle causal subspace. Persona-response labels are provisional, and benign adequacy under persona framing requires further review.

## The six notebooks

| Notebook | Role in the research |
|---|---|
| [01 — Feature coverage](notebooks/01_feature_coverage.ipynb) | Establish the limits of the observed feature inventory. |
| [02 — Candidate discovery](notebooks/02_candidate_discovery.ipynb) | Select and evaluate the frozen request-label-associated groups. |
| [03 — Reconstruction and removal](notebooks/03_reconstruction_and_removal.ipynb) | Audit SAE fidelity and test a separate eight-feature group. |
| [04 — Persona activation mapping](notebooks/04_persona_activation_mapping.ipynb) | Compare candidate activity and responses across matched framings. |
| [05 — Five-feature interventions](notebooks/05_five_feature_interventions.ipynb) | Test a selected contribution against intervention controls. |
| [06 — Layer trajectories](notebooks/06_layer_trajectories.ipynb) | Compare residual updates and states across model depth. |

Each notebook includes figures, numeric tables, and its measurement definitions. See [RUNNING.md](RUNNING.md) for CPU replay and Kaggle instructions, and [source_manifest.json](source_manifest.json) for provenance.

Resources: [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it), [Gemma Scope 2 1B IT](https://huggingface.co/google/gemma-scope-2-1b-it), [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak).
