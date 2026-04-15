# TimeTank Temporal Annotation Experiment

This repository contains  the TimeTank temporal annotation experiment.

It includes:

- one `1000`-instance sample corpus
- two generation pipelines
  - a single-pass annotation pipeline
  - a multi-pass annotation-review pipeline
- one retained evaluation package
  - the gold-restricted precision/recall/F1 evaluation


## Repository Contents

- `data/TimeTankSample.csv`
  - the `1000`-row gold-standard sample used in the experiments
- `.env.example`
  - example environment configuration for OpenRouter access
- `timetank_annotation/`
  - shared Python package for prompts, schemas, configuration, logging, and annotators
- `scripts/run_single_pass_annotation.py`
  - runs the single-pass generation experiment
- `scripts/run_multi_pass_annotation.py`
  - runs the multi-pass generation experiment
- `generation_results/single_pass/`
  - bundled final outputs of the single-pass experiment
- `generation_results/multi_pass/`
  - bundled final outputs of the multi-pass experiment
- `evaluation/`
  - evaluation , including scripts, result tables and ailed outputs

## Experimental Setup

### 1. Single-pass generation

The single-pass pipeline sends each sentence once to the model and requests temporal annotations directly.

Core implementation files:

- `timetank_annotation/single_pass_annotator.py`
- `timetank_annotation/single_pass_prompts.py`
- `scripts/run_single_pass_annotation.py`

### 2. Multi-pass generation

The multi-pass pipeline uses an annotation-review loop:

- an annotation agent proposes annotations
- a review agent checks them against the TimeInfo interval semantics
- the loop can revise the output for up to `3` rounds

Core implementation files:

- `timetank_annotation/multi_pass_annotator.py`
- `timetank_annotation/multi_pass_prompts.py`
- `scripts/run_multi_pass_annotation.py`

## Evaluation

This evaluation computes:

- precision
- recall
- F1
- accuracy
- sentence-level exact rate

Important methodological note:

- the evaluation is gold-restricted
- the `1000` rows of `TimeTankSample.csv` are the fixed evaluation instances
- extra predicted spans that are not present in the gold standard are exported separately
- those extra spans are not included in the main precision/recall/F1 score

This choice is intentional and matches the retained protocol for this repository.

Main evaluation outputs:

- `evaluation/tables/model_metrics.csv`
- `evaluation/tables/per_label_metrics.csv`
- `evaluation/tables/confusion_matrix.csv`
- `evaluation/tables/experiment_deltas.csv`
- `evaluation/results/ignored_extra_predictions_all.csv`

## Installation

Python `3.10+` is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

The generation scripts require an OpenRouter API key:

Create a local environment file from the provided template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set your values. The repository includes the following example variables:

- `OPENROUTER_API_KEY`
- `OPENROUTER_APP_URL`
- `OPENROUTER_APP_TITLE`

You can also export the variables directly instead of using `.env`:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

Optional variables:

- `OPENROUTER_APP_URL`
- `OPENROUTER_APP_TITLE`

## How to Run

### Run the single-pass generation pipeline

```bash
python scripts/run_single_pass_annotation.py --reasoning-effort high
```

Default paths:

- input: `data/TimeTankSample.csv`
- output: `generation_results/single_pass`

### Run the multi-pass generation pipeline

```bash
python scripts/run_multi_pass_annotation.py --reasoning-effort high --max-review-rounds 3
```

Default paths:

- input: `data/TimeTankSample.csv`
- output: `generation_results/multi_pass`

### Run the retained evaluation

```bash
python evaluation/scripts/run_all_evaluation.py
```

This command:

- rebuilds the gold-restricted evaluation tables
- rebuilds the detailed result files


## Included Results

This repository already includes the final outputs for:

- the single-pass experiment
- the multi-pass experiment
- the retained precision/recall/F1 evaluation

So you can inspect the results immediately without rerunning generation.

## Notes

- this is kept for compatibility with the original experiment outputs
- the evaluation scripts work with those files as bundled in this repository
