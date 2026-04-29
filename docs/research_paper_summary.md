# Research Paper Summary: Geometric Complexity Scaling in Gemma Residual Streams

## Project Goal

This project tests whether increasingly complex language-model tasks induce increasingly nonlinear geometry in the residual stream of an instruction-tuned model. The central research question is:

> Does task complexity correspond to measurable changes in residual-stream geometry, such as weaker linear compression, higher intrinsic dimensionality, and higher trajectory curvature?

The experiment compares three task classes intended to span increasing cognitive complexity:

1. **Type I: Fact retrieval**
   The model answers short factual questions.

2. **Type II: Sentiment classification**
   The model classifies a sentence as positive or negative.

3. **Type III: Mathematical reasoning**
   The model solves grade-school math word problems and outputs a final numeric answer.

The initial hypothesis was that Type I should exhibit relatively low curvature and strong PCA compression, Type II should be intermediate, and Type III should exhibit higher curvature, weaker linear fit, and possible middle-layer intrinsic-dimension spikes.

## Model

The model used throughout the project is:

- **Model ID:** `google/gemma-4-E2B-it`
- **Model family:** Gemma instruction-tuned model
- **Reason for using instruction-tuned version:** The study evaluates task performance under natural instruction-following conditions, so the instruction-tuned checkpoint is more appropriate than a base model.
- **Generation mode:** deterministic decoding with `do_sample=False`
- **Thinking mode:** explicitly disabled for every prompt
- **Default dtype on HPC inference:** `bfloat16`
- **Device placement:** `device_map="auto"`

The code loads the model using Hugging Face Transformers. It first attempts `AutoModelForCausalLM`; if that fails, it falls back to `AutoModelForImageTextToText`, which is needed for some newer Gemma model classes. The processor is loaded with `AutoProcessor`.

## Prompting Protocol

Every task uses the same system prompt:

```text
You are a helpful assistant.
```

Prompts are formatted with the Gemma chat template using:

```python
processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

This is important because the experiment is intended to measure instruction-following behavior under the model's intended chat format. Thinking mode is disabled because the project is not studying explicit chain-of-thought or Gemma thinking behavior.

Task-specific user prompts are:

### Fact Retrieval

```text
Answer the factual question with only the shortest correct answer. Do not explain.

Question: {question}
```

### Sentiment Classification

```text
Classify the sentiment of the sentence. Reply with exactly one word: positive or negative.

Sentence: {text}
```

### Mathematical Reasoning

```text
Solve the math word problem. You may reason briefly, but the final line must be: Final answer: <number>

Problem: {question}
```

The math prompt permits brief reasoning because GSM8K generally requires multi-step computation, but correctness is based only on the final extracted numeric answer.

## Datasets

Each task uses a seeded subset of 2,000 examples per seed. The subset is selected by shuffling the dataset deterministically with the seed and selecting the first 2,000 rows. The three seeds are:

```text
0, 1, 2
```

### Type I: Fact Retrieval

- **Dataset:** `mandarjoshi/trivia_qa`
- **Configuration:** `rc.nocontext`
- **Split:** `train`
- **Relevant schema fields:**
  - `question`
  - `answer.value`
  - `answer.normalized_value`
  - `answer.matched_wiki_entity_name`
  - `answer.normalized_matched_wiki_entity_name`
  - `answer.aliases`
  - `answer.normalized_aliases`
- **Prompt input:** `question`
- **Target:** all available answer aliases and normalized answer forms

TriviaQA was chosen because it is a natural open-ended fact-retrieval benchmark. The no-context configuration reduces confounding from passage reading and makes the task closer to factual recall.

### Type II: Sentiment Classification

- **Dataset:** `SetFit/sst2`
- **Split:** `train`
- **Relevant schema fields:**
  - `text`
  - `label`
  - `label_text`
- **Prompt input:** `text`
- **Target:** `positive` or `negative`

SST-2 was chosen because it is simple to use, has clear binary labels, and is a standard sentiment benchmark.

### Type III: Mathematical Reasoning

- **Dataset:** `openai/gsm8k`
- **Configuration:** `main`
- **Split:** `train`
- **Relevant schema fields:**
  - `question`
  - `answer`
- **Prompt input:** `question`
- **Target:** final numeric answer extracted from the gold solution

GSM8K was chosen because it requires multi-step arithmetic reasoning and is substantially more complex than factual lookup or binary sentiment classification.

## Correctness Scoring

The experiment first runs inference on all 2,000 examples for each task and seed. Each output is scored for task correctness. Correctness is used both as a performance measure and as a filter for the main geometry analysis.

### Fact Retrieval Scoring

TriviaQA scoring is deliberately more permissive than raw exact match because open-ended answers can be phrased in multiple valid ways. The implementation uses official-style normalization:

- Lowercase text
- Replace underscores with spaces
- Remove punctuation
- Remove English articles `a`, `an`, and `the`
- Collapse whitespace

The model output is cleaned by removing special tokens, stripping common answer prefixes, and extracting a short answer span.

For each prediction, the code compares against all available answer strings from `value`, `normalized_value`, matched entity names, aliases, and normalized aliases. It computes:

- Normalized exact match
- Token-level F1 against each alias
- Maximum alias F1

A TriviaQA answer is marked correct if:

```text
exact_match == true OR max_f1 >= 0.5
```

The threshold was included to avoid systematically undercounting open-ended fact answers that partially match a longer alias.

### Sentiment Scoring

The output is parsed for the first valid label in:

```text
positive, negative
```

Correctness is exact normalized label match.

### Math Scoring

The system extracts the final number from the model output and compares it to the gold final answer extracted from the GSM8K answer field. Numeric comparison handles typical formatting differences such as commas or equivalent numeric strings.

## Inference Records

For every example, the system saves a JSONL record containing:

- Task name and task type
- Dataset name and configuration
- Seed
- Dataset row id and local index
- Chat messages
- Rendered prompt
- Target
- Raw model output
- Parsed response
- Parsed answer
- Score details
- Correctness
- Input length
- Generated length
- Generated token IDs
- Creation timestamp

This design makes the experiment auditable and enables later activation backfilling without rerunning generation.

## Activation Extraction

The core geometric object is the residual-stream hidden-state trajectory across model layers for a comparable token position.

The primary activation point is:

```text
final generated answer token
```

For each retained example, the code stores one hidden-state vector per layer at this final generated token position. This yields an activation tensor with shape approximately:

```text
num_examples x num_layers x hidden_dim
```

In this run, Gemma hidden states produced 36 layer-indexed vectors with hidden dimension 1536.

### Correct Activations

During the original inference run, hidden states were retained for examples that the model answered correctly. These were saved in files named:

```text
outputs/gemma4_e2b/activations/{task}_seed{seed}_correct_activations.npz
```

### Incorrect Activations

After initial analysis, incorrect-example activations were added to support correct-vs-incorrect comparisons. To avoid rerunning generation, the system uses the saved prompt and saved generated token IDs from the inference JSONL files. It performs a teacher-forced forward pass over:

```text
saved_prompt_tokens + saved_generated_token_ids
```

and extracts hidden states at the final generated token. This avoids the cost and stochastic risk of regenerating outputs while recovering the missing residual-stream states for incorrect examples.

Incorrect activations are saved in:

```text
outputs/gemma4_e2b/activations/{task}_seed{seed}_incorrect_activations.npz
```

## Geometry Metrics

Metrics are computed per task, seed, outcome group, and layer. Outcome groups are:

```text
correct
incorrect
```

The main metric file is:

```text
outputs/gemma4_e2b/metrics/geometry_metrics_all.csv
```

### PCA Metrics

PCA is used to test how well the layerwise activation cloud can be approximated by a low-dimensional linear subspace.

The implementation uses up to 10 PCA components:

```text
n_components = min(10, num_examples - 1, hidden_dim)
```

The reported PCA metrics are:

- `pca_explained_variance`
- `pca_residual_variance = 1 - explained_variance`
- `pca_reconstruction_mse`

Higher PCA residual variance and higher reconstruction MSE indicate weaker linear compressibility.

### Isomap

Isomap is used as a nonlinear manifold diagnostic. The metric reported is:

```text
isomap_reconstruction_error
```

The implementation uses up to 12 neighbors, capped by the number of examples:

```text
n_neighbors = min(12, num_examples - 1)
```

Higher Isomap reconstruction error suggests a less well-preserved low-dimensional manifold under the chosen neighborhood graph and embedding dimension.

### Locally Linear Embedding

LLE is also computed as a nonlinear manifold diagnostic:

```text
lle_reconstruction_error
```

Although not requested in the latest compact summary table, it is included in the pipeline and plots.

### Global TwoNN Intrinsic Dimensionality

The TwoNN estimator uses the ratio between first and second nearest-neighbor distances. It estimates a global intrinsic dimensionality for the layerwise activation cloud:

```text
intrinsic_dim_twonn
```

Higher values suggest a higher-dimensional activation geometry, but the estimator can be sensitive to sample size, clustering, and anisotropy.

### Local Intrinsic Dimensionality

To address the limitations of global TwoNN, the implementation also computes local maximum-likelihood intrinsic-dimensionality estimates:

```text
local_id_mle_k10
local_id_mle_k20
```

These use k-nearest-neighbor distance profiles to estimate local dimensionality and then average across examples. These metrics are intended to better capture local geometric complexity than a single global TwoNN estimate.

### Layer-Path Curvature

The experiment also measures trajectory curvature across layers, rather than only analyzing isolated layer snapshots. For each example and interior layer, the implementation computes a finite-difference curvature-like quantity using consecutive residual-state differences:

```text
prev_vec = h_l - h_{l-1}
next_vec = h_{l+1} - h_l
curvature_l = ||next_vec - prev_vec|| / (||prev_vec||^2 + epsilon)
```

The metric is averaged over examples for each layer. Endpoint layers have undefined curvature and are stored as NaN.

The main curvature metric is:

```text
euclidean_layer_curvature
```

### Semantic-Pullback Curvature Approximation

The pipeline also computes an approximate semantic/logit-space curvature. It projects residual vectors through a deterministic sampled subset of the model output embedding/unembedding matrix. The sample size is:

```text
semantic_vocab_sample = 256
```

This gives a tractable approximation to how residual-stream geometry behaves under a projection related to output-token logits. The resulting metric is:

```text
semantic_pullback_layer_curvature
```

## Visualization

The plotting pipeline generates mean ± standard deviation plots over the three random seeds. Plots are saved under:

```text
outputs/gemma4_e2b/plots/
```

Generated plots include:

- `accuracy_by_task.png`
- `correct_counts_by_task.png`
- `pca_residual_variance_by_layer.png`
- `pca_reconstruction_mse_by_layer.png`
- `isomap_reconstruction_error_by_layer.png`
- `lle_reconstruction_error_by_layer.png`
- `intrinsic_dim_twonn_by_layer.png`
- `local_id_mle_k10_by_layer.png`
- `local_id_mle_k20_by_layer.png`
- `euclidean_layer_curvature_by_layer.png`
- `semantic_pullback_layer_curvature_by_layer.png`

For correct-vs-incorrect comparisons, the plots distinguish `outcome_group`.

## Experimental Setup

### Sampling

For each task and each seed:

1. Load the dataset split.
2. Shuffle deterministically using the seed.
3. Select the first 2,000 examples.
4. Run Gemma inference in chat format.
5. Score correctness.
6. Save all inference records.
7. Save correct activations during generation.
8. Backfill incorrect activations from saved generations using teacher forcing.
9. Compute metrics separately for correct and incorrect groups.
10. Aggregate and plot metrics across seeds.

### Seeds

The experiment uses three seeds:

```text
0, 1, 2
```

These seeds affect dataset shuffling and deterministic sampling of the unembedding subset for semantic projection.

### Hardware and HPC Execution

The full experiment was run on an HPC cluster with Slurm. Inference and activation backfilling are parallelized as Slurm arrays over the 9 combinations of:

```text
3 tasks x 3 seeds
```

The array concurrency limit is:

```text
#SBATCH --array=0-8%4
```

This permits up to four concurrent GPU jobs, matching the computational budget.

Each Slurm array task uses:

```text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --time=96:00:00
```

The postprocessing job runs after successful completion of the full inference or backfill array using a Slurm `afterok` dependency.

### Workspace Paths

Main output directory:

```text
/scratch/afz225/res_stream_project/outputs/gemma4_e2b
```

Hugging Face cache:

```text
/scratch/afz225/.cache
```

Conda environment:

```text
/scratch/afz225/res_stream_project/.conda-env
```

Slurm logs:

```text
/scratch/afz225/res_stream_project/slurm/logs
```

## Software Environment

The project is implemented as a Python package named:

```text
geometric-complexity-scaling
```

Primary libraries:

- Python 3.10
- PyTorch 2.5.1
- Torchvision 0.20.1
- CUDA 12.1 through `pytorch-cuda=12.1`
- Hugging Face Transformers >= 4.55
- Hugging Face Datasets >= 2.19
- Accelerate >= 0.30
- NumPy >= 1.24
- Pandas >= 2.0
- Scikit-learn >= 1.3
- Matplotlib >= 3.7
- Seaborn >= 0.13
- tqdm >= 4.66
- pytest >= 8.0

The Conda environment is specified in:

```text
environment.yml
```

The pip/package metadata is specified in:

```text
pyproject.toml
requirements.txt
```

## Quality Control and Tests

The unit test suite validates major components without downloading Gemma. The implemented tests cover:

- Dataset schema handling for TriviaQA, SST-2, and GSM8K
- Optional real dataset contract tests when `RUN_DATASET_CONTRACT_TESTS=1`
- TriviaQA exact match and F1 scoring behavior
- Open-ended fact-answer cleanup
- Sentiment answer parsing
- GSM8K numeric parsing
- Gemma chat formatting with `enable_thinking=False`
- Geometry metric behavior on synthetic activations
- Empty activation handling
- CLI smoke tests for metrics and plotting

The current test result was:

```text
17 passed, 1 skipped
```

## Observed Task Performance

Each task was evaluated on 2,000 examples per seed.

| Task | Seed 0 Accuracy | Seed 1 Accuracy | Seed 2 Accuracy | Mean Accuracy |
|---|---:|---:|---:|---:|
| fact | 0.4625 | 0.4640 | 0.4825 | 0.4697 |
| math | 0.4190 | 0.4190 | 0.4375 | 0.4252 |
| sentiment | 0.8785 | 0.8620 | 0.8760 | 0.8722 |

Correct and incorrect retained examples:

| Task | Seed | Correct Examples | Incorrect Examples |
|---|---:|---:|---:|
| fact | 0 | 925 | 1075 |
| fact | 1 | 928 | 1072 |
| fact | 2 | 965 | 1035 |
| math | 0 | 838 | 1162 |
| math | 1 | 838 | 1162 |
| math | 2 | 875 | 1125 |
| sentiment | 0 | 1757 | 243 |
| sentiment | 1 | 1724 | 276 |
| sentiment | 2 | 1752 | 248 |

The sentiment task has much higher accuracy than fact retrieval or math, which matters for interpretation because the incorrect sentiment group is much smaller.

## Correct-vs-Incorrect Geometry Summary

The following table summarizes the direction of the mean difference between correct and incorrect activations. It aggregates across seeds and non-final layers. A direction is reported only if the sign is consistent across seeds and the relative gap is at least approximately 2%; otherwise the entry is marked as no clear difference.

| Metric | fact | math | sentiment |
|---|---|---|---|
| PCA residual variance | correct > incorrect | incorrect > correct | correct > incorrect |
| PCA reconstruction MSE | correct > incorrect | incorrect > correct | correct > incorrect |
| Isomap | correct > incorrect | incorrect > correct | correct > incorrect |
| TwoNN ID | incorrect > correct | no clear difference | correct > incorrect |
| Curvature | correct > incorrect | incorrect > correct | correct > incorrect |

Interpretation:

- For **math**, incorrect examples consistently show higher PCA residual variance, higher PCA reconstruction MSE, higher Isomap error, and higher curvature. This supports the idea that failed reasoning cases occupy a more nonlinear or less linearly compressible residual geometry.
- For **fact** and **sentiment**, correct examples often show higher PCA residual variance, higher reconstruction MSE, higher Isomap error, and higher curvature. This suggests that correctness is not always associated with simpler geometry. In easier or more retrieval/classification-oriented tasks, correct answers may involve richer or more varied internal representations, while incorrect outputs may collapse into simpler failure modes.
- TwoNN ID behaves less consistently than the other metrics. It indicates higher incorrect dimensionality for fact retrieval, no clear math difference, and higher correct dimensionality for sentiment. This supports the decision to include local ID estimators as additional diagnostics.

## Relevance to the Original Hypothesis

The original hypothesis predicted a monotonic increase in nonlinear geometry from fact retrieval to sentiment to math reasoning. The results are more nuanced:

- Math reasoning does show the expected relationship between failure and higher nonlinear/geometric complexity.
- Sentiment is much easier for the model and has fewer incorrect examples, so correct-vs-incorrect comparisons are affected by class imbalance in retained examples.
- Fact retrieval is open-ended and alias-sensitive. Even with improved F1 scoring, correctness may reflect answer surface form and knowledge availability as much as task complexity.
- The geometry of correct examples is not necessarily simpler. Correct performance may require using a more expressive part of the residual-stream representation, while incorrect answers may result from generic or low-diversity failure trajectories.

A stronger paper framing would therefore avoid claiming a simple monotonic task-complexity law. A more defensible conclusion is:

> Residual-stream geometry varies systematically with both task type and outcome correctness, but the relationship is not a simple monotonic function of task complexity. Mathematical reasoning failures exhibit higher nonlinear and curvature-based complexity, while factual and sentiment tasks show evidence that correct responses can occupy richer or less linearly compressible geometries.

## Suggested Paper Structure

### Abstract

State that the project investigates whether task complexity and correctness are reflected in residual-stream geometry for `google/gemma-4-E2B-it`. Mention the three task types, 2,000 examples per task/seed, three seeds, hidden-state extraction at the final generated answer token, and metrics such as PCA residual variance, Isomap error, intrinsic dimension, and layer-path curvature. Summarize the main finding: math failures show higher nonlinear complexity, while fact and sentiment do not follow a simple monotonic complexity pattern.

### Introduction

Motivate the question of whether neural representations become geometrically more complex for harder tasks. Introduce residual streams as a useful representation space for transformer interpretability. Explain why comparing fact retrieval, sentiment classification, and mathematical reasoning is a useful testbed.

### Related Work

Discuss:

- Transformer residual stream analysis
- Representation geometry in neural networks
- Intrinsic dimensionality estimation
- Linear probes and PCA-based compression
- Manifold learning methods such as Isomap and LLE
- Reasoning benchmarks such as GSM8K
- Instruction-tuned language models and chat formatting

### Methods

Include:

- Model details
- Datasets and task definitions
- Prompt formatting and disabled thinking mode
- Sampling protocol over three seeds
- Correctness scoring
- Hidden-state extraction
- Correct and incorrect activation handling
- Geometry metrics
- Aggregation and visualization
- HPC setup

### Results

Include:

- Accuracy by task
- Correct/incorrect retained-example counts
- PCA residual variance by layer
- PCA reconstruction MSE by layer
- Isomap and LLE error by layer
- TwoNN and local ID by layer
- Euclidean and semantic-pullback curvature by layer
- Correct-vs-incorrect summary table

### Discussion

Emphasize:

- Math failures show the clearest evidence for higher nonlinear geometric complexity.
- Correct examples are not universally simpler.
- Task difficulty, answer format, and retained-example imbalance affect interpretation.
- Different metrics capture different aspects of geometry.
- Global TwoNN ID may be unstable; local ID metrics are useful additions.

### Limitations

Important limitations:

- Only one model was tested.
- Only three task families were tested.
- The sample size was 2,000 examples per task/seed, not the full datasets.
- Correct and incorrect groups are imbalanced, especially for sentiment.
- Hidden states are analyzed at the final generated answer token, which may miss earlier reasoning dynamics.
- Incorrect activations are recovered by teacher-forced forward passes over saved generations, which is computationally efficient but not identical to storing every generation-time hidden state.
- Geometry metrics such as Isomap, LLE, and intrinsic dimension can be sensitive to sample size, neighborhood choices, anisotropy, and preprocessing.
- The semantic-pullback curvature is an approximation based on a sampled unembedding subset rather than a full vocabulary projection.
- TriviaQA correctness remains imperfect because open-ended factual answers have many valid surface forms.

### Conclusion

A defensible conclusion is that residual-stream geometry contains measurable information about task type and correctness, but the relationship between task complexity and geometric complexity is not monotonic. The strongest evidence for the original hypothesis appears in mathematical reasoning failures, which show higher nonlinear reconstruction error and curvature than correct math solutions. Fact retrieval and sentiment classification show different patterns, suggesting that correctness, task format, and representation diversity all influence residual-stream geometry.

## Key Files and Reproducibility Artifacts

Source code:

- `src/geometric_complexity_scaling/tasks.py`
- `src/geometric_complexity_scaling/inference.py`
- `src/geometric_complexity_scaling/backfill.py`
- `src/geometric_complexity_scaling/geometry.py`
- `src/geometric_complexity_scaling/plotting.py`
- `src/geometric_complexity_scaling/scoring.py`

Environment:

- `environment.yml`
- `requirements.txt`
- `pyproject.toml`

Slurm scripts:

- `slurm/run_inference_array.sbatch`
- `slurm/run_postprocess.sbatch`
- `slurm/submit_pipeline.sh`
- `slurm/backfill_incorrect_activations.sbatch`
- `slurm/submit_backfill_pipeline.sh`

Outputs:

- `outputs/gemma4_e2b/inference/*.jsonl`
- `outputs/gemma4_e2b/activations/*.npz`
- `outputs/gemma4_e2b/metrics/*.csv`
- `outputs/gemma4_e2b/plots/*.png`

## Concise Report-Ready Claim

This project provides an empirical test of whether transformer residual-stream geometry scales with task complexity. Using `google/gemma-4-E2B-it` on fact retrieval, sentiment classification, and mathematical reasoning tasks, the study extracts final-answer-token hidden states across layers and evaluates linear compressibility, nonlinear manifold reconstruction, intrinsic dimensionality, and layer-path curvature. The results show that incorrect mathematical reasoning examples have higher nonlinear reconstruction error and curvature than correct examples, consistent with the hypothesis that failed reasoning occupies a more complex residual geometry. However, fact retrieval and sentiment classification show the opposite direction for several metrics, indicating that residual-stream geometric complexity is shaped by correctness, answer diversity, and task format rather than task difficulty alone.
