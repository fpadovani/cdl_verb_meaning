# Is Child-Directed Language Optimized for Word Learning? A Computational Study of Verb Meaning Acquisition

This repository contains the code and data to replicate the experiments of the paper by Francesca Padovani, Jaap Jumelet, Yevgen Matusevych and Arianna Bisazza.

## Setup

Create a virtual environment with Python 3.10.12:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Generation Pipeline

The experiments use four corpora: **CHILDES**, **Wikipedia**, **CANDOR**, and **BNC**. Each must be downloaded manually before running the pipeline.

- **CHILDES**: Download the `childes.train` file from the [BabyLM challenge data](https://babylm.github.io/) (`train_100M` split) and place it under `corpora/source/train_100M/`.
- **Wikipedia**: Download two Wikipedia text dump files and place them as `corpora/source/wikipedia1.txt` and `corpora/source/wikipedia2.txt`.
- **CANDOR**: Access is available upon request from the dataset authors via the [official CANDOR website](https://guscooney.com/candor-dataset/). Once your request is approved, you will receive the data by email. After downloading, update the `main_folder` path in the notebook to point to the received data folder. Note that, as described in the paper, the CANDOR dataset splits are composed of three sources: CANDOR itself, the spoken portion of the [British National Corpus (BNC)](https://www.english-corpora.org/bnc/), and the Switchboard corpus, which is included in the BabyLM challenge `train_100M` data.
- **BNC**: The written and spoken portions of the BNC were accessed through the `2554/` folder available in the repository of [You et al. (2021)](https://www.nature.com/articles/s41598-021-95392-x), which inspired our work. The data can be accessed via their [OSF repository](https://osf.io/hcj7y/?view_only=87bf26f2343c4a9ebc93d69aaaf6eddb). Place the `2554/` folder locally and update the `root_dir` path in the notebook accordingly.

Once all corpora are in place, run the notebook to clean the data and generate the train, dev, and test splits for each corpus:

```bash
generate_experimental_conditions/prepare_data.ipynb
```

### Generating Experimental Conditions

Three manipulated versions of each corpus can then be generated using the following notebooks:

- **Replace Word** (`generate_experimental_conditions/replace_word.ipynb`): replaces nouns, adjectives, adverbs, and embedded verbs with frequency-matched words of the same POS, leaving the main verb unchanged.
- **Shuffle Order — 1-gram** (`generate_experimental_conditions/shuffle_order_1gram.ipynb`): randomly shuffles all tokens in each sentence, keeping punctuation in place.
- **Shuffle Order — NP-level** (`generate_experimental_conditions/shuffle_order_np.ipynb`): shuffles word order while preserving the internal structure of noun phrases, following [Zhu et al. (2025)](https://arxiv.org/abs/2508.12482).


## Model Training

Models are trained as causal language models (GPT-2) across 5 random seeds. Below is an example training command for the CHILDES `replace_word` condition:

```bash
for seed in 13 42 30 51 67; do
    CUDA_VISIBLE_DEVICES=1 python ./train/clm_trainer.py \
        --wandb_project sy-optimized \
        --tokenizer_name ./tokenizers/english/childes/childes_rand_tokenizer_new.json \
        --with_tracking \
        --per_device_train_batch_size 256 \
        --per_device_eval_batch_size 256 \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --seed $seed \
        --vocab_size 30000 \
        --lr_scheduler_type linear \
        --num_warmup_steps 0 \
        --gradient_accumulation_steps 1 \
        --push_to_hub \
        --output_dir ./model_trained/cds_w_$seed \
        --model_type gpt2 \
        --trust_remote_code \
        --dataset_folder ./datasets \
        --context_length 128 \
        --language en \
        --validation_type validation \
        --order random \
        --input_file ./corpora/english/CHILDES_rand/replace_word \
        --hub_model_id cds_w \
        --hub_token <your_huggingface_token>
done
```

Adjust `--input_file`, `--output_dir`, `--hub_model_id`, and `--tokenizer_name` for each corpus and condition.

## Evaluation

### Semantic Minimal Pairs

Corpus-specific semantic minimal pairs used for the verb meaning evaluation can be generated using:

```bash
jupyter notebook evaluation/clm_semantic/data/generate_minpairs.ipynb
```

Pre-generated minimal pairs for each corpus are already available under `evaluation/clm_semantic/data/verb_focus/`.

### Running All Evaluations

To run a comprehensive evaluation of semantic and syntactic minimal pairs across all test suites (Zorro, BLiMP, and Fit-CLAMS) for all models, use:

```bash
python evaluation/semantic_minimal_pairs/run_all_evaluations.py
```

This script covers all three datasets and all trained models across conditions.
