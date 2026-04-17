import os
import pickle
import pandas as pd
import tempfile
import shutil
import csv
from collections import defaultdict
from statistics import mean
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from minicons import scorer


model_eval_domains = {
    "cds": ["childes"],
    "candor": ["candor"],
    "bnc": ["bnc"],
    "wiki": ["wiki"]
}

blimp_dir = "./evaluation/syntactic_test_suites/blimp"
output_dir_blimp = "./evaluation/blimp_zorro_scores/blimp/new"
os.makedirs(output_dir_blimp, exist_ok=True)

zorro_dir = "./evaluation/syntactic_test_suites/zorro"
output_dir_zorro = "./evaluation/blimp_zorro_scores/zorro/zorro_new"
os.makedirs(output_dir_zorro, exist_ok=True)

fitclams_dirs = {
    "childes": "./evaluation/syntactic_test_suites/en_fitclams/childes",
    "wiki": "./evaluation/syntactic_test_suites/en_fitclams/wiki",
    "bnc": "./evaluation/syntactic_test_suites/en_fitclams/bnc",
    "candor": "./evaluation/syntactic_test_suites/en_fitclams/candor"
}

output_dirs_fitclams = {
    "childes": "./evaluation/blimp_zorro_scores/fitclams/new/childes",
    "wiki": "./evaluation/blimp_zorro_scores/fitclams/new/wiki",
    "bnc": "./evaluation/blimp_zorro_scores/fitclams/new/bnc",
    "candor": "./evaluation/blimp_zorro_scores/fitclams/new/candor"
}

for d in output_dirs_fitclams.values():
    os.makedirs(d, exist_ok=True)

models = [ 'wiki_o_30', 'candor_o_30', 'bnc_o_30', 'cds_o_30']

# and you can add more models to the list as needed, just make sure they follow the naming convention and are available in the Hugging Face Hub under the "fpadovani" namespace.

# --------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------
def atomic_write(path, write_fn):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix="tmp-")
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_checkpoint_csvs(ckpt_name, results, output_base, repo_subdir):
    base = os.path.join(output_base, repo_subdir, "checkpoints", ckpt_name)
    os.makedirs(base, exist_ok=True)
    for paradigm, df in results.items():
        out_path = os.path.join(base, f"{paradigm}_scores.csv")
        atomic_write(out_path, lambda tmp: df.to_csv(tmp, index=False))

def load_accuracy_summary(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_accuracy_summary_row(output_base, repo_subdir, ckpt_name, accuracies_dict):
    path = os.path.join(output_base, repo_subdir, "accuracy_summary.csv")
    acc_df = load_accuracy_summary(path)
    if 'checkpoint' not in acc_df.columns:
        acc_df = pd.DataFrame(columns=['checkpoint'] + sorted(accuracies_dict.keys()))
    new_row = {'checkpoint': ckpt_name}
    new_row.update(accuracies_dict)
    acc_df = acc_df[acc_df['checkpoint'] != ckpt_name]
    acc_df = pd.concat([acc_df, pd.DataFrame([new_row])], ignore_index=True, sort=False)
    cols = ['checkpoint'] + [c for c in acc_df.columns if c != 'checkpoint']
    acc_df = acc_df[cols]
    atomic_write(path, lambda tmp: acc_df.to_csv(tmp, index=False))

def save_full_state(output_base, repo_subdir, all_results, all_accuracies):
    base = os.path.join(output_base, repo_subdir, "state")
    os.makedirs(base, exist_ok=True)
    atomic_write(os.path.join(base, "all_state.pkl"),
                 lambda tmp: pickle.dump({'all_results': all_results, 'all_accuracies': all_accuracies}, open(tmp, "wb")))

def download_model_and_checkpoints(repo_id: str):
    print(f"Downloading repository: {repo_id} ...")
    repo_path = snapshot_download(repo_id=repo_id)
    print(f"Local snapshot path: {repo_path}\n")

    all_items = os.listdir(repo_path)
    check_folders = sorted(
        [item for item in all_items if item.startswith("check-") and os.path.isdir(os.path.join(repo_path, item))],
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else x
    )
    check_folders.append("final-check")

    local_checkpoints = {ckpt: (repo_path if ckpt == "final-check" else os.path.join(repo_path, ckpt))
                         for ckpt in check_folders}

    for name, path in local_checkpoints.items():
        print(f"{name:20s} -> {path}")
    return repo_path, local_checkpoints


# --------------------------------------------------------
# SEMANTIC MINIMAL PAIRS EVALUATION FUNCTION
# --------------------------------------------------------
def evaluate_semantic_min_pairs_per_bin_resume(checkpoint_paths, eval_file_path, output_csv_path, tokenizer,
                                              limit_pairs=140000, batch_size=64, cuda_device="cuda:1"):
    from collections import defaultdict


    def load_pairs(path):
        pairs_data = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s1, s2, bin_num = row.get("sentence1"), row.get("sentence2"), row.get("bin_num")
                if not s1 or not s2 or not bin_num: 
                    continue
                # strip whitespace from bin_num
                pairs_data.append((s1.strip().lower(), s2.strip().lower(), bin_num.strip()))
        return pairs_data

    raw_pairs = load_pairs(eval_file_path)[:limit_pairs]
    all_sentences, pair_map = [], []
    for s1, s2, bin_num in raw_pairs:
        all_sentences.extend([s1, s2])
        pair_map.append(bin_num)

    processed_checkpoints = set()
    if os.path.exists(output_csv_path):
        try:
            with open(output_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                processed_checkpoints = set(row["checkpoint"] for row in reader)
        except Exception:
            pass

    def evaluate_checkpoint_per_bin_manual_batch(ckpt_name, ckpt_path):
        print(f"Loading model for {ckpt_name}...")
        try:
            lm_scorer = scorer.IncrementalLMScorer(model=ckpt_path, tokenizer=tokenizer, device=cuda_device)
        except Exception as e:
            print(f"Failed to load model from {ckpt_path}: {e}")
            return None
        all_scores = []
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:i + batch_size]
            try:
                scores = lm_scorer.sequence_score(batch)
                all_scores.extend(scores)
            except Exception as e:
                print(f"Error scoring batch at {i}: {e}")
                return None
        if len(all_scores) != len(all_sentences):
            return None
        bin_results = defaultdict(lambda: {"correct": 0, "total": 0})
        for i, bin_num in enumerate(pair_map):
            score1 = all_scores[2 * i]
            score2 = all_scores[2 * i + 1]
            bin_results[bin_num]["total"] += 1
            if score1 > score2:
                bin_results[bin_num]["correct"] += 1
        return {b: (v["correct"] / v["total"]) for b, v in bin_results.items()}

    unique_bins = sorted(list(set(pair_map)))
    fieldnames = ["checkpoint"] + unique_bins
    file_exists = os.path.exists(output_csv_path)

    with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for ckpt_name, ckpt_path in sorted(checkpoint_paths.items()):
            if ckpt_name in processed_checkpoints:
                print(f"Skipping {ckpt_name}")
                continue
            bin_acc = evaluate_checkpoint_per_bin_manual_batch(ckpt_name, ckpt_path)
            if bin_acc is None:
                print(f"Skipping {ckpt_name} (failed)")
                continue
            row = {"checkpoint": ckpt_name}
            row.update(bin_acc)
            writer.writerow(row)
            f.flush()
            print(f"Saved results for {ckpt_name}")

def evaluate_checkpoint_zorro_blimp_optimized(checkpoint_name, model_path, tokenizer, dataset_dir, max_pairs=None, batch_size=64):
    """
    Evaluates a checkpoint on the Zorro dataset using manual batch processing.
    """
    device = "cuda:1"
    # Use reduction=lambda x: x.sum(0).item() as provided in original function
    reduction = lambda x: x.sum(0).item()
    
    print(f"\n🚀 Evaluating checkpoint: {checkpoint_name} (Batch size: {batch_size})")
    
    # 1. Load Scorer Model
    try:
        scorer_model = scorer.IncrementalLMScorer(model=model_path, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"❌ Failed to load model for {checkpoint_name}: {e}")
        # Return empty results and accuracies on failure
        return {}, {'error': str(e)} 

    file_results = {}
    acc_per_file = {}

    for filename in sorted(os.listdir(dataset_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(dataset_dir, filename)
        paradigm_name = filename.replace(".txt", "")

        with open(filepath, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # Data Cleaning and Preparation
        if len(lines) % 2 != 0:
            print(f"⚠️ Warning: Odd number of lines in {filename}. Dropping last line.")
            lines = lines[:-1]

        # 2. Extract and Prepare All Sentences for Batching
        # Pairs: [(grammatical_sent, ungrammatical_sent), ...]
        pairs_full = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
        
        if max_pairs is not None:
            pairs = pairs_full[:max_pairs]
        else:
            pairs = pairs_full
        
        if not pairs:
            print(f"⚠️ Skipping {paradigm_name}: 0 pairs found after filtering.")
            continue
            
        # Create a single list of all sentences, ordered: [G1, U1, G2, U2, ...]
        all_sentences = []
        for sent_gramm, sent_ungramm in pairs:
            # Lowercase sentences once here (Optimization)
            all_sentences.append(sent_gramm.lower()) 
            all_sentences.append(sent_ungramm.lower())
            
        num_sentences = len(all_sentences)
        
        # 3. Batch Scoring (Manual Iteration Fix)
        all_scores = []
        print(f"  Scoring {paradigm_name} ({len(pairs)} pairs) in batches...")
        
        for i in range(0, num_sentences, batch_size):
            batch = all_sentences[i:i + batch_size]
            try:
                # Call sequence_score on the batch
                scores = scorer_model.sequence_score(batch, reduction=reduction)
                all_scores.extend(scores)
            except Exception as e:
                print(f"❌ Error scoring batch in {paradigm_name} at index {i}: {e}. Skipping paradigm.")
                acc_per_file[paradigm_name] = 0.0
                file_results[paradigm_name] = pd.DataFrame()
                break

        if len(all_scores) != num_sentences:
            continue

        correct = 0
        results_list = []
        num_pairs = len(pairs)
        
        for i in range(num_pairs):
            sent_gramm, sent_ungramm = pairs[i]
            prob_gramm = all_scores[2 * i]
            prob_ungramm = all_scores[2 * i + 1]

            is_correct = prob_gramm > prob_ungramm
            correct += int(is_correct)

            results_list.append({
                "sentence_gramm": sent_gramm,
                "sentence_ungramm": sent_ungramm,
                f"probability_{checkpoint_name}_gramm": prob_gramm,
                f"probability_{checkpoint_name}_ungramm": prob_ungramm,
                f"correct_{checkpoint_name}": is_correct
            })

        acc = correct / num_pairs
        acc_per_file[paradigm_name] = acc
        file_results[paradigm_name] = pd.DataFrame(results_list)

        print(f"✅ {paradigm_name}: {acc:.3f} (on {num_pairs} pairs)")

    return file_results, acc_per_file


def evaluate_checkpoint_fitclams_optimized(
    checkpoint_name, model_path, tokenizer, dataset_dir,
    lang="en", max_pairs=None, batch_size=64
):
    """
    Optimized version of evaluate_checkpoint_csv_fine:
    - batches sentences for token_score
    - tokenizes only once per sentence
    - computes verb spans using tokens (not string search)
    - avoids per-sentence model calls
    """
    import numpy as np

    device = "cuda:1"
    reduction = lambda x: x.sum(0).item()

    print(f"\n🚀 Evaluating checkpoint (optimized): {checkpoint_name} (batch size {batch_size})")

    # ---- Load model ----
    try:
        model = scorer.IncrementalLMScorer(
            model=str(model_path),
            tokenizer=tokenizer,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {}, {'error': str(e)}

    file_results = {}
    acc_per_file = {}

    # --------------------------------------------------------------------------
    # Helper: find verb spans in *tokenized* form
    # --------------------------------------------------------------------------
    def get_verb_span_tokens(tokens, paradigm_name):
        """
        tokens: list of strings (tokenized)
        Returns (start, end) token indices or None
        """

        try:
            # Long VP coordination
            if "long_vp_coord" in paradigm_name:
                if lang == "en":
                    coord = "and"
                elif lang == "fr":
                    coord = "et"
                elif lang == "de":
                    coord = "und"
                idx = tokens.index(coord) + 1
                return (idx, idx + 1)

            # Object-relative within animate
            elif "obj_rel_within_anim" in paradigm_name:
                if lang in ["en", "fr"]:
                    idx = len(tokens) - 2
                elif lang == "de":
                    idx = len(tokens) - 3
                return (idx, idx + 1)

            # Default: last verb (last token)
            else:
                return (len(tokens) - 1, len(tokens))

        except ValueError:
            return None

    # --------------------------------------------------------------------------
    # MAIN LOOP OVER FILES
    # --------------------------------------------------------------------------
    for filename in sorted(os.listdir(dataset_dir)):
        if not filename.endswith(".csv") or "scored" in filename:
            continue

        filepath = os.path.join(dataset_dir, filename)
        paradigm_name = filename.replace(".csv", "")

        df = pd.read_csv(filepath, header=None)
        lines = [str(l).strip() for l in df[0].tolist() if str(l).strip()]

        if len(lines) % 2 != 0:
            lines = lines[:-1]

        pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        if not pairs:
            continue

        # Flatten for batching
        all_sentences = []
        for g, u in pairs:
            all_sentences.append(g)
            all_sentences.append(u)

        # Pre-tokenize once
        all_tokenized = [tokenizer.tokenize(s) for s in all_sentences]
        all_spans = [
            get_verb_span_tokens(tok, paradigm_name)
            for tok in all_tokenized
        ]

        # Batch scoring
        all_scores = []
        print(f"  Scoring {paradigm_name} ({len(pairs)} pairs)...")

        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:i + batch_size]

            try:
                batch_scores = model.token_score(batch)
            except Exception as e:
                print(f"❌ Error in scoring batch at {i}: {e}")
                acc_per_file[paradigm_name] = 0.0
                file_results[paradigm_name] = pd.DataFrame()
                continue

            all_scores.extend(batch_scores)

        # Now compute per-pair probabilities
        results = []
        correct = 0

        for i, (g, u) in enumerate(pairs):
            idx_g = 2 * i
            idx_u = 2 * i + 1

            span_g = all_spans[idx_g]
            span_u = all_spans[idx_u]

            if span_g is None or span_u is None:
                results.append({
                    "sentence_gramm": g,
                    "sentence_ungramm": u,
                    f"probability_{checkpoint_name}_gramm": None,
                    f"probability_{checkpoint_name}_ungramm": None,
                    f"correct_{checkpoint_name}": None
                })
                continue

            tok_probs_g = [x[1] for x in all_scores[idx_g][span_g[0]:span_g[1]]]
            tok_probs_u = [x[1] for x in all_scores[idx_u][span_u[0]:span_u[1]]]

            prob_g = sum(tok_probs_g)
            prob_u = sum(tok_probs_u)

            is_correct = prob_g > prob_u
            correct += int(is_correct)

            results.append({
                "sentence_gramm": g,
                "sentence_ungramm": u,
                f"probability_{checkpoint_name}_gramm": prob_g,
                f"probability_{checkpoint_name}_ungramm": prob_u,
                f"correct_{checkpoint_name}": is_correct
            })

        acc = correct / len(pairs)
        acc_per_file[paradigm_name] = acc
        file_results[paradigm_name] = pd.DataFrame(results)

        print(f"✅ {paradigm_name}: {acc:.3f}")

    return file_results, acc_per_file



# --------------------------------------------------------
# MAIN LOOP OVER MODELS
# --------------------------------------------------------
for model in models:
    print(f"\n========== Evaluating model: {model} ==========")
    repo_id = f"fpadovani/{model}"
    repo_path, local_checkpoints = download_model_and_checkpoints(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    repo_subdir = repo_id.split('/')[-1]

    # ------------------ BLIMP ------------------
    all_results, all_accuracies = {}, {}
    state_path = os.path.join(output_dir_blimp, repo_subdir, "state", "all_state.pkl")
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                loaded = pickle.load(f)
                all_results = loaded.get('all_results', {})
                all_accuracies = loaded.get('all_accuracies', {})
        except Exception:
            pass
    from functools import partial
    # Using your provided evaluate_checkpoint_zorro_blimp_optimized function
    for ckpt_name, ckpt_path in local_checkpoints.items():
        if ckpt_name in all_accuracies:
            continue
        try:
            results, accuracies = evaluate_checkpoint_zorro_blimp_optimized(ckpt_name, ckpt_path, tokenizer, dataset_dir=blimp_dir, batch_size=64)
        except Exception as e:
            print(f"BLIMP evaluation failed for {ckpt_name}: {e!r}")
            accuracies = {'error': str(e)}
            results = {}
        all_results[ckpt_name] = results
        all_accuracies[ckpt_name] = accuracies
        save_checkpoint_csvs(ckpt_name, results, output_dir_blimp, repo_subdir)
        save_accuracy_summary_row(output_dir_blimp, repo_subdir, ckpt_name, accuracies)
        save_full_state(output_dir_blimp, repo_subdir, all_results, all_accuracies)

    # ------------------ ZORRO ------------------
    all_results, all_accuracies = {}, {}
    state_path = os.path.join(output_dir_zorro, repo_subdir, "state", "all_state.pkl")
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                loaded = pickle.load(f)
                all_results = loaded.get('all_results', {})
                all_accuracies = loaded.get('all_accuracies', {})
        except Exception:
            pass
    for ckpt_name, ckpt_path in local_checkpoints.items():
        if ckpt_name in all_accuracies:
            continue
        try:
            results, accuracies = evaluate_checkpoint_zorro_blimp_optimized(ckpt_name, ckpt_path, tokenizer, dataset_dir=zorro_dir, batch_size=64)
        except Exception as e:
            print(f"ZORRO evaluation failed for {ckpt_name}: {e!r}")
            accuracies = {'error': str(e)}
            results = {}
        all_results[ckpt_name] = results
        all_accuracies[ckpt_name] = accuracies
        save_checkpoint_csvs(ckpt_name, results, output_dir_zorro, repo_subdir)
        save_accuracy_summary_row(output_dir_zorro, repo_subdir, ckpt_name, accuracies)
        save_full_state(output_dir_zorro, repo_subdir, all_results, all_accuracies)

    # ------------------ FIT-CLAMS ------------------
    for key in fitclams_dirs.keys():
        dataset_dir = fitclams_dirs[key]
        output_dir_fit = output_dirs_fitclams[key]
        all_results, all_accuracies = {}, {}
        state_path = os.path.join(output_dir_fit, repo_subdir, "state", "all_state.pkl")
        if os.path.exists(state_path):
            try:
                with open(state_path, "rb") as f:
                    loaded = pickle.load(f)
                    all_results = loaded.get('all_results', {})
                    all_accuracies = loaded.get('all_accuracies', {})
            except Exception:
                pass
        for ckpt_name, ckpt_path in local_checkpoints.items():
            if ckpt_name in all_accuracies:
                continue
            try:
                results, accuracies = evaluate_checkpoint_fitclams_optimized(ckpt_name, ckpt_path, tokenizer, dataset_dir=dataset_dir)
            except Exception as e:
                print(f"FIT-CLAMS ({key}) evaluation failed for {ckpt_name}: {e!r}")
                accuracies = {'error': str(e)}
                results = {}
            all_results[ckpt_name] = results
            all_accuracies[ckpt_name] = accuracies
            save_checkpoint_csvs(ckpt_name, results, output_dir_fit, repo_subdir)
            save_accuracy_summary_row(output_dir_fit, repo_subdir, ckpt_name, accuracies)
            save_full_state(output_dir_fit, repo_subdir, all_results, all_accuracies)

    # ------------------ SEMANTIC MINIMAL PAIRS ------------------
    output_base_sem = "./evaluation/semantic_minimal_pairs/evaluation_scores/new_clean"
    os.makedirs(output_base_sem, exist_ok=True)

    # Determine the evaluation file based on model type
    if "cds" in model:
        eval_file = "./evaluation/semantic_minimal_pairs/data/verb_final_csv/cds_bin.csv"
        folder = 'cds'
    elif "candor" in model:
        eval_file = "./evaluation/semantic_minimal_pairs/data/verb_final_csv/candor_bin.csv"
        folder = 'candor'
    elif "bnc" in model:
        eval_file = "./evaluation/semantic_minimal_pairs/data/verb_final_csv/bnc_bin.csv"
        folder = 'bnc'
    elif "wiki" in model:
        eval_file = "./evaluation/semantic_minimal_pairs/data/verb_final_csv/wiki_bin.csv"
        folder = 'wiki'
    else:
        eval_file = None
        print(f"⚠️ No semantic minimal pairs dataset defined for model {model}, skipping.")
    
    # Evaluate if a file was selected
    if eval_file:
        out_csv = os.path.join(output_base_sem, folder, f"{model}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        evaluate_semantic_min_pairs_per_bin_resume(
            checkpoint_paths=local_checkpoints,
            eval_file_path=eval_file,
            output_csv_path=out_csv,
            tokenizer=tokenizer
        )


    import pathlib

    hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub")

    print(f"Cleaning up local snapshot and HF cache for {model} ...")

    # 1️⃣ Remove the snapshot folder from snapshot_download
    try:
        shutil.rmtree(repo_path)
        print(f"Deleted snapshot folder: {repo_path}")
    except Exception as e:
        print(f"Failed to delete snapshot folder {repo_path}: {e}")

    # 2️⃣ Remove the HF cache folder
    repo_cache_path = pathlib.Path(hf_cache_base) / f"models--{repo_id.replace('/', '--')}"
    if repo_cache_path.exists():
        try:
            shutil.rmtree(repo_cache_path)
            print(f"Deleted HF cache folder: {repo_cache_path}")
        except Exception as e:
            print(f"Failed to delete HF cache folder {repo_cache_path}: {e}")
    else:
        print(f"No HF cache folder found for {model} at {repo_cache_path}")

print("\n========== All models processed ==========")