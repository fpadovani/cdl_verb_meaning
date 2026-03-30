from transformers import TrainerCallback
import os
import shutil
import numpy as np
    

def log_validation_batch(eval_dataloader, tokenizer, device, log_file):
    """
    Logs validation batch content (decoded tokens) to a file.
    """
    with open(log_file, "a") as log:
        log.write("\n--- Validation Batch Start ---\n")
        
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Decode the input IDs back into tokens
            decoded_tokens = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            
            for i, tokens in enumerate(decoded_tokens):
                log.write(f"Sample {i}:\n{tokens}\n")
            
            log.write("--- Validation Batch End ---\n\n")
            break  # Log only the first batch for efficiency


class ValidationLoggingCallback(TrainerCallback):
    """
    Custom callback to log validation batches whenever validation is performed.
    """
    def __init__(self, log_file, tokenizer):
        self.log_file = log_file
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Triggered during evaluation to log validation batch content.
        """
        eval_dataloader = kwargs.get("eval_dataloader")
        if eval_dataloader:
            log_validation_batch(eval_dataloader, self.tokenizer, args.device, self.log_file)



class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step}: loss = {logs['loss']}")
    
class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, checkpoint_steps, tokenizer=None, push_to_hub=False, repo_id=None, hub_token=None):
        self.output_dir = output_dir
        self.checkpoint_steps = checkpoint_steps
        self.tokenizer = tokenizer  # store tokenizer reference
        self.push_to_hub = push_to_hub
        self.repo_id = repo_id
        self.hub_token = hub_token

    def on_step_end(self, args, state, control, **kwargs):
        """Save model only at predefined steps and optionally push to Hugging Face Hub."""
        if state.global_step in self.checkpoint_steps:
            checkpoint_path = os.path.join(self.output_dir, f"check-{state.global_step}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save locally
            kwargs['model'].save_pretrained(checkpoint_path)

            # Save tokenizer safely
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(checkpoint_path)
            else:
                print(f"⚠️ No tokenizer provided; skipping saving tokenizer at step {state.global_step}")

            print(f"Checkpoint saved at step {state.global_step}")

            # Push to Hugging Face Hub if enabled
            if self.push_to_hub and self.repo_id:
                print(f"Pushing checkpoint-{state.global_step} to the hub...")
                kwargs['model'].push_to_hub(
                    repo_id=self.repo_id,
                    commit_message=f"Checkpoint-{state.global_step}",
                    token=self.hub_token
                )
                if self.tokenizer is not None:
                    self.tokenizer.push_to_hub(
                        repo_id=self.repo_id,
                        commit_message=f"Checkpoint-{state.global_step}",
                        token=self.hub_token
                    )


def compute_log_checkpoints(total_training_steps, num_checkpoints=10):
    return np.unique(
        np.round(np.logspace(0, np.log10(total_training_steps), num=num_checkpoints)).astype(int)
    ).tolist()


