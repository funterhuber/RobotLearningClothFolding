import wandb
import pandas as pd
 
api = wandb.Api()
 
# Update these to match your wandb entity/project/run
entity = None  # your wandb username, or None if default
project = "multitask-dit-experiments"
run_id = "run2_larger_steps_h49_a16_1"
 
if entity:
    run = api.run(f"{entity}/{project}/{run_id}")
else:
    run = api.run(f"{project}/{run_id}")
 
# Download full history
history = run.history()
history.to_csv("run1_history.csv", index=False)
print(f"Downloaded {len(history)} rows to run1_history.csv")
print(history.columns.tolist())
