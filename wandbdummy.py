## uv add wandb

import wandb

wandb.login()  # only needed once per environment, or use env var WANDB_API_KEY

run = wandb.init(project="mlx-hackernews", job_type="upload")

artifact = wandb.Artifact(name="ABHNModel", type="model")
artifact.add_file("ABHNModel.pth")

run.log_artifact(artifact)
run.finish()