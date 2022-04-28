"""
Perform a study on the accuracy of the deterministic program
by varying the sampling.
"""

import os
import yaml
from tqdm import tqdm

sampling = [100,200,300,400,500,600,700]

if __name__ == "__main__":

    config = yaml.load(open("analytics/config.yml", "r"),Loader=yaml.FullLoader)
    filepath = "./program_results.csv"
    base_dir = "./program_size"

    os.system(f"echo method,dataset,correct,wrong,mean_cost,std_cost,mean_length,std_length > {filepath}")

    for m, c, d  in zip(config.get("experiments").get("models"), config.get("experiments").get("configs"), config.get("experiments").get("dataset")):
        print(f"[*] Training {m} {c}")

        # Train the various agents
        for k in tqdm(sampling):
            save_path = os.path.join(base_dir, f"{d}_{k}.pth")
            os.system(
                f"mpirun -n 4 python -m generalized_alphanpi.core.automa {m} INTERVENE --config {c} --max-tries {k} --single-core --tree --save-automa --automa-model-path {save_path}")

        # Evaluate the various agents
        for k in tqdm(sampling):
            save_path = os.path.join(base_dir, f"{d}_{k}.pth")
            os.system(
                    f"python -m generalized_alphanpi.core.validate_static {save_path} INTERVENE --config {c} --tree --to-stdout --single-core --save >>  {filepath}")