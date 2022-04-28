
import os
import yaml

if __name__ == "__main__":

    tr = [
        "analytics/evaluation/traces/synthetic_long",
        "analytics/evaluation/traces/german",
        "analytics/evaluation/traces/synthetic",
        "analytics/evaluation/traces/adult"
    ]

    config = yaml.load(open("analytics/config.yml", "r"),Loader=yaml.FullLoader)
    filepath = "analytics/evaluation/results_intersection.csv"

    os.system(f"echo method,dataset,correct,mean_cost,std_cost,mean_length,std_length > {filepath}")

    for t, c in zip(tr, config.get("experiments").get("configs")):
        print(f"[*] Running {t} {c}")
        os.system(
            f"python analytics/apply_to_env.py {t} --config {c} >>  {filepath}")
