
import os
import yaml

if __name__ == "__main__":


    config = yaml.load(open("analytics/config.yml", "r"),Loader=yaml.FullLoader)
    filepath = "results.csv"

    os.system(f"echo method,dataset,correct,wrong,mean_cost,std_cost,mean_length,std_length > {filepath}")

    for m, c in zip(config.get("experiments").get("programs"), config.get("experiments").get("configs")):
        print(f"[*] Running {m} {c}")
        os.system(
            f"python -m generalized_alphanpi.core.validate_static {m} INTERVENE --config {c} --tree --to-stdout --single-core --save >>  {filepath}")

    for m,c in zip(config.get("experiments").get("models"), config.get("experiments").get("configs")):
        print(f"[*] Running {m} {c}")
        os.system(f"python -m generalized_alphanpi.core.validate {m} INTERVENE --config {c} --to-stdout --save >> {filepath}")

    for m,c in zip(config.get("experiments").get("models"), config.get("experiments").get("configs")):
        print(f"[*] Running agent only {m} {c}")
        os.system(f"python -m generalized_alphanpi.core.validate_agent {m} INTERVENE --config {c} --to-stdout --save >>  {filepath}")

    for m,c in zip(config.get("experiments").get("programs"), config.get("experiments").get("configs")):
        print(f"[*] Running competitor {m} {c}")
        os.system(f"python baseline/cscf/genetic.py --config {c} --std-out >>  {filepath}")
