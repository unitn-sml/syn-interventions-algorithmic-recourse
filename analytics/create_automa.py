import os
import yaml

if __name__ == "__main__":


    config = yaml.load(open("analytics/config.yml", "r"),Loader=yaml.FullLoader)

    for m,c,p in zip(config.get("experiments").get("models"), config.get("experiments").get("configs"), config.get("experiments").get("programs")):
        print(f"[*] Creating automa for {m} {c}")
        os.system(f"mpirun -n 4 python -m generalized_alphanpi.core.automa {m} INTERVENE --config {c} --max-tries 250 --tree --save-automa --automa-model-path {p} --single-core")