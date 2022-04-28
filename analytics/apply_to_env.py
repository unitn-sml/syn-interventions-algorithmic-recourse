import yaml
from argparse import ArgumentParser

from generalized_alphanpi.utils import import_dyn_class

import random
import numpy as np
import pandas as pd

import os
from os import listdir
from os.path import isfile, join

seed = 2021
random.seed(seed)
np.random.seed(seed)

def get_unique_ids(files):
    unique_ids = []
    for trace_file in files:
        tmp = pd.read_csv(trace_file)["id"].unique()
        if len(unique_ids) > 0:
            unique_ids = list(set(tmp) & set(unique_ids))
        else:
            unique_ids = list(tmp)
    return unique_ids

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the correct traces")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--std-out", default=False, action="store_true", help="Print to stdout directly")

    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # Compute all the files present
    onlyfiles = [join(args.path,f) for f in listdir(args.path) if isfile(join(args.path, f))]

    # Compute the unique ids present
    unique_ids = get_unique_ids(onlyfiles)

    # Contains all the results
    final_results = []

    for trace_file in onlyfiles:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {}),
            **config.get("validation").get("environment").get("configuration_parameters", {})
        )

        method = os.path.basename(trace_file).split("-")[1]
        dataset = config.get("validation").get("dataset_name")

        env.validation = True

        # Read traces.
        traces = pd.read_csv(trace_file)

        if "rule" in traces.columns:
            traces.drop(columns=["rule"], inplace=True)

        # Filter by unique ids
        #traces = traces[traces.id.isin(unique_ids)]

        lengths = []
        correct = []
        costs = []

        # Specify which env we are looking at
        id_env = 0

        print(traces.id.unique())

        for id in traces.id.unique():

            cost = 0

            # Skip env we are not able to solve
            while id_env != id:
                env.start_task(env.prog_to_idx["INTERVENE"])
                env.end_task()
                id_env += 1

            traces_list = traces[traces.id == id].values.tolist()

            precondition_satisfied = True

            env.start_task(env.prog_to_idx["INTERVENE"])

            lengths.append((id,len(traces_list)))

            for k, (_, p, a) in enumerate(traces_list):

                pindex = env.programs_library[p].get("index")

                if a.isnumeric():
                    a = int(a)

                aindex = env.complete_arguments.index(a)

                # check preconditions
                if not env.prog_to_precondition.get(p)(a):
                    precondition_satisfied = False

                if not precondition_satisfied:
                    print(method, id, p, a, env.memory)

                if precondition_satisfied:
                    if not args.std_out:
                        print(f"{k} - Apply: ", p, a, env.get_cost(pindex, aindex), env.memory)
                    cost += env.get_cost(pindex, aindex)
                    _ = env.act(p, a)
                else:
                    print()
                    break

            evaluation_result = env.prog_to_postcondition.get("INTERVENE")(None, None) and precondition_satisfied

            if len(traces_list) > 0:
                correct.append((id, evaluation_result))

            if evaluation_result:
                costs.append((id, cost))

            if not args.std_out:
                print("Final:", evaluation_result, env.memory)
                print()

            env.end_task()

            id_env+= 1

        accuracy = np.sum([1 if x[1] else 0 for x in correct])/config.get("validation").get("iterations")

        costs = [x[1] for x in costs if x[0] in unique_ids]
        lengths = [x[1] for x in lengths if x[0] in unique_ids]

        final_results.append([
            method,
            dataset,
            accuracy,
            np.sum(costs) / len(costs),
            np.std(costs),
            np.sum(lengths) / len(lengths),
            np.std(lengths)
        ])

        #len(costs), np.sum(lengths) / len(lengths), np.sum(costs) / len(costs), len(traces.id.unique()) - np.sum(
                #correct)

    for d in final_results:
        print(",".join([str(x) for x in d]))

    if args.std_out:
        print()
