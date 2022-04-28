import yaml
from argparse import ArgumentParser

from generalized_alphanpi.utils import import_dyn_class

from baseline.cscf.CSCF import CSCF
from baseline.cscf.problem_factory import ProblemFactory, pool

from pymoo.optimize import minimize
from pymoo.util.display import MultiObjectiveDisplay

import random
import numpy as np

import os
import time

import pandas as pd

seed = 2021
random.seed(seed)
np.random.seed(seed)

def optimization_cfg():
    # For the EA
    total_pop = 200
    n_elites_frac = 0.2
    n_elites = 1  # doesn't make a difference since we use NDS and not crowding distance
    offsprings_frac = 0.8
    n_offspring = int(offsprings_frac * total_pop)
    mutants_frac = 0.2
    n_mutants = int(mutants_frac * total_pop)
    bias = 0.7
    eliminate_duplicates = True

    n_generations = 50

    return n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, n_generations


def setup_optimizer(n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, seed):
    algorithm = CSCF(
        n_elites=n_elites,
        n_offsprings=n_offspring,
        n_mutants=n_mutants,
        bias=bias,
        eliminate_duplicates=eliminate_duplicates,
    )
    return algorithm

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--std-out", default=False, action="store_true", help="Print to stdout directly")

    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    env = import_dyn_class(config.get("environment").get("name"))(
        **config.get("environment").get("configuration_parameters", {}),
        **config.get("validation").get("environment").get("configuration_parameters", {})
    )

    method = "cscf"
    dataset = config.get("validation").get("dataset_name")

    env.validation = True

    save_optimal_population_trace = False # This requires a lot of memory

    rewards_total = []
    length = []
    costs_total = []

    best_sequences = []

    total_evaluations = []

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))
    for userid in range(0, iterations):
        if not args.std_out:
            print(f"Running experiment {userid+1}")

        env.start_task(1)

        problem = ProblemFactory(env)

        n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, n_generations = optimization_cfg()
        algorithm = setup_optimizer(n_elites, n_offspring, n_mutants, bias, eliminate_duplicates, seed)

        termination = ("n_gen", n_generations)

        # perform a copy of the algorithm to ensure reproducibility
        import copy
        obj = copy.deepcopy(algorithm)

        # let the algorithm know what problem we are intending to solve and provide other attributes
        obj.setup(problem, termination=termination,
                  seed=seed,
                  display=MultiObjectiveDisplay(),
                  save_history=save_optimal_population_trace,
                  verbose=not args.std_out,
                  return_least_infeasible=False)

        # until the termination criterion has not been met
        evals = 0
        while obj.has_next():
            # perform an iteration of the algorithm
            obj.next()
            evals += obj.evaluator.n_eval

        res = obj.result()

        #res = minimize(
        #    problem,
        #    algorithm,
        #    termination,
        #    seed=seed,
        #    display=MultiObjectiveDisplay(),
        #    save_history=save_optimal_population_trace,
        #    verbose=not args.std_out,
        #    return_least_infeasible=False
        #)

        if res.X is not None:

            res_s = np.array([problem.decoder.decode(instance) for instance in res.X], dtype=np.int64)

            sequences = [
                problem.create_sequence(sol) for i, sol in enumerate(res_s)
            ]

            costs = []
            rewards = []
            for C, F, s in zip(res.G, res.F, sequences):
                rewards.append(C[0])
                costs.append(F[0])

            # Invert the cost and rewards to make it similar to the other experiments
            rewards = [1 if r == -1 else np.inf for r in rewards]

            # Multiply the reward with the cost
            costs = np.multiply(costs, rewards)

            # For each sequence, pick the least expensive one
            c_r_seq = list(zip(costs, rewards, sequences))
            c_r_seq.sort(key=lambda x: x[0])

            # Get the first sequence and save its costs
            cost_b, reward_b, sequence_b = c_r_seq[0]
            rewards_total.append(1 if reward_b == 1 else 0)
            length.append(len(sequence_b)) # Add one for the stop action
            costs_total.append(cost_b)
            total_evaluations.append(evals)

            best_sequences.append((userid, sequence_b))

        else:
            rewards_total.append(0)

        env.end_task()

    # Close multiprocessing pool
    pool.close()

    rewards_total = rewards_total if rewards_total else [0]
    length = length if length else [0]
    costs_total = costs_total if costs_total else [0]
    total_evaluations = total_evaluations if total_evaluations else [0]

    complete = f"{method},{dataset},{np.mean(rewards_total)},{1-np.mean(rewards_total)},{np.mean(costs_total)},{np.std(costs_total)},{np.mean(length)},{np.std(length)},{np.mean(total_evaluations)},{np.std(total_evaluations)}"

    ts = time.localtime(time.time())
    date_time = '-{}-{}_{}_{}-{}_{}.csv'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

    results_filename = config.get("validation").get("save_results_name") + date_time
    if not args.std_out:
        results_file = open(
            os.path.join(config.get("validation").get("save_results"), f"cscf-{dataset}-{results_filename}"), "w"
        )

    # Print to stdout if needed
    if args.std_out:
        print(complete)

    if not args.std_out:
        results_file.write(f"method,dataset,correct,wrong,mean_cost,std_cost,mean_length,std_length,mean_evaluations,std_evaluations\n")
        results_file.write(complete + '\n')
        results_file.close()

    # Save sequences to file
    df_sequences = []
    for k, x in best_sequences:
        for p, a in x:
            df_sequences.append([k, p, a])

    # Create a dataframe and save sequences to disk
    if df_sequences:
        best_sequences = pd.DataFrame(df_sequences, columns=["id", "program", "arguments"])
        best_sequences.to_csv(
            os.path.join(config.get("validation").get("save_results"), f"traces-cscf-{dataset}-{results_filename}"),
            index=None)