import numpy as np
import pandas as pd

from pymoo.core.problem import Problem

from baseline.cscf.decoder import Decoder
from baseline.cscf.utils.gowers import gower_matrix

import gower

from itertools import repeat
import copy

from multiprocessing.pool import ThreadPool
pool = ThreadPool(4)

class ProblemFactory(Problem):
    """
    Factory that generates a model with the environment and the various functions
    """

    def __init__(
        self,
        environment,
        decoder=None,
        chosen_objectives=None,
        max_sequence_length=5
    ):

        if chosen_objectives is None:
            chosen_objectives = [
                #"feature_tweaking_frequencies",
                "summed_costs",
                "gowers_distance"
            ]

        self.chosen_objectives = chosen_objectives

        self.env = environment
        self.init_observation = self.env.memory.copy()
        self.x0 = self.env.parse_observation(self.env.memory) # Environment

        filtered_actions = {k:v for k,v in self.env.programs_library.items() if v["level"] == 0 or v["level"] == -1}
        self.available_actions = {idx:(v,filtered_actions[v]) for idx,v in enumerate(filtered_actions)}

        self.n_actions = len(self.available_actions) # Remove the level 1 program
        #self.max_sequence_length = max_sequence_length
        self.max_sequence_length = len(self.available_actions)

        assert self.max_sequence_length == self.n_actions, print(self.max_sequence_length, self.n_actions)

        self.invalid_costs = np.inf

        self.xxl = np.zeros(self.n_actions, dtype=np.float64)
        self.xxu = np.zeros(self.n_actions, dtype=np.float64)

        # Each action has inidiviudal costs in our formulation
        # Additionally, we add gowers_distance
        # and the sequence length
        n_objectives = 0
        if "summed_costs" in self.chosen_objectives:
            n_objectives += 1
        if "feature_tweaking_frequencies" in self.chosen_objectives:
            n_objectives += len(self.x0)
        if "summed_costs_discounted" in self.chosen_objectives:
            n_objectives += 1
        if "gowers_distance" in self.chosen_objectives:
            n_objectives += 1

        n_constraints = 2

        super().__init__(
            # Twice since we have the sequence and numerical part of equal lengths
            n_var=self.max_sequence_length * 2,
            n_obj=n_objectives,
            n_constr=n_constraints,
            elementwise_evaluation=False,
            # General algorithm bounds are [0,1] for each variable since
            # we use BRKGA. The decoder will translate it for us to the actual space.
            xl=np.zeros(self.n_actions * 2),
            xu=np.ones(self.n_actions * 2),
        )

        # For translation of the bounds
        if decoder is None:
            self.decoder = Decoder(self)
        else:
            self.decoder = decoder(self)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the solution by decoding the genotype
        :param x:
        :param out:
        :param args:
        :param kwargs:
        :return:
        """

        decoded_x = np.array(
            [self.decoder.decode(instance) for instance in x], dtype=np.int64
        )

        assert len(decoded_x) == len(x)
        #assert decoded_x.dtype == np.float64

        # Compute sequences and tweaking_values and check if they are valid
        #valid_seqs = np.array([self.check_valid_seq(sol) for sol in decoded_x])
        #

        sequences = [
            self.create_sequence(sol) for i, sol in enumerate(decoded_x)
        ]
        assert len(sequences) == len(decoded_x)

        full_sequence = [
            len(s) > 0 for s in sequences
        ]

        def _find_stop(s):
            for k, (p, a) in enumerate(s):
                if p=="STOP" and k == len(s)-1:
                    return True
            return False

        # Check if the last element is a stop action and filter those sequences
        # which does not compute this
        valid_seqs = [
            _find_stop(s) if full_sequence[k] else False for k, s in enumerate(sequences)
        ]
        assert len(valid_seqs) == len(decoded_x)
        sequences = [sequences[i] if valid_seqs[i] else None for i in range(len(sequences))]

        assert len(sequences) == len(decoded_x)

        #tweaking_values = [
        #    self.get_tweaking_values(sol) if valid_seqs[i] else None
        #    for i, sol in enumerate(decoded_x)
        #]
        #assert len(tweaking_values) == len(decoded_x)

        # Compute objectives
        seq_lengths = [
            self.seq_length(sol) if valid_seqs[i] else self.invalid_costs
            for i, sol in enumerate(decoded_x)
        ]
        assert len(seq_lengths) == len(decoded_x)

        costs, rewards, tweaked_instances = self.get_costs(sequences)

        #(
        #    tweaked_instances,
        #    costs,
        #    discounts,
        #    penalties,
        #) = self.get_individual_costs_and_tweaked_instance(tweaking_values, sequences)

        #assert costs.shape == discounts.shape
        #assert costs.shape[1] == self.n_actions, costs.shape[1]

        # ! round costs to keep sanity
        # ! Does not really interfere with the comparison (4th digit after comma)
        #decimal_places = 4
        #costs = np.around(costs, decimal_places)
        #assert costs.dtype == np.float64, costs.dtype

        assert len(costs) == len(decoded_x)
        assert len(tweaked_instances) == len(decoded_x)
        #assert len(penalties) == len(decoded_x)

        costs = [float(sum(x)) for x in costs]

        # build fitness vector
        fitness_vec = []
        if "summed_costs" in self.chosen_objectives or "summed_costs_discounted" in self.chosen_objectives:
            #fitness_vec.append(costs.sum(axis=1))
            fitness_vec.append(costs)
        #if "summed_costs_discounted" in self.chosen_objectives:
            #dc = costs * discounts
            #assert dc.shape == costs.shape, (dc.shape, costs.shape)
        #    fitness_vec.append(dc.sum(axis=1))
        if "feature_tweaking_frequencies" in self.chosen_objectives:
            fitness_vec.append(
                self.get_feature_tweaking_frequencies(tweaking_values, sequences)
            )
        if "gowers_distance" in self.chosen_objectives:
            #distances = self.get_gowers_distance(tweaked_instances, valid_seqs)
            distances = gower.gower_matrix(tweaked_instances)
            #print(distances.shape)
            distances = np.sum(distances, axis=0)
            distances = distances.flatten()
            #print(len(distances))
            #print(len(distances), len(distances[0]), len(tweaked_instances), len(fitness_vec[0]))
            fitness_vec.append(distances)


        fitness = np.column_stack(fitness_vec)
        assert fitness.dtype == np.float64, fitness.dtype

        # Check if we successfully predict the correct value
        #g1 = self.check_target_class_condition(tweaked_instances, valid_seqs)

        g1 = rewards
        g2 = [1.0 if not v else 0.0 for v in valid_seqs]
        #g3 = penalties

        #penalty = np.column_stack([g1, g2, g3])
        penalty = np.column_stack([g1, g2])
        assert penalty.dtype == np.float64, penalty.dtype

        assert len(penalty) == len(fitness)
        assert len(fitness) == len(decoded_x)

        assert fitness.shape[1] == self.n_obj, (fitness.shape[1], self.n_obj)
        assert penalty.shape[1] == self.n_constr, penalty.shape[1]

        out["F"] = fitness.astype(float)
        out["G"] = penalty
        out["pheno"] = decoded_x

        hashs = np.array(
            [
                hash(
                    str(
                        sorted(
                            [
                                self.available_actions[i][1]["index"] for i in dx[: self.max_sequence_length] if i != -1
                            ]
                        )
                    )
                )
                for dx in decoded_x
            ]
        )
        assert len(hashs) == len(decoded_x)
        out["hash"] = hashs
        #out["tweaked_instances"] = tweaked_instances
        out["action_costs"] = costs
        #out["action_discounts"] = discounts
        # always additionally log these two values
        #out["summed_costs"] = costs.sum(axis=1)
        #out["summed_costs_discounted"] = costs.sum(axis=1)
        out["summed_costs"] = costs
        out["summed_costs_discounted"] = costs
        #out["summed_costs_discounted"] = (costs * discounts).sum(axis=1)

    def get_costs(self, sequences):

        def parallelize(x, env, init_observation):

            tweaked_instances = []
            cost_single = []

            if x is None:
                cost_single = [1000]
                reward = 100
                tweaked_instances.append([env.get_observation().tolist()[:-1].copy()])

            else:

                env.memory = init_observation.copy()

                precondition_satisfied = True

                tweaked_instance = []

                for p, a in x:

                    tweaked_instance.append(env.get_observation().tolist()[:-1].copy())

                    if not precondition_satisfied:
                        cost_single.append(100)
                        continue

                    pindex = env.programs_library[p].get("index")
                    aindex = env.complete_arguments.index(a)

                    # check preconditions
                    if not env.prog_to_precondition.get(p)(a):
                        precondition_satisfied = False

                    if precondition_satisfied:
                        cost_single.append(env.get_cost(pindex, aindex))
                        _ = env.act(p, a)
                    else:
                        cost_single.append(100)

                evaluation_result = env.prog_to_postcondition.get("INTERVENE")(None, None)

                tweaked_instances.append(tweaked_instance)

                # If the preconditions are not satisfied, we place everything positive to violate
                # the constraint
                if not precondition_satisfied:
                    cost_single = [10 * x for x in cost_single]
                    # We invert here since we need the constraint to be less than zero
                    reward = 100
                else:
                    evaluation_result = evaluation_result and not (env.memory==init_observation.copy()) and len(x) > 1
                    if not evaluation_result:
                        cost_single = [10 * x for x in cost_single]
                    reward = -1 if evaluation_result else 100

            return cost_single, reward, tweaked_instances

        copied_envs = [copy.deepcopy(self.env) for _ in sequences]
        M = pool.starmap(parallelize, zip(sequences, copied_envs, repeat(self.init_observation.copy())))

        costs = [x[0] for x in M]
        rewards = [x[1] for x in M]
        tweaked_instances = [x[2] for x in M]

        return costs, rewards, pd.DataFrame(tweaked_instances)

    def get_individual_costs_and_tweaked_instance(self, tweaking_values, sequences):
        n_sols = len(tweaking_values)
        costs = np.zeros((n_sols, self.n_actions), dtype=float)
        discounts = np.ones((n_sols, self.n_actions), dtype=float)
        penalties = np.zeros(n_sols, dtype=float)
        tweaked_instances = []
        for i, tweak in enumerate(tweaking_values):
            if tweak is None:
                tweaked_instances.append(np.full(len(self.x0), np.nan))
                costs[i, :] = np.full(self.n_actions, self.invalid_costs)
            else:
                (
                    tweaked_instance,
                    sequence_costs,
                    sequence_discounts,
                    penalty,
                ) = sequences[i].unroll_actions_individual_costs(
                    self.x0.copy(),
                    tweak,
                    self.n_actions,
                )
                assert penalty.ndim == 1
                assert sequence_costs.ndim == 1
                assert len(tweaked_instance) == len(self.x0)
                tweaked_instances.append(tweaked_instance)
                costs[i, :] = sequence_costs
                discounts[i, :] = sequence_discounts
                penalties[i] += penalty.sum()
        return (
            np.array(tweaked_instances),
            costs,
            discounts,
            penalties,
        )

    def check_valid_seq(self, x):
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]

        if all(action_order_part == -1):
            return False
        return True

    def check_target_class_condition(self, x, valid):
        _x = x[valid]
        preds = self.blackbox(np.array(_x))
        output = np.full(len(x), 1.0)
        correct = np.ones(len(_x))
        correct[preds == self.target_class] = 0.0
        output[valid] = correct
        return output

    def get_feature_tweaking_frequencies(self, tweaking_values, sequences):
        n_sols = len(tweaking_values)
        res = np.zeros((n_sols, len(self.x0)), dtype=float)
        for i, tweak in enumerate(tweaking_values):
            if tweak is None:
                res[i, :] = np.full(len(self.x0), self.invalid_costs)
            else:
                tweaked_instances = sequences[i].get_tweaked_instance_after_each_action(
                    self.x0.copy(),
                    tweak,
                )
                assert len(tweaked_instances) == len(tweak)
                old_instance = self.x0.copy()
                for new_instance in tweaked_instances:
                    res[i] += self.get_feature_changes(old_instance, new_instance)
                    old_instance = new_instance
        return res

    def get_feature_changes(self, old_instance, new_instance):
        assert old_instance.ndim == 1
        assert new_instance.ndim == 1
        assert len(old_instance) == len(new_instance)
        res = np.zeros(len(old_instance))
        diffs = new_instance != old_instance
        res[diffs] = 1.0
        return res

    def _wrap_action(self, actions, args):

        act_arg_pair = []

        for p, a in zip(actions, args):

            p_name = self.available_actions.get(p)[0]
            a_values = self.env.arguments.get(self.env.programs_library.get(p_name).get("args"))

            a_val = a_values[a]

            act_arg_pair.append((p_name, a_val))

        return act_arg_pair


    def create_sequence(self, x, prin=False):

        """Create sequence of correct action pairs"""

        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]
        active = action_order_part[action_order_part != -1]
        args_active = []

        for k, v in enumerate(x[split_point:]):
            if x[k] != -1:
                args_active.append(v)

        #print("X: ", x)
        #print("CHOOSEN: ", active, args_active)

        #actions_params = [copy.copy(self._wrap_action(i, )) for i in active]
        actions_params = self._wrap_action(active, args_active)

        actions = [params for params in actions_params]
        return actions

    def get_tweaking_values(self, x) -> np.array:
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point].astype(int)
        value_part = x[split_point:]
        idxs = action_order_part[action_order_part != -1]
        tweaking_values = value_part[idxs].copy()
        return tweaking_values

    def seq_length(self, x) -> int:
        assert x.ndim == 1, x.ndim
        split_point = self.max_sequence_length
        action_order_part = x[:split_point]
        return len(action_order_part[action_order_part != -1])

    def get_gowers_distance(self, x, valid) -> float:
        _x = x.copy()[valid]
        original_cat_idx = self.env.categorical_features

        gowers = gower_matrix(
            _x,
            self.x0.reshape(1, -1),
            cat_features=original_cat_idx,
            norm=self.num_max,
            norm_ranges=self.num_ranges,
        )
        assert gowers.dtype == np.float64 or gowers.dtype == np.float32, gowers.dtype
        assert len(gowers.flatten()) == len(_x)
        gowers = gowers.flatten()
        # round to avoid being too sensitive
        gowers = np.around(gowers, 2)
        assert gowers.ndim == 1, gowers.shape
        res = np.full(len(x), 1.0, dtype=np.float64)
        res[valid] = gowers
        return res
