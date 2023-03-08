from classifiers.train_model import Net
from data.synthetic.generate import get_loan

from rl_mcts.core.environment import Environment

from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

def get_loan(f):

    job, education, income = f.get("job"), f.get("education"), f.get("income")
    housing = f.get("house")

    if education == "triennale" or education == "magistrale" or education == "phd":
        if job == "impiegato" or job == "private":
            if income >= 50000:
                if housing == "own":
                    return "good"
        elif job == "manager" or job == "ceo":
            if income >= 80000:
                return "good"

    return "bad"

class SynEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim=20):
        super(SynEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, encoding_dim)
        self.l2 = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x

class SynEnvironmentDeter(Environment):

    def __init__(self, features, weights, encoder, scaler):
        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'CHANGE_EDUCATION': self._change_education,
                                                'CHANGE_JOB': self._change_occupation,
                                                'CHANGE_INCOME': self._change_income,
                                                'CHANGE_HOUSE': self._change_house,
                                                'CHANGE_RELATION': self._change_relation
                                                }.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        'CHANGE_EDUCATION': self._change_education_p,
                                                        'CHANGE_JOB': self._change_occupation_p,
                                                        'CHANGE_INCOME': self._change_income_p,
                                                        'CHANGE_HOUSE': self._change_house_p,
                                                        'CHANGE_RELATION': self._change_relation_p,
                                                        'INTERVENE': self._placeholder_stop}.items()))

        self.prog_to_postcondition = OrderedDict(sorted({'INTERVENE': self._intervene_postcondition}.items()))

        self.programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'args': 'NONE'},
                                                    'CHANGE_EDUCATION': {'level': 0, 'args': 'EDU'},
                                                    'CHANGE_JOB': {'level': 0, 'args': 'WORK'},
                                                    'CHANGE_INCOME': {'level': 0, 'args': 'INT'},
                                                    'CHANGE_HOUSE': {'level': 0, 'args': 'HOUS'},
                                                    'CHANGE_RELATION': {'level': 0, 'args': 'RELATION'},
                                                    'INTERVENE': {'level': 1, 'args': 'NONE'}}.items()))

        self.arguments = OrderedDict(sorted({
                                                "WORK": ["disoccupato", "operaio", "privato", "impiegato", "manager", "ceo"],
                                                "EDU": ["nessuno", "diploma", "triennale", "magistrale", "phd"],
                                                "RELATION": ["single", "sposato/a", "divorziato/a", "vedovo/a"],
                                                "HOUS": ["none", "rent", "own"],
                                                "INT": [5000, 10000, 20000, 30000, 40000, 50000],
                                                "NONE": [0]
                                            }.items()))

        self.prog_to_cost = OrderedDict(sorted({'STOP': self._stop_cost,
                                                'CHANGE_EDUCATION': self._change_education_cost,
                                                'CHANGE_JOB': self._change_occupation_cost,
                                                'CHANGE_INCOME': self._change_income_cost,
                                                'CHANGE_HOUSE': self._change_house_cost,
                                                'CHANGE_RELATION': self._change_relation_cost
                                                }.items()))

        self.previous_classification = 0
        self.numerical_cols = ["age", "credit", "income"]
        self.categorical_cols = ["education", "job", "house", "sex", "relationship", "country"]
        self.boolean_cols = [
                                  "30 <= age < 40",
                                  "40 <= age < 50",
                                  "50 <= age < 60",
                                  "age >= 60",
                                  "0 <= credit < 1000",
                                  "1000 <= credit < 10000",
                                  "10000 <= credit < 30000",
                                  "30000 <= credit < 50000",
                                  "credit >= 50000",
                                  "0 <= income < 10000",
                                  "10000 <= income < 30000",
                                  "30000 <= income < 50000",
                                  "50000 <= income < 70000",
                                  "70000 <= income < 100000",
                                  "100000 <= income < 150000",
                                  "150000 <= income < 200000",
                                  "income >= 200000"
                              ]
        self.parsed_columns = self.boolean_cols + self.categorical_cols

        self.setup_system(self.boolean_cols, self.categorical_cols, encoder, scaler,
                      None, None, net_layers=5, net_size=108)

        self.max_depth_dict = {1: 7}

        super().__init__(features, weights, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_depth_dict,
                         complete_arguments=self.complete_arguments, prog_to_cost=self.prog_to_cost,
                         custom_tensorboard_metrics=self.custom_tensorboard_metrics)

    def get_state_str(self, state):
        result = get_loan(state)
        return state, result

    def _placeholder_stop(self, args=None):
        return True

    def _get_boolean_conditions(self, data):

        return [
            30 <= data.get("age") < 40,
            40 <= data.get("age") < 50,
            50 <= data.get("age") < 60,
            data.get("age") >= 60,
            0 <= int(data.get("credit")) < 1000,
            1000 <= data.get("credit") < 10000,
            10000 <= data.get("credit") < 30000,
            30000 <= data.get("credit") < 50000,
            data.get("credit") >= 50000,
            0 <= int(data.get("income")) < 10000,
            10000 <= int(data.get("income")) < 30000,
            30000 <= int(data.get("income")) < 50000,
            50000 <= int(data.get("income")) < 70000,
            70000 <= int(data.get("income")) < 100000,
            100000 <= int(data.get("income")) < 150000,
            150000 <= int(data.get("income")) < 200000,
            data.get("income") >= 200000
        ]

    def parse_observation(self, data):
        """
        Parse an environment, by keeping the categorical values, but by removing
        the numerical values in favour of booleans conversions
        :param data:
        :return:
        """

        booleans = self._get_boolean_conditions(data)

        data = pd.DataFrame.from_records([data])
        data.drop(columns=self.numerical_cols, inplace=True)

        return booleans + data.values[0].tolist()

    def preprocess_single(self, data):

        data.reset_index(drop=True, inplace=True)
        cat_ohe = self.data_encoder.transform(data[self.categorical_cols]).toarray()
        ohe_df = pd.DataFrame(cat_ohe, columns=self.data_encoder.get_feature_names_out(input_features=self.categorical_cols))
        data.reset_index(drop=True, inplace=True)
        data = pd.concat([data, ohe_df], axis=1).drop(columns=self.categorical_cols, axis=1)

        return torch.FloatTensor(data.values[0]), data

    def transform_user(self):
        data_df = pd.DataFrame.from_records([self.features])
        return self.preprocess_single(data_df)

    def init_env(self):

        result = get_loan(self.features)
        classification = 1 if result == "bad" else 0

        self.previous_classification = classification

    def reset_env(self, task_index):

        result = get_loan(self.features)
        classification = 1 if result == "bad" else 0

        self.previous_classification = classification
        self.has_been_reset = True

        return 0, 0

    def reset_to_state(self, state):
        self.features = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    def _stop(self, arguments=None):
        return True

    def _change_education(self, arguments=None):
        self.features["education"] = arguments

    def _change_occupation(self, arguments=None):
        self.features["job"] = arguments

    def _change_income(self, arguments=None):
        self.features["income"] += arguments

    def _change_house(self, arguments=None):
        self.features["house"] = arguments

    def _change_relation(self, arguments=None):
        self.features["relationship"] = arguments

    def _change_income_p(self, arguments=None):
        return self.features["income"] > 0

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) > self.arguments["EDU"].index(self.features.get("education"))

    def _change_occupation_p(self, arguments=None):
        return self.arguments["WORK"].index(arguments) > self.arguments["WORK"].index(self.features.get("job"))

    def _change_house_p(self, arguments=None):
        return self.arguments["HOUS"].index(arguments) > self.arguments["HOUS"].index(self.features.get("house"))

    def _change_relation_p(self, arguments=None):

        if self.features.get("relationship") in ["single", "divorziato/a"]:
            if self.features.get("relationship") == "single":
                return arguments != "vedovo/a" and arguments != "divorziato/a"
            else:
                return arguments != "vedovo/a"

        return self.arguments["RELATION"].index(arguments) > self.arguments["RELATION"].index(self.features.get("relationship"))

    def _rescale_by_relation(self):
        if self.features.get("relationship") == "sposato/a":
            return 0.5
        elif self.features.get("relationship") == "divorziato/a":
            return 0.8
        else:
            return 1

    def _rescale_by_edu(self):
        if self.features.get("education") == "phd":
            return 0.3
        elif self.features.get("education") == "magistrale":
            return 0.4
        elif self.features.get("education") == "triennale":
            return 0.5
        elif self.features.get("education") == "diploma":
            return 0.7
        else:
            return 1

    def _rescale_by_job(self):
        if self.features.get("job") == "ceo":
            return 0.4
        if self.features.get("job") == "manager":
            return 0.5
        elif self.features.get("job") == "privato":
            return 0.6
        elif self.features.get("job") == "impiegato":
            return 0.7
        elif self.features.get("job") == "operaio":
            return 0.8
        else:
            return 1

    def _rescale_by_income(self):
        if self.features.get("income") > 130000:
            return 0.1
        elif self.features.get("income") > 100000:
            return 0.2
        elif self.features.get("income") > 70000:
            return 0.3
        elif self.features.get("income") > 50000:
            return 0.4
        elif self.features.get("income") > 30000:
            return 0.7
        else:
            return 1

    def _stop_cost(self, arguments=None):
        return 1

    def _change_education_cost(self, arguments=None):
        if arguments == "nessuno":
            result = 1
        elif arguments == "diploma":
            result = 2
        elif arguments == "triennale":
            result = 3
        elif arguments == "magistrale":
            result = 4
        else:
            result = 5
        return result

    def _change_occupation_cost(self, arguments=None):
        if arguments == "disoccupato":
            result = 5
        elif arguments == "operaio":
            result = 6
        elif arguments == "privato":
            result = 7
        elif arguments == "impiegato":
            result = 7
        elif arguments == "manager":
            result = 8
        else:
            result = 9
        return result * self._rescale_by_edu()

    def _change_income_cost(self, arguments=None):
        return self.arguments["INT"].index(arguments)+14 * self._rescale_by_job()

    def _change_house_cost(self, arguments=None):
        if arguments == "none":
            result = 10
        elif arguments == "rent":
            result = 12
        else:
            result = 14
        return result * self._rescale_by_income()

    def _change_relation_cost(self, arguments=None):

        if arguments == "sposato/a":
            result = 5
        elif arguments == "divorziato/a":
            result = 8
        else:
            result = 8

        return result


    def _intervene_postcondition(self, init_state, current_state):

        result = get_loan(self.features)
        val_out = 1 if result == "bad" else 0

        self.custom_tensorboard_metrics["call_to_the_classifier"] += 1

        return val_out != self.previous_classification

    def get_observation_columns(self):

        _, data = self.transform_user()

        # Drop numeric columns
        data.drop(columns=self.numerical_cols, inplace=True)

        return self.boolean_cols + list(data.columns)

    def get_observation(self):

        _, data = self.transform_user()

        # Drop numeric columns
        data.drop(columns=self.numerical_cols, inplace=True)

        # Get boolean version of the features
        numeric_bools = self._get_boolean_conditions(self.features)
        bools = torch.FloatTensor(np.concatenate((numeric_bools, data.values[0]), axis=0))

        return bools

    def get_state(self):
        return self.features.copy()

    def get_obs_dimension(self):
        return len(self.get_observation())

    def get_mask_over_args(self, program_index):
        """
        Return the available arguments which can be called by that given program
        :param program_index: the program index
        :return: a max over the available arguments
        """

        program = self.get_program_from_index(program_index)
        permitted_arguments = self.programs_library[program]["args"]

        mask = []
        for k, r in self.arguments.items():
            if k == permitted_arguments:
                mask.append(np.ones(len(r)))
            else:
                mask.append(np.zeros(len(r)))

        return np.concatenate(mask, axis=None)

    def get_additional_parameters(self):
        return {
            "types": self.arguments
        }

    def compare_state(self, state_a, state_b):
        return state_a == state_b

