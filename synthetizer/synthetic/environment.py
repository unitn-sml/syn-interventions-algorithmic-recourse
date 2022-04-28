from classifiers.train_model import Net
from data.synthetic.generate import get_job, get_income, get_housing, get_loan

from generalized_alphanpi.environments.environment import Environment

from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

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

    def __init__(self, classifier, encoder, scaler, dataset="data/synthetic/train.csv",
                 sample_from_errors_prob=0.3, sample_env=True):
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

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.arguments_index = [(i, v) for i, v in enumerate(self.complete_arguments)]

        self.max_depth_dict = {1: 7}

        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        self.data = pd.read_csv(dataset, sep=",")
        self.data = self.data.dropna() # Drop columns with na
        self.data = self.data[self.data.loan == "bad"]

        self.y = self.data.loan
        self.y.reset_index(drop=True, inplace=True)

        self.data = self.data.drop(columns=["loan"])
        self.data.reset_index(drop=True, inplace=True)

        # Load encoder (one-hot) and scaler
        self.data_encoder = pickle.load(open(encoder, "rb"))
        self.data_scaler = pickle.load(open(scaler, "rb"))

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

        # Needed for validation
        self.sample_env = sample_env
        self.current_idx = 0

        # Custom metric we want to print at each iteration
        self.custom_tensorboard_metrics = {
            "call_to_the_classifier": 0
        }

        super().__init__(self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_depth_dict,
                         complete_arguments=self.complete_arguments, prog_to_cost=self.prog_to_cost,
                         sample_from_errors_prob=sample_from_errors_prob,
                         custom_tensorboard_metrics=self.custom_tensorboard_metrics)

    def get_state_str(self, state):
        result = get_loan(state[2], state[1], state[3], state[5], state[4], state[7])
        return state, result

    def _placeholder_stop(self, args=None):
        return True

    def _get_boolean_conditions(self, data):

        return [
            30 <= data[0] < 40,
            40 <= data[0] < 50,
            50 <= data[0] < 60,
            data[0] >= 60,
            0 <= int(data[5]) < 1000,
            1000 <= data[5] < 10000,
            10000 <= data[5] < 30000,
            30000 <= data[5] < 50000,
            data[5] >= 50000,
            0 <= int(data[3]) < 10000,
            10000 <= int(data[3]) < 30000,
            30000 <= int(data[3]) < 50000,
            50000 <= int(data[3]) < 70000,
            70000 <= int(data[3]) < 100000,
            100000 <= int(data[3]) < 150000,
            150000 <= int(data[3]) < 200000,
            data[3] >= 200000
        ]

    def parse_observation(self, data):
        """
        Parse an environment, by keeping the categorical values, but by removing
        the numerical values in favour of booleans conversions
        :param data:
        :return:
        """

        booleans = self._get_boolean_conditions(data)

        data = pd.DataFrame([data], columns=self.data.columns)
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
        data_df = pd.DataFrame([self.memory], columns=self.data.columns)
        return self.preprocess_single(data_df)

    def init_env(self):

        self.memory = self.sample_from_failed_state("INTERVENE")
        if self.memory is None:
            if self.sample_env:
                self.memory = self.data.sample(1).values.tolist()[0]
            else:
                self.memory = self.data.iloc[[self.current_idx]].values.tolist()[0]
                if self.current_idx > len(self.data):
                    self.current_idx = len(self.data) - 1

        result = get_loan(self.memory[2], self.memory[1], self.memory[3], self.memory[5], self.memory[4], self.memory[7])
        classification = 1 if result == "bad" else 0

        self.previous_classification = classification

    def reset_env(self, task_index):

        task_name = self.get_program_from_index(task_index)

        self.memory = self.sample_from_failed_state(task_name)
        if self.memory is None:
            if self.sample_env:
                self.memory = self.data.sample(1).values.tolist()[0]
            else:
                self.memory = self.data.iloc[[self.current_idx]].values.tolist()[0]
                self.current_idx += 1
                if self.current_idx > len(self.data):
                    self.current_idx = len(self.data)-1

        result = get_loan(self.memory[2], self.memory[1], self.memory[3], self.memory[5], self.memory[4], self.memory[7])
        classification = 1 if result == "bad" else 0

        self.previous_classification = classification
        self.has_been_reset = True

        return 0, 0

    def reset_to_state(self, state):
        self.memory = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    def _stop(self, arguments=None):
        return True

    def _change_education(self, arguments=None):
        self.memory[1] = arguments

    def _change_occupation(self, arguments=None):
        self.memory[2] = arguments

    def _change_income(self, arguments=None):
        self.memory[3] += arguments

    def _change_house(self, arguments=None):
        self.memory[4] = arguments

    def _change_relation(self, arguments=None):
        self.memory[7] = arguments

    def _change_income_p(self, arguments=None):
        return self.memory[3] > 0

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) > self.arguments["EDU"].index(self.memory[1])

    def _change_occupation_p(self, arguments=None):
        return self.arguments["WORK"].index(arguments) > self.arguments["WORK"].index(self.memory[2])

    def _change_house_p(self, arguments=None):
        return self.arguments["HOUS"].index(arguments) > self.arguments["HOUS"].index(self.memory[4])

    def _change_relation_p(self, arguments=None):

        if self.memory[7] in ["single", "divorziato/a"]:
            if self.memory[7] == "single":
                return arguments != "vedovo/a" and arguments != "divorziato/a"
            else:
                return arguments != "vedovo/a"

        return self.arguments["RELATION"].index(arguments) > self.arguments["RELATION"].index(self.memory[7])

    def _rescale_by_relation(self):
        if self.memory[7] == "sposato/a":
            return 0.5
        elif self.memory[7] == "divorziato/a":
            return 0.8
        else:
            return 1

    def _rescale_by_edu(self):
        if self.memory[1] == "phd":
            return 0.3
        elif self.memory[1] == "magistrale":
            return 0.4
        elif self.memory[1] == "triennale":
            return 0.5
        elif self.memory[1] == "diploma":
            return 0.7
        else:
            return 1

    def _rescale_by_job(self):
        if self.memory[2] == "ceo":
            return 0.4
        if self.memory[2] == "manager":
            return 0.5
        elif self.memory[2] == "privato":
            return 0.6
        elif self.memory[2] == "impiegato":
            return 0.7
        elif self.memory[2] == "operaio":
            return 0.8
        else:
            return 1

    def _rescale_by_income(self):
        if self.memory[3] > 130000:
            return 0.1
        elif self.memory[3] > 100000:
            return 0.2
        elif self.memory[3] > 70000:
            return 0.3
        elif self.memory[3] > 50000:
            return 0.4
        elif self.memory[3] > 30000:
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

        result = get_loan(self.memory[2], self.memory[1], self.memory[3], self.memory[5], self.memory[4], self.memory[7])
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
        numeric_bools = self._get_boolean_conditions(self.memory)
        bools = torch.FloatTensor(np.concatenate((numeric_bools, data.values[0]), axis=0))

        # Get classification
        #result = get_loan(self.memory[2], self.memory[1], self.memory[3], self.memory[5], self.memory[4], self.memory[7])
        #classification = torch.FloatTensor([1]) if result == "bad" else torch.FloatTensor([0])

        return bools
        #return torch.cat((bools,classification), 0)

    def get_state(self):
        return list(self.memory)

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
            "programs_types": self.programs_library,
            "types": self.arguments
        }

    def compare_state(self, state_a, state_b):
        return state_a == state_b

