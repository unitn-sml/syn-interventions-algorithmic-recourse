from classifiers.train_model import Net

from generalized_alphanpi.environments.environment import Environment

from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class GermanEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim=20):
        super(GermanEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, encoding_dim)
        self.l2 = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class GermanEnvironment(Environment):

    def __init__(self, classifier, encoder, scaler, dataset="./data/german_credit/train.csv",
                 sample_from_errors_prob=0.3, sample_env=True):
        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'CHANGE_SAVINGS': self._change_savings,
                                                'CHANGE_JOB': self._change_job,
                                                'CHANGE_CREDIT': self._change_credit,
                                                'CHANGE_HOUSING': self._change_housing,
                                                'CHANGE_DURATION': self._change_duration,
                                                'CHANGE_PURPOSE': self._change_purpose,
                                                }.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        'CHANGE_JOB': self._p_change_job,
                                                        'CHANGE_SAVINGS': self._p_change_savings,
                                                        'CHANGE_HOUSING': self._p_change_housing,
                                                        'CHANGE_CREDIT': self._p_change_credit,
                                                        'CHANGE_DURATION': self._p_change_duration,
                                                        'CHANGE_PURPOSE': self._p_change_purpose,
                                                        'INTERVENE': self._placeholder_stop}.items()))

        self.prog_to_postcondition = OrderedDict(sorted({'INTERVENE': self._intervene_postcondition}.items()))

        self.programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'args': 'NONE'},
                                                    'CHANGE_SAVINGS': {'level': 0, 'args': 'SAVINGS'},
                                                    'CHANGE_JOB': {'level': 0, 'args': 'JOB'},
                                                    'CHANGE_HOUSING': {'level': 0, 'args': 'HOUSE'},
                                                    'CHANGE_CREDIT': {'level': 0, 'args': 'CREDIT'},
                                                    'CHANGE_DURATION': {'level': 0, 'args': 'DURATION'},
                                                    'CHANGE_PURPOSE': {'level': 0, 'args': 'PURPOSE'},
                                                    'INTERVENE': {'level': 1, 'args': 'NONE'}}.items()))

        self.arguments = OrderedDict(sorted({
                                                "JOB": ["unskilled_non_resident","unskilled_resident","skilled","highly_skilled"],
                                                "HOUSE": ["free", "rent", "own"],
                                                "CREDIT": [100, 1000, 2000, 5000],
                                                "SAVINGS": ["unknown", "little", "moderate", "rich", "quite_rich"],
                                                "DURATION": [10, 20, 30],
                                                "PURPOSE": ['business', 'car', 'domestic_appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'],
                                                "NONE": [0]
                                            }.items()))

        self.prog_to_cost = OrderedDict(sorted({'STOP': self._stop_cost,
                                                'CHANGE_SAVINGS': self._change_savings_cost,
                                                'CHANGE_JOB': self._change_job_cost,
                                                'CHANGE_CREDIT': self._change_credit_cost,
                                                'CHANGE_HOUSING': self._change_housing_cost,
                                                'CHANGE_DURATION': self._change_duration_cost,
                                                'CHANGE_PURPOSE': self._change_purpose_cost,
                                                }.items()))

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.arguments_index = [(i, v) for i, v in enumerate(self.complete_arguments)]

        self.max_depth_dict = {1: 5}

        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        self.data = pd.read_csv(dataset, sep=",")
        self.data = self.data.dropna()
        self.data = self.data[self.data.risk == "bad"]
        self.data = self.data[self.data.predicted == self.data.risk]

        self.y = self.data.risk
        self.y.reset_index(drop=True, inplace=True)

        self.data = self.data.drop(columns=["risk"])
        self.data = self.data.drop(columns=["predicted"])
        self.data.reset_index(drop=True, inplace=True)

        # Load encoder and scaler
        self.data_encoder = pickle.load(open(encoder, "rb"))
        self.data_scaler = pickle.load(open(scaler, "rb"))

        # Load the classifier
        checkpoint = torch.load(classifier)
        self.classifier = Net(29, 5) # Taken empirically from the classifier
        self.classifier.load_state_dict(checkpoint)

        self.categorical_cols = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose"]
        self.numerical_cols = ["age", "credit_amount", "duration"]
        self.boolean_cols = [
                                  "age < 19",
                                  "19 <= age < 33",
                                  "33 <= age < 47",
                                  "47 <= age < 61",
                                  "61 <= age < 75",
                                  "age >= 75",
                                  "credit_amount < 250",
                                  "250 <= credit_amount < 4794",
                                  "4794 <= credit_amount < 9588",
                                  "9588 <= credit_amount < 14382",
                                  "14382 <= credit_amount < 19176",
                                  "credit_amount >= 19176",
                                  "duration < 4",
                                  "4 <= duration < 21",
                                  "21 <= duration < 38",
                                  "38 <= duration < 55",
                                  "55 <= duration < 72",
                                  "duration >= 72"
                              ]
        self.parsed_columns = self.boolean_cols + self.categorical_cols

        self.previous_classification = 0

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
        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)
        return state, torch.round(val_out).item()

    def _placeholder_stop(self, args=None):
        return True

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

    def _get_boolean_conditions(self, data):

        return [
            data[0] < 19,
            19 <= data[0] < 33,
            33 <= data[0] < 47,
            47 <= data[0] < 61,
            61 <= data[0] < 75,
            data[0] >= 75,
            data[6] < 250,
            250 <= data[6] < 4794,
            4794 <= data[6] < 9588,
            9588 <= data[6] < 14382,
            14382 <= data[6] < 19176,
            data[6] >= 19176,
            data[0] < 4,
            4 <= data[7] < 21,
            21 <= data[7] < 38,
            38 <= data[7] < 55,
            55 <= data[7] < 72,
            data[7] >= 72
        ]

    def correct_numeric(self, df):
        df[self.numerical_cols] = df[self.numerical_cols].apply(pd.to_numeric)
        return df

    def preprocess_single(self, data):

        data.reset_index(drop=True, inplace=True)
        data[self.numerical_cols] = self.data_scaler.transform(data[self.numerical_cols])
        cat_ohe = self.data_encoder.transform(data[self.categorical_cols]).toarray()
        ohe_df = pd.DataFrame(cat_ohe,
                              columns=self.data_encoder.get_feature_names_out(input_features=self.categorical_cols))
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

        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)

        self.previous_classification = torch.round(val_out).item()

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
                    self.current_idx = len(self.data) - 1

        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)

        self.previous_classification = torch.round(val_out).item()
        self.has_been_reset = True

        return 0, 0

    def reset_to_state(self, state):
        self.memory = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    ## PRECONDITIONS

    def _p_change_job(self, arguments=None):
        return self.arguments["JOB"].index(arguments) > self.arguments["JOB"].index(self.memory[2])

    def _p_change_housing(self, arguments=None):
        return self.arguments["HOUSE"].index(arguments) > self.arguments["HOUSE"].index(self.memory[3])

    def _p_change_credit(self, arguments=None):
        return self.memory[6] > 0 and (self.memory[6]-self.arguments["CREDIT"].index(arguments)) > 0

    def _p_change_savings(self, arguments=None):
        return self.arguments["SAVINGS"].index(arguments) > self.arguments["SAVINGS"].index(self.memory[4])

    def _p_change_purpose(self, arguments=None):
        return arguments != self.memory[8]

    def _p_change_duration(self, arguments=None):
        return self.memory[7] > 0

    ## ACTIONS

    def _stop(self, arguments=None):
        return True

    def _change_job(self, arguments=None):
        self.memory[2] = arguments

    def _change_housing(self, arguments=None):
        self.memory[3] = arguments

    def _change_credit(self, arguments=None):
        self.memory[6] -= arguments

    def _change_savings(self, arguments=None):
        self.memory[4] = arguments

    def _change_purpose(self, arguments=None):
        self.memory[8] = arguments

    def _change_duration(self, arguments=None):
        self.memory[7] = arguments

    def _intervene_postcondition(self, init_state, current_state):
        self.custom_tensorboard_metrics["call_to_the_classifier"] += 1
        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)
        return torch.round(val_out).item() != self.previous_classification

    ## COSTS

    def _stop_cost(self, arguments=None):
        return 5

    def _change_savings_cost(self, arguments=None):
        return self.arguments["SAVINGS"].index(arguments)+10 * self._rescale_by_job()

    def _change_job_cost(self, arguments=None):
        return self.arguments["JOB"].index(arguments)+10 * self._rescale_by_age()

    def _change_housing_cost(self, arguments=None):
        return self.arguments["HOUSE"].index(arguments)+10 * self._rescale_by_savings()

    def _change_credit_cost(self, arguments=None):
        return self.arguments["CREDIT"].index(arguments)+10

    def _change_purpose_cost(self, arguments=None):
        return 10

    def _change_duration_cost(self, arguments=None):
        return self.arguments["DURATION"].index(arguments)+10

    def _rescale_by_job(self):
        if self.memory[2] in ["skilled", "highly_skilled"]:
            return 0.2
        elif self.memory[2] == "skilled":
            return 0.3
        elif self.memory[2] == "unskilled_resident":
            return 0.6
        else:
            return 1

    def _rescale_by_savings(self):
        if self.memory[4] in ["quite_rich", "rich"]:
            return 0.2
        elif self.memory[4] == "moderate":
            return 0.3
        elif self.memory[4] == "little":
            return 0.7
        else:
            return 1

    def _rescale_by_age(self):
        if self.memory[0] < 25:
            return 0.2
        elif self.memory[0] < 40:
            return 0.3
        elif self.memory[0] < 55:
            return 0.7
        else:
            return 1

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
        #with torch.no_grad():
        #    tmp = self.transform_user()[0]
        #    val_out = self.classifier(tmp)
        #    classification = torch.round(val_out)

        #return torch.cat((bools, classification), 0)
        return bools

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
