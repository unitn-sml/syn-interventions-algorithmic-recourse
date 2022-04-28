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

class AdultEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim=20):
        super(AdultEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, encoding_dim)
        self.l2 = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x

class AdultEnvironment(Environment):

    def __init__(self, classifier, encoder, scaler, dataset="./data/adult_score/train.csv",
                 sample_from_errors_prob=0.3, sample_env=True):

        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                'CHANGE_WORKCLASS': self._change_workclass,
                                                'CHANGE_EDUCATION': self._change_education,
                                                'CHANGE_OCCUPATION': self._change_occupation,
                                                'CHANGE_RELATIONSHIP': self._change_relationship,
                                                'CHANGE_HOURS': self._change_hours
                                                }.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        'CHANGE_WORKCLASS': self._change_workclass_p,
                                                        'CHANGE_EDUCATION': self._change_education_p,
                                                        'CHANGE_OCCUPATION': self._change_occupation_p,
                                                        'CHANGE_RELATIONSHIP': self._change_relationship_p,
                                                        'CHANGE_HOURS': self._change_hours_p,
                                                        'INTERVENE': self._placeholder_stop}.items()))

        self.prog_to_postcondition = OrderedDict(sorted({'INTERVENE': self._intervene_postcondition}.items()))

        self.programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'args': 'NONE'},
                                                    'CHANGE_WORKCLASS': {'level': 0, 'args': 'WORK'},
                                                    'CHANGE_EDUCATION': {'level': 0, 'args': 'EDU'},
                                                    'CHANGE_OCCUPATION': {'level': 0, 'args': 'OCC'},
                                                    'CHANGE_RELATIONSHIP': {'level': 0, 'args': 'REL'},
                                                    'CHANGE_HOURS': {'level': 0, 'args': 'HOUR'},
                                                    'INTERVENE': {'level': 1, 'args': 'NONE'}}.items()))

        self.arguments = OrderedDict(sorted({
                                                "WORK": ["Never-worked", "Without-pay", "Self-emp-not-inc", "Self-emp-inc","Private", "Local-gov", "State-gov", "Federal-gov", "?"],
                                                "EDU": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate', 'Assoc-acdm', 'Assoc-voc', 'Prof-school'],
                                                "OCC": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"],
                                                "REL": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
                                                "HOUR": list(range(0,25)),
                                                "NONE": [0]
                                            }.items()))

        self.prog_to_cost = OrderedDict(sorted({'STOP': self._stop_cost,
                                                'CHANGE_WORKCLASS': self._change_workclass_cost,
                                                'CHANGE_EDUCATION': self._change_education_cost,
                                                'CHANGE_OCCUPATION': self._change_occupation_cost,
                                                'CHANGE_RELATIONSHIP': self._change_relationship_cost,
                                                'CHANGE_HOURS': self._change_hours_cost
                                                }.items()))

        self.cost_per_argument = {
            "WORK": {"Never-worked": 4, "Without-pay":5, "Self-emp-not-inc":6, "Self-emp-inc": 6,
                     "Private":7, "Local-gov": 7, "State-gov":8, "Federal-gov":8, "?": 2},
            "OCC": {"Tech-support": 8,
                     "Craft-repair": 6,
                     "Other-service": 6,
                     "Sales": 8,
                     "Exec-managerial": 9,
                     "Prof-specialty": 8,
                     "Handlers-cleaners": 7,
                     "Machine-op-inspct":7,
                     "Adm-clerical":8,
                     "Farming-fishing":6,
                     "Transport-moving":6,
                     "Priv-house-serv":6,
                     "Protective-serv":6,
                     "Armed-Forces":6,
                     "?": 5
                    },
            "REL": {"Wife": 5, "Own-child":6, "Husband":5, "Not-in-family":4, "Other-relative":4, "Unmarried":4}

        }

        self.categorical_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
        self.numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
        self.boolean_cols = [
                                  "age < 17",
                                  "19 <= age < 33",
                                  "33 <= age < 47",
                                  "47 <= age < 61",
                                  "61 <= age < 75",
                                  "age >= 90",
                                  "capital_gain == 0",
                                  "0 < capital_gain < 25000",
                                  "25000 <= capital_gain < 50000",
                                  "50000 <= capital_gain < 75000",
                                  "75000 <= capital_gain < 99999",
                                  "capital_gain >= 99999",
                                  "capital_loss == 0",
                                  "0 <= capital_loss < 1089",
                                  "1089 <= capital_loss < 2178",
                                  "2178 <= capital_loss < 3267",
                                  "3267 <= capital_loss < 4356",
                                  "capital_loss >= 4356",
                                  "0 <= hours_per_week < 25",
                                  "25 <= hours_per_week < 50",
                                  "50 <= hours_per_week < 75",
                                  "75 <= hours_per_week < 100",
                                  "hours_per_week >= 100"
                              ]

        # Set up the max length of the interventions
        self.max_depth_dict = {1: 5}

        # Set up the dataset
        self.setup_dataset(dataset, "<=50K", "income_target", "predicted")

        # Set up the system with various informations
        self.setup_system(self.boolean_cols, self.categorical_cols, encoder, scaler,
                          classifier, Net, sample_env, net_layers=5, net_size=108)

        # Call parent constructor
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
            data[0] < 17,
            19 <= data[0] < 33,
            33 <= data[0] < 47,
            47 <= data[0] < 61,
            61 <= data[0] < 75,
            data[0] >= 90,
            data[10] == 0,
            0 < data[10] < 25000,
            25000 <= data[10] < 50000,
            50000 <= data[10] < 75000,
            75000 <= data[10] < 99999,
            data[10] >= 99999,
            data[11] == 0,
            0 <= data[11] < 1089,
            1089 <= data[11] < 2178,
            2178 <= data[11] < 3267,
            3267 <= data[11] < 4356,
            data[11] >= 4356,
            0 <= data[12] < 25,
            25 <= data[12] < 50,
            50 <= data[12] < 75,
            75 <= data[12] < 100,
            data[12] >= 100
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

    ### ACTIONS

    def _stop(self, arguments=None):
        return True

    def _change_workclass(self, arguments=None):
        self.memory[1] = arguments

    def _change_education(self, arguments=None):
        self.memory[3] = arguments

    def _change_occupation(self, arguments=None):
        self.memory[6] = arguments

    def _change_relationship(self, arguments=None):
        self.memory[7] = arguments

    def _change_hours(self, arguments=None):
        self.memory[12] += arguments

    ### ACTIONA PRECONDTIONS

    def _change_workclass_p(self, arguments=None):
        return self.cost_per_argument["WORK"][arguments] >= self.cost_per_argument["WORK"][self.memory[1]]

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) > self.arguments["EDU"].index(self.memory[3])

    def _change_occupation_p(self, arguments=None):
        return self.cost_per_argument["OCC"][arguments] >= self.cost_per_argument["OCC"][self.memory[6]]

    def _change_relationship_p(self, arguments=None):

        if arguments == "Wife":
            return self.memory[7] != "Husband"
        elif arguments == "Husband":
            return self.memory[7] != "Wife"

        return self.cost_per_argument["REL"][arguments] >= self.cost_per_argument["REL"][self.memory[7]]

    def _change_hours_p(self, arguments=None):
        return self.memory[12] == arguments

    ### COSTS

    def _stop_cost(self, arguments=None):
        return True

    def _change_workclass_cost(self, arguments=None):
        return (self.cost_per_argument.get("WORK").get(arguments)+10) * self._rescale_by_edu()

    def _change_education_cost(self, arguments=None):
        return self.arguments.get("EDU").index(arguments)+10

    def _change_occupation_cost(self, arguments=None):
        return (self.cost_per_argument.get("OCC").get(arguments)+10) * self._rescale_by_workclass()

    def _change_relationship_cost(self, arguments=None):
        return self.cost_per_argument.get("REL").get(arguments)+10

    def _change_hours_cost(self, arguments=None):
        return self.arguments["HOUR"].index(arguments)+10

    # Rescaling

    def _rescale_by_edu(self):
        return 1/(len(self.arguments["EDU"])-self.arguments.get("EDU").index(self.memory[3]))

    def _rescale_by_workclass(self):
        return 1/(len(self.arguments["WORK"])-self.arguments.get("WORK").index(self.memory[1]))

    ### POSTCONDITIONS

    def _intervene_postcondition(self, init_state, current_state):
        self.custom_tensorboard_metrics["call_to_the_classifier"] += 1
        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)
        return torch.round(val_out).item() != self.previous_classification

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

        return bools
        #return torch.cat((bools, classification), 0)

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