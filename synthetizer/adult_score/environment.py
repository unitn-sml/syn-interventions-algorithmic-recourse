from classifiers.train_model import Net

from rl_mcts.core.environment import Environment

from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdultEnvironment(Environment):

    def __init__(self, f, model, preprocessor):

        self.preprocessor = preprocessor

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
                                                        }.items()))

        self.prog_to_postcondition = self._intervene_postcondition

        self.programs_library = OrderedDict(sorted({'STOP': {'index': 0, 'level': -1, 'args': 'NONE'},
                                                    'CHANGE_WORKCLASS': {'index': 1, 'level': 0, 'args': 'WORK'},
                                                    'CHANGE_EDUCATION': {'index': 2, 'level': 0, 'args': 'EDU'},
                                                    'CHANGE_OCCUPATION': {'index': 3, 'level': 0, 'args': 'OCC'},
                                                    'CHANGE_RELATIONSHIP': {'index': 4, 'level': 0, 'args': 'REL'},
                                                    'CHANGE_HOURS': {'index': 5, 'level': 0, 'args': 'HOUR'},
                                                    'INTERVENE': {'index': 6, 'level': 1, 'args': 'NONE'}}.items()))

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

        self.max_depth_dict = 5

        # Call parent constructor
        super().__init__(f, model, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_depth_dict,
                         prog_to_cost=self.prog_to_cost)

    def get_state_str(self, state):
        with torch.no_grad():
            tmp = self.transform_user()[0]
            val_out = self.classifier(tmp)
        return state, torch.round(val_out).item()

    def _placeholder_stop(self, args=None):
        return True

    def reset_to_state(self, state):
        self.features = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    ### ACTIONS

    def _stop(self, arguments=None):
        return True

    def _change_workclass(self, arguments=None):
        self.features["workclass"] = arguments

    def _change_education(self, arguments=None):
        self.features["education"] = arguments

    def _change_occupation(self, arguments=None):
        self.features["occupation"] = arguments

    def _change_relationship(self, arguments=None):
        self.features["relationship"] = arguments

    def _change_hours(self, arguments=None):
        self.features["hours_per_week"] += arguments

    ### ACTIONA PRECONDTIONS

    def _change_workclass_p(self, arguments=None):
        return self.cost_per_argument["WORK"][arguments] >= self.cost_per_argument["WORK"][self.features.get("workclass")]

    def _change_education_p(self, arguments=None):
        return self.arguments["EDU"].index(arguments) > self.arguments["EDU"].index(self.features.get("education"))

    def _change_occupation_p(self, arguments=None):
        return self.cost_per_argument["OCC"][arguments] >= self.cost_per_argument["OCC"][self.features.get("occupation")]

    def _change_relationship_p(self, arguments=None):

        if arguments == "Wife":
            return self.features.get("relationship") != "Husband"
        elif arguments == "Husband":
            return self.features.get("relationship") != "Wife"

        return self.cost_per_argument["REL"][arguments] >= self.cost_per_argument["REL"][self.features.get("relationship")]

    def _change_hours_p(self, arguments=None):
        return self.features.get("hours") == arguments

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
        return 1/(len(self.arguments["EDU"])-self.arguments.get("EDU").index(self.features.get("education")))

    def _rescale_by_workclass(self):
        return 1/(len(self.arguments["WORK"])-self.arguments.get("WORK").index(self.features.get("workclass")))

    ### POSTCONDITIONS

    def _intervene_postcondition(self, init_state, current_state):
        obs = self.preprocessor.transform(
            pd.DataFrame.from_records(
                [self.features]
            )
        )
        return self.model.predict(obs)[0] == 0

    def get_observation(self):

        obs = self.preprocessor.transform(
            pd.DataFrame.from_records(
                [self.features]
            )
        )
        return torch.FloatTensor(obs)

    def get_additional_parameters(self):
        return {
            "types": self.arguments
        }