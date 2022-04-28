import pandas as pd
from pandas import DataFrame
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ENCODER = OneHotEncoder(handle_unknown='ignore')
SCALER = MinMaxScaler()

def preprocess(data: DataFrame, numerical_cols, categorical_cols, target, target_value, fit=True) -> DataFrame:
    """
    Preprocess the German data by creating one-hot encodings for the
    categorical variables and by applying np.log to the numeric values.
    The response variable (risk) is mapped to 1 (bad) and 0 (good).

    :param data: original dataframe with german data
    :return: preprocessed dataframe
    """

    # Apply ohe on data
    if fit:
        ENCODER.fit(data[categorical_cols])
        SCALER.fit(data[numerical_cols])

    cat_ohe = ENCODER.transform(data[categorical_cols]).toarray()
    data[numerical_cols] = SCALER.transform(data[numerical_cols])

    ohe_df = pd.DataFrame(cat_ohe, columns=ENCODER.get_feature_names_out(input_features=categorical_cols))
    data = pd.concat([data, ohe_df], axis=1).drop(columns=categorical_cols, axis=1)

    data[target] = data[target].apply(lambda x: 1 if x==target_value else 0)

    return data


def resample(data: DataFrame, target: str) -> DataFrame:
    """
    Resample the original dataframe to have a balanced distribution of classes.

    :param data: original dataframe
    :return: resampled dataframe
    """

    lst = [data]
    max_size = data[target].value_counts().max()
    for class_index, group in data.groupby(target):
        lst.append(group.sample(max_size - len(group), replace=True))
    data = pd.concat(lst)
    return data


def split_data(path, numerical_cols, categorical_cols, target, target_value,
               do_resample=True):
    # Convert features
    # Oversample minority class
    data = pd.read_csv(path, sep=",")

    train, test = train_test_split(data, test_size=0.2, stratify=data[target])

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train_preprocessed = preprocess(train.copy(), numerical_cols, categorical_cols, target, target_value)
    test_preprocessed = preprocess(test.copy(), numerical_cols, categorical_cols, target, target_value, fit=False)

    if do_resample:
        train_preprocessed = resample(train_preprocessed, target)

    return train, test, train_preprocessed, test_preprocessed


class Data(Dataset):

    def __init__(self, data, target):
        self.data = data

        # Split predictor from features
        self.y = self.data[target]
        self.data = self.data.drop(columns=[target])

    def feature_size(self):
        return len(self.data.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx].values).float()
        response = torch.tensor(self.y.iloc[idx]).float()
        return features, response