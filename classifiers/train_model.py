from classifiers.utils.data import Data, split_data, ENCODER, SCALER
from classifiers.utils.net import Net

import numpy as np
import random

import yaml
import os

from datetime import datetime

import pickle

import torch
import torch.optim as optim

from argparse import  ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the config file")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed = config.get("seed")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    categorical_cols = config.get("categorical", [])
    numerical_cols = config.get("numerical", [])
    target = config.get("target")

    train_unp, test_unp, train_data, test_data = split_data(config.get("data"),
                                       numerical_cols,
                                       categorical_cols,
                                       target,
                                       config.get("target_bad"),
                                       do_resample=config.get("resample", False)
                                       )

    train = Data(train_data, target)
    validation = Data(test_data, target)

    net = Net(train.feature_size(), config.get("layers", 4))
    print("Feature size: ", train.feature_size())

    trainloader = torch.utils.data.DataLoader(train, batch_size=100,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(validation, batch_size=100,
                                              shuffle=True, num_workers=2)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(config.get("iterations")):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            labels = labels.view((len(labels), 1))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches

                correct = 0
                for i, val in enumerate(valloader, 0):
                    inputs, labels = val

                    val_out = net(inputs)
                    labels = labels.view((len(labels), 1))

                    correct += torch.round(val_out).eq(labels).sum().item()


                print('[%d, %5d] loss: %.3f / val: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10, correct/len(validation)))
                running_loss = 0.0

    train_data.drop(columns=[target], inplace=True)
    test_data.drop(columns=[target], inplace=True)

    train_data["predicted"] = train_data.apply(lambda x:
                                               config.get("target_bad")
                                               if torch.round(net(torch.FloatTensor([x]))).item() == 1.0
                                               else config.get("target_good"), axis=1)
    test_data["predicted"] = test_data.apply(lambda x:
                                                config.get("target_bad")
                                                if torch.round(net(torch.FloatTensor([x]))).item() == 1.0
                                                else config.get("target_good"), axis=1)

    test_data.reset_index(drop=True, inplace=True)
    train_data.reset_index(drop=True, inplace=True)

    train_unp.reset_index(drop=True, inplace=True)
    test_unp.reset_index(drop=True, inplace=True)

    train_unp["predicted"] = train_data["predicted"]
    test_unp["predicted"] = test_data["predicted"]

    print("Saving unprocessed data")
    train_unp.to_csv(config.get("train_save_path", "train.csv"), index=None)
    test_unp.to_csv(config.get("test_save_path", "test.csv"), index=None)

    print("Saving the model")
    time = datetime.now().strftime("%d%m%Y-%H%M%S")
    model_path = os.path.join(
        config.get("save_path"),
        config.get("model_name")+"_"+time+".pth"
    )
    torch.save(net.state_dict(), model_path)

    print("Saving the encoder")
    encoder_path = os.path.join(
        config.get("save_path"),
        config.get("model_name") + "_encoder_" + time + ".pth"
    )
    with open(encoder_path, "wb") as f:
        pickle.dump(ENCODER, f)

    print("Saving the numerical rescaler")
    encoder_path = os.path.join(
        config.get("save_path"),
        config.get("model_name") + "_scaling_" + time + ".pth"
    )
    with open(encoder_path, "wb") as f:
        pickle.dump(SCALER, f)

    print('Finished Training')
