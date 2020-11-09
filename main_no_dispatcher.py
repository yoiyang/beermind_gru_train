import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from models import *
import pandas as pd
from configs import cfg
from configs_beerstyle import encode_style
import pandas as pd
from nltk.translate import bleu_score
from sklearn.utils import shuffle
import torch.nn.functional as F
import datetime

from functions import *

if __name__ == "__main__":
    train_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Train.csv"
    #"BeerAdvocatePA4/Beeradvocate_Train.csv"
    test_data_fname = "/datasets/cs190f-public/BeerAdvocateDataset/BeerAdvocate_Test.csv"
    out_fname = "outputs.txt"

    # train_batch, valid_batch = batchGenerator(data,cfg['batch_size'])
    # X_train, y_train = process_train_data(train_batch)
    # X_valid, y_valid = process_train_data(valid_batch)

    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)

    data = load_data(train_data_fname)
    train(model,cfg,data) # Train the model
    # clean the data after train
    del data

    # train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    # train_data, train_labels = process_train_data(train_data) # Converting DataFrame to numpy array
    # X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels) # Splitting the train data into train-valid data
    X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    del test_data

    # cancel the train mode and start generating using temperature
    cfg['train'] = False
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    torch.save(model.state_dict(), "./model_final") # save the model
    save_to_file(outputs, out_fname) # Save the generated outputs to a file
