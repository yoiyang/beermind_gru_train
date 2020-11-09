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
    # test blue score

    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)

    # load the model
    model.load_state_dict(torch.load("./model_cache_at2_16000",map_location='cpu'))
    # deactivate the training mode
    model.eval()

    # get the test data (only the style and rating)
    test_data = load_data("/datasets/cs190f-public/Beeradvocate_TestOriginal.csv")

    for i in range(1):

        # sample test data
        test_batch = test_data.sample(n = 100)
        test_batch.reset_index(drop=True, inplace=True)

        # pad EOS in the end to make same dimension
        pad_data(test_batch)

        X_test = process_test_data(test_batch)
        y_test = process_train_label(test_batch)
        del test_batch

        cfg['train'] = False
        # let the model generat
        reviews = generate(model,X_test,cfg)

        # Save the generated outputs to a file
        out_fname = "outputs_for_test" + str(i) + ".txt"
        save_to_file(reviews, out_fname)
        del X_test

        # compare bleu score
        y_test = torch.from_numpy(y_test).long().to(computing_device)

        # find blue score
        # Arugment # 1 = list(list(list(words))) -> a list of a lists of reference
        # Argument # 2 = list(list(string)) -> a list of hypotheses
        blue_scores = bleu_score.corpus_bleu(wordify(sentencify(y_test),ref = True),wordify(reviews))

        del y_test, reviews

        print("Bleu score:" + str(blue_scores))
