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

    style_rating_inds = {}
    test_data.sort_values(by=['styles', 'ratings'],inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    for line in iter(test_data.iterrows()):
        s = encode_style[line[1]['styles']]
        if s not in style_rating_inds:
            style_rating_inds[s]=dict()
        r = int(line[1]['ratings'])
        if r not in style_rating_inds[s]:
            style_rating_inds[s][r] = line[0]

    b_score = 0
    i = 0

    # try:
    with open("reviews_tau_04.txt") as f:
        while i < len(test_data):
            # find blue score
            # Arugment # 1 = (list(list(words))) -> a list of a lists of reference
            # Argument # 2 = (list(string)) -> a list of hypotheses
            blue_scores = bleu_score.sentence_bleu(wordify([test_data['reviews'][i]]),
                                                 wordify([f.readline()])[0])
            b_score += blue_scores
            i += 1
            if (i % 1000 == 0):
                print("Bleu score:" + str(blue_scores))


    print(b_score/count)
