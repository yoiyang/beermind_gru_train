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
    count = 0
    # try:
    for i in range(1):

        # sample test data
        for s in style_rating_inds:
            for r in style_rating_inds[s]:
                # find same rating and style
                i = style_rating_inds[s][r]
                j = i+1
                style = find_style(s)
    #             while (test_data['styles'][j] == style and int(test_data['ratings'][j]) == r):
    #                 j += 1

                print("For " + style + " rating " + str(r) + " from " + str(i) + " to " + str(j))
                test_batch = test_data[i:j]
                test_batch.reset_index(drop=True, inplace=True)

                pad_data(test_batch)
                X_test = process_test_data(test_data[i:i+1])
                y_test = process_train_label(test_batch)
                del test_batch

                cfg['train'] = False
                # let the model generat
                reviews = generate(model,X_test,cfg)

                # Save the generated outputs to a file
    #             out_fname = "outputs_for_test" + str(i) + ".txt"
    #             save_to_file(reviews, out_fname)

                del X_test

                # compare bleu score
                y_test = torch.from_numpy(y_test).long().to(computing_device)

                # find blue score
                # Arugment # 1 = (list(list(words))) -> a list of a lists of reference
                # Argument # 2 = (list(string)) -> a list of hypotheses
    #                 print(wordify(sentencify(y_test),)
    #                 print(reviews)
                blue_scores = bleu_score.sentence_bleu(wordify(sentencify(y_test),ref = False),
                                                     wordify(reviews)[0])
                b_score += blue_scores
                count += 1
                print("Bleu score:" + str(blue_scores))
                del y_test, reviews

    # except:
        print(b_score/count)
