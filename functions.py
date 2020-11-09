import string
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

# convert each character to an one-hot vector sized 97 using ASCII value - 32
# exceptions: SOS = 95, EOS = 96, unknown char = 97
def char2pos(c):
    pos = ord(c)
    # if it's SOS or EOS
    if (pos == 2):
        pos = 95
    elif (pos == 3):
        pos = 96
    else:
        # other normal ASCII
        pos -= 32
        if pos < 0:
            # make the \n\t a space
            pos = 0
        elif pos > 94:
            pos = 97

    return pos

def char2oh(c):
    ans = np.zeros(98)
    ans[char2pos(c)] = 1
    return ans

def pos2char(pos):
    pos = int(pos)

    # special characters
    if (pos == 95):
        return '\x02' #SOS
    elif (pos == 96):
        return '\x03' #EOS
    elif (pos == 97):
        return '\t' #unknown -> null

    # normal characters
    return chr(pos + 32)

def oh2char(oh):
    # find which one
    pos = oh.argmax()
    return pos2char(pos)

def load_data(fname):
    # From the csv file given by filename and return a pandas DataFrame of the read csv.
    df = pd.read_csv(fname,header=None,low_memory=False)
    # drop the headers
    df.drop(0,inplace=True)

    if ("Train" in fname or "TestOriginal" in fname):
        # only neened 5: beer/style and 10: review overall 13 for review
        df = df[[5,10,13]]

        # rename columns to 0,1,2 for beer/style, review overall, and review
        df.rename(index=int, columns={5: "styles", 10: "ratings",13:"reviews"},inplace = True)

        # append all reviews with SOS and EOS
        df['reviews'] = '\x02' + df['reviews'] + '\x03'

        # drop Nan reviews
        df.dropna(subset=['reviews'],inplace=True)

    else:
        # if it is test data
        df = df[[5,10]]
        # rename columns to 0,1 for beer/style, review overall
        df.rename(index=int, columns={5: "styles", 10: "ratings"},inplace = True)

    # shuffle the data
    df = shuffle(df)

    df.reset_index(drop=True, inplace=True)

    # cast all ratings as float
    df['ratings'] = df['ratings'].astype(float)

    # I have found that all ratings are within 0 <= x <= 5 with correct format

    return df

def process_test_data(data):
    # Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.

    dat = np.zeros([len(data),1,203])

    review_num = 0
    for line in iter(data.iterrows()):

        # for style
        dat[review_num][0][encode_style[line[1]['styles']]] = 1

        # tensor for rating
        dat[review_num][0][104] = line[1]['ratings']

        # encoded review with a <SOS> in it
        dat[review_num][0][105:] = char2oh('\x02')

        review_num += 1

    # convert data into np array
    return dat

def process_train_data(data):
    # Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).

    dat = np.zeros([len(data),len(data['reviews'][0]),203])
    labels = np.zeros([len(data),len(data['reviews'][0]),1])
    review_num = 0
    for line in iter(data.iterrows()):

        # tensor for style
        s = np.zeros(203)
        s[encode_style[line[1]['styles']]] = 1

        # tensor for rating
        s[104] = line[1]['ratings']

        char_num = 0
        # encoded review characters
        for c in line[1]['reviews']:

            # find its one-hot encoding
            oh_c = char2oh(c)

            # encode the character and put into the buffer
            s[105:] = oh_c

            # put it in data
            dat[review_num][char_num] = s

            # encode label
            if c != '\x02':
                labels[review_num][char_num-1] = [char2pos(c)]

            char_num += 1

        # put one end in the end of label
        labels[review_num][char_num-1] = [char2pos('\x03')]

        review_num += 1

    return dat,labels

def pad_data(orig_data):
    # Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character
    # representation in one hot encoding.

    maxlength = 0
    for i in range(len(orig_data['reviews'])):
        length = len(orig_data['reviews'][i])
        if length > maxlength:
            maxlength = length

    # pad the EOS in each review's end
    for i in range(len(orig_data['reviews'])):
        review = orig_data['reviews'][i]
        orig_data.at[i,'reviews'] = review + ('\x03' * (maxlength - len(review)))
        #orig_data.set_value(i,'reviews', review + ('\x03' * (maxlength - len(review))))

    # this is in-place so nothing is returned
    return

def batchGenerator(train_data,batchSize, validate = False):
    # this function randomly samples a minibatch out of all the data
    ind = (len(train_data) * 8) // 10
    if(validate):
        train_batch = train_data.iloc[ind:].sample(n = batchSize)
    else:
        train_batch = train_data.iloc[:ind].sample(n = batchSize)
    train_batch.reset_index(drop=True, inplace=True)

    # pad EOS in the end to make same dimension
    pad_data(train_batch)

    return train_batch

def sentencify(positions):
    reviews = [""] * len(positions)
    # convert one hot encoding to strings
    for i in range(len(positions)):
        for j in range(len(positions[i])):
            reviews[i] += pos2char(positions[i][j][0])
    return reviews

def wordify(reviews, ref = False):
    # split string into words
    for i in range(len(reviews)):
        reviews[i] = reviews[i].split()
#         reviews[i] = [x.strip('.,!?@\x03') for x in reviews[i]]
        reviews[i] = [x.strip(string.punctuation+'\x03\x02\x00') for x in reviews[i]]
        if (ref):
            reviews[i] = [reviews[i]]

    return reviews


def train(model, cfg, data):
    # Train the model!
    torch.manual_seed(1)

    # What is train
    if (not cfg['train']):
        print("Set config[\'train\'] = True to train the model")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = cfg['learning_rate'])

    if cfg['cuda']:
        computing_device = torch.device("cuda")
#         criterion.cuda()
    else:
        computing_device = torch.device("cpu")

    # keep track of things
    train_losses = []
    valid_losses = []
#     blue_scores = np.zeros(cfg['epochs'])

    for epoch in range(cfg['epochs']):

        # for ever epoch, run review on all training data
        for batch_round in range(len(data) // cfg['batch_size']):

            # get train data from minibatch
            train_batch = batchGenerator(data,cfg['batch_size'])
            X_train, y_train = process_train_data(train_batch)
            del train_batch
            # put the data onto GPU if possible
            X_train = torch.from_numpy(X_train).float().to(computing_device)
            y_train = torch.from_numpy(y_train).long().to(computing_device)

            # find dimensions
            batchSize = len(y_train)

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            optimizer.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden(batchSize)

            # Step 3. Run our forward pass.
            outputs = model(X_train)

            del X_train

            # Step 4. Compute the loss, gradients, and update the parameters by
            sequenceSize = len(y_train[0])

            loss = criterion(outputs.view(batchSize * sequenceSize,-1), y_train.view(batchSize * sequenceSize))

            del y_train, outputs

            loss.backward()
            optimizer.step()

            # save the model to disk
            if (batch_round > 500 and batch_round % 500 == 0):
                torch.save(model.state_dict(), "./model_cache_at" + str(epoch) + "_" + str(batch_round))

            # record test and validation loss for every 50 batches
            if (batch_round % 200 == 0):

                print("Epoch: " + str(epoch) + ",Batch: " + str(batch_round) + " at: " + str(datetime.datetime.now()))

                # training error
                train_losses.append(loss.item())
                print("\tTrain losses:" + str(loss.item()))
                del loss

                # sample data from a different part of training data
                valid_batch = batchGenerator(data,cfg['batch_size'], validate = True )
                # get validate data
                X_valid, y_valid = process_train_data(valid_batch)
                del valid_batch
                X_valid = torch.from_numpy(X_valid).float().to(computing_device)
                y_valid = torch.from_numpy(y_valid).long().to(computing_device)

                # get dimensions
                batchSize = len(y_valid)
                sequenceSize = len(y_valid[0])

                model.hidden = model.init_hidden(batchSize)
                outputs = model(X_valid)

                del X_valid
                # validation error
                loss = criterion(outputs.view(batchSize * sequenceSize,-1)
                                 , y_valid.view(batchSize * sequenceSize))

                del y_valid, outputs, batchSize, sequenceSize
                valid_losses.append(loss.item())
                print("\tValidate losses:" + str(loss.item()))
                del loss

    print("Done")
    print("Training Loss: " + str(train_losses))
    print("Validate Loss: " + str(valid_losses))


def generate(model, X_test, cfg):
    # Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.

    if (cfg['train']):
        print("Set config[\'train\'] = False to generate text")
        return

    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")

    batchSize = len(X_test)
    reviews = [""] * batchSize

     # prepare the model
    model.zero_grad()
    # each time we are only generating one character
    model.hidden = model.init_hidden(batchSize)

    # keep track of the maximum length
    count = cfg['max_len']

    # let the model generate next characters
    while(count > 0):

        # get the next output from the model
        output = model(torch.from_numpy(X_test).float().to(computing_device))

        # if use temperature when testing
        if (not cfg['train']):
            output /= cfg['gen_temp']

        # find probability of each character
        output = F.softmax(output,dim=2)

        # check if all are done
        eoses = 0

        for i in range(batchSize):
            probs = output[i][0]
            # find the character
            pos = torch.multinomial(probs,1)

            if (pos == 96):
                eoses += 1
            # add current char to the review
            reviews[i] += pos2char(pos)

            # update X_test
            oh = np.zeros(98)
            oh[pos] = 1
            X_test[i][0][-98:] = oh
            del oh

        # if all are done
        if (eoses == batchSize):
            break

        count -= 1

    return reviews


def save_to_file(reviews, fname):
    # Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.

    with open(fname,'w') as f:
        for line in reviews:
            line = line.strip("\x02\x03")
            f.write(line.split("\x03")[0] + '\n')

def process_train_label(data):
    # Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).

    labels = np.zeros([len(data),len(data['reviews'][0]),1])
    review_num = 0
    for line in iter(data.iterrows()):

        # tensor for style
        s = np.zeros(203)
        s[encode_style[line[1]['styles']]] = 1

        # tensor for rating
        s[104] = line[1]['ratings']

        char_num = 0
        # encoded review characters
        for c in line[1]['reviews']:

            # encode label
            if c != '\x02':
                labels[review_num][char_num-1] = [char2pos(c)]

            char_num += 1

        # put one end in the end of label
        labels[review_num][char_num-1] = char2pos('\x03')

        review_num += 1

    return labels
