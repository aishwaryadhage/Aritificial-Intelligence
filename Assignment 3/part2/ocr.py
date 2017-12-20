#!/usr/bin/python
# coding=utf-8

#
# ./ocr.py : Perform optical character recognition, usage:
# ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Aishwaray Dhage(adhage), Ninaad Joshi(ninjoshi), Siddharth Pathak(sidpath)
# (based on skeleton code by D. Crandall, Oct 2017)
#
# For predicting characters, HMM is very useful, as the characters depend on the context and the characters which come before
# and after the current character i.e what the character2 in the sentence is, depends on character1 and character3 of the sentence.
# HMM handles this nicely, as it takes into consideration the transistion probability between characters based on the HMM model.
# Each hidden node in the HMM is a character, and observed node is the pixel number/position.

# We have used Forward-Backward Algorithm to implement Variable Elimination
# There are three main parts in this algorithm-forward algorithm which calculates the probability of current state given the evidence of the history  ie P(Wk|S1:k).
# Backward Algorithm which calculates the same probabilities as forward but in backwards ie P(Sk+1:n|Wk).
# Third step is to calculate posterior marginals of hidden states i.e  P(Wk|S)=P(Sk+1:n|Wk)*P(Wk|S1:k) where n is the number of states and k is the current state.
# Observations and Difficulties faced-
# In this algorithm we have considered punishing factor as 10**-3  for emission probability and transition probability if that tag is not present in the dictionary.
# We observed change in accuracy as we change the punishing factor,the most appropriate one we found for our algorithm is 10**-3.
# When we were generating forward and backward values of each tag in this algorithm, we found that our algorithm goes into underflow
# To avoid underflow we have normalized forward and backward probability values(alpha and beta values) i.e. divided all the probabilities values by maximum value among all values
# Prof.David Crandall suggested above mentioned normalization to avoid underflow.
# Discussed about the working of forward-backward algorithm with Praneta Paithankar
# Reference-https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

# Viterbi uses emission probability, initial probability and transistion probability.
# In Viterbi algorithm, if the transisiton probability is not present then I punished it with factor of 10**-6, and
# if emission probability is not present then punish it by a factor of 10**-6. 
# We decided these values after experimentation.
# We implemented Viterbi using a bottom-up dynamic approach. It calculates the values of first word and then uses them for the second one, and so on. 
# Using these values, Viterbi calculates MAP, and gives sequence of characters for the input sentence.
# Discussed how Viterbi algorithm works with Akshay Naik.

"""
Calculation of Initial, Transition and Emission Probabilities from Training
Data:

Cleaning of Training Data :

The training sequence is assumed to be containing the below mentioned
characters and special characters and will discard any other special characters
if encountered in the training data using a regular expression as mentioned in
the code.

Initial Probabilities :

The initial probabilities are calculated by using the training data text file by
considering the number of occurrences of each alphabet in the given training
sequence. The probability is calculated by taking the occurrences of the
considered letter and dividing it by the total number of initial characters.
The counts are stored in the 'initial_count_table' dictionary and the
probabilities are stored in the 'initial_probability_table' dictionary.

Transition Probabilities :

The transition probabilities are calculated by considering successive letter
pairs from the training text data by getting the count of all such pairs in the
text file. These counts are stored in the 'succeeding_character_count_table'
dictionary and the calculated probabilities are stored in the
'transition_probability_table' dictionary.

Emission Probabilities :

The emission probabilities are calculated by considering each pixel's position
and value (1 = Black pixel, 0 = White pixel) in the training data and the
testing data. The probabilities are multiplied by weighted constants which were
calculated by considering different constants according the values for each
pixel in the corresponding data sets for the given testing data.

Another calculation was performed to calculate the average black pixel counts
for each block in the training and the testing data sets. The average for the
testing and training image sequence have been calculated to implement the
relative conditions of : very sparse, sparse and dense blocks (relative to the
provided training data).

The important conditions for calculating relative the very sparse, sparse and
the dense blocks are mentioned below :

(An assumption has been made while calculating the average pixels for the data
that the pixel scattering is uniform across the whole sequence.)

1. Sparse image sequence condition :

The sparse image sequence condition is triggered if the average black pixel
count of the testing sequence is less than or equal to half of the average black
pixel count of the training sequence.

2. Average image sequence condition :

The average image sequence condition is triggered if the average black pixel
count of the testing sequence is more than half of the average black pixel
count but less than or equal to the average black pixel count of the training
sequence.

3. Dense image sequence condition :

The dense image sequence condition is triggered if the average black pixel count
if the testing sequence is more than the average black pixel count of the
training sequence.


The probabilities for individual pixels of the training and testing sequences
have been calculated using weights for their values and positions in the
training data.

(An assumption has been made that the training data provided will be less noisy
or more ideal for considering the multiplying constants)

The important conditions for using the specified constants are mentioned below :


1. matching pixels and testing pixel color black :

If the pixels match and the pixel is black then there is a high probability that
the testing alphabet matches the training alphabet. Thus, a higher probability
value is multiplied to the emission probability of the corresponding alphabet.

2. matching pixels and testing pixel color white :

If the pixels match and the pixel is white then there is some probability
that the testing alphabet matches the training alphabet. Thus, a medium
probability value is multiplied to the emission probability of the corresponding
alphabet.

3. non matching pixels and testing pixel color white :

If the pixels do not match and the training pixel is black then there is some
probability that the testing alphabet matches the training alphabet as the
testing alphabet could be incomplete. Thus, a medium probability value is
multiplied to the emission probability of the corresponding alphabet.

4. non matching pixels and testing pixel color white :

If the pixels do not match and the training pixel is white then there is very
less probability that the testing alphabet matches the training alphabet as the
pixel in the testing alphabet could be noise. Thus, a low probability value is
multiplied to the emission probability of the corresponding alphabet.

A key assumption is that the black pixels are of high regard in a sparse as
the white pixels are more in number, thus considerably higher probabilities for
black pixels in a sparse block are beneficial for prediction. Same scenario can
be considered for the average condition. The difference arises when the pixels
don't match and the training pixel is black, which can be considered as
incomplete, thus implying that the pixel may be present in the actual alphabet.
Thus, two different values have been multiplied to calculate the probabilities
of the alphabets.

In case of the dense condition, the matching black as well as white pixels are
given higher importance while calculating the probabilities as the unmatched
black pixels in the testing blocks have higher chances of being noise. Thus,
if the pixels are non matching and the training pixel is white, then the testing
pixel has a high probability of being a noise. Thus, lower probabilities are
considered for such pixels.

"""

import copy
import math
import re
import sys
from PIL import Image

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

# load the letters from the file images
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH,
                       CHARACTER_WIDTH):
        result += [[1 if px[x, y] < 1 else 0 for x in
                    range(x_beg, x_beg + CHARACTER_WIDTH) for y in
                    range(0, CHARACTER_HEIGHT)]]
    return result

# load training letters
def load_training_letters(fname):
    TRAIN_LETTERS = \
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()," \
                    ".-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in
            range(0, len(TRAIN_LETTERS))}

# read data from the training text file to calculate probabilities
def read_data(fname):
    file = open(fname, 'r')
    total_character_count = 0
    total_initial_characters = 0
    for line in file:
        # splitting the training data because the bc.train file from the part1
        # of the assignment was selected as training data. If the split is
        # removed, a normal text file can be used for training the model
        # without using the splits.
        data = [w for w in line.split()][0::2]
        
        # if the training text file does't contain the POS tags,
        # please use the following code
        # data = [w for w in line.split()]
        
        data = " ".join(data)
        # using regular expression to remove special characters which are not to
        # be considered for prediction
        data = re.sub("[^A-Za-z0-9.,\-\(\)!? \"']", "", data)
        data = list(data)  # make list from string
        len_data = len(data)
        
        total_character_count += len_data
        total_initial_characters += 1
        
        if len_data > 0:
            for index in range(0, len_data - 1):
                # calculate the succeeding character and transition character counts
                succeeding_character_count_table[data[index]] = succeeding_character_count_table.get(data[index], 0) + 1
                transition_probability_table[data[index], data[index + 1]] = transition_probability_table.get((data[index], data[index + 1]), 0) + 1
            # calculate the initial character counts
            initial_count_table[data[0]] = initial_count_table.get(data[0], 0) + 1
    
    # calculate initial probability
    for initial in initial_count_table:
        initial_probability_table[initial] = (initial_count_table[initial] * 1.0)/total_initial_characters
    
    # calculate transition probability
    for k, v in transition_probability_table.items():
        transition_probability_table[k] = (transition_probability_table[k] * 1.0)/succeeding_character_count_table[k[0]]

    # calculate emission probability
    calculate_emission_probability(test_letters)

# function to calculate the emission probabilities
def calculate_emission_probability(test_letters):
    train_sum = sum([sum(train_letters[train_letter]) for train_letter in train_letters])
    test_sum = sum([sum(test_letter) for test_letter in test_letters])
    
    train_avg = train_sum / len(train_letters)
    test_avg = test_sum / len(test_letters)
    sparse_condition = train_avg / 2
    dense_condition = train_avg
    # sparse testing condition
    if 0 <= test_avg <= sparse_condition:
        for i, block in enumerate(test_letters):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.99
                        else:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.5
                    else:
                        if not train_letters[train_letter][j]:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.01
                        else:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.3
    # average training condition
    elif sparse_condition < test_avg <= dense_condition:
        for i, block in enumerate(test_letters):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.99
                        else:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.5
                    else:
                        if not train_letters[train_letter][j]:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.01
                        else:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.2
    # dense training condition
    else:
        for i, block in enumerate(test_letters):
            emission_probability_table[i] = {}
            for train_letter in train_letters:
                for j, pixel in enumerate(block):
                    if pixel == train_letters[train_letter][j]:
                        if pixel:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.8
                        else:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.7
                    else:
                        if not train_letters[train_letter][j]:
                            # training pixel is white
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.2
                        else:
                            # training pixel is black
                            emission_probability_table[i][train_letter] = emission_probability_table[i].get(train_letter, 1) * 0.3
    return emission_probability_table

# naive bayes rule where prior has been assumed to be constant
def hmm_simplified(test_letters):
    result = []
    for k, v in emission_probability_table.items():
        result.append(max(v, key = v.get))
    return result

# Viterbi algorithm
def hmm_viterbi(test_letters):
    viterbi = {}
    temp = []
    viterbi_temp = {}
    for i, word in enumerate(test_letters):
        if i == 0:
            for j in train_letters:
                viterbi[j] = [-math.log(initial_probability_table.get(j, 10 ** -6)) + (-math.log(emission_probability_table[i].get(j, (10 ** -6)))), [j]]
        else:
            for j in train_letters:
                for k in viterbi:
                    temp.append([(viterbi[k][0] + (-math.log(transition_probability_table.get((k, j), 10 ** -6)))), viterbi[k][1] + [j]])
                x = min(temp, key=lambda y: y[0])
                viterbi_temp[j] = [x[0] + (-math.log(emission_probability_table[i].get(j, 10 ** -6))), x[1]]
                del temp[:]
            viterbi.clear()
            viterbi = copy.deepcopy(viterbi_temp)
            viterbi_temp.clear()
    return min(viterbi.values(), key=lambda y: y[0])[1]

# Variable elimination algorithm
def hmm_ve(test_letters):
    alpha_prev = {}
    alpha = []
    # forward algorithm
    for i, word in enumerate(test_letters):
        alpha_curr = {}
        
        for pos in train_letters:
            if i == 0:
                alpha_curr_sum = (initial_probability_table.get(pos, 10 ** -3))
            else:
                alpha_curr_sum = sum(
                    (transition_probability_table.get((pos1, pos), 0.0001)) * (alpha_prev[pos1]) for pos1 in train_letters)
            alpha_curr[pos] = (alpha_curr_sum * (emission_probability_table[i].get(pos, 10 ** -3)))
        temp_max = max(alpha_curr.values())
        new_alpha = {key: val / temp_max for key, val in alpha_curr.iteritems()}
        alpha.append(new_alpha)
        alpha_prev = new_alpha
    # backward algorithm
    beta = []
    b_prev = {}
    prev_letter = len(test_letters) - 1
    for i, word in enumerate(test_letters[::-1]):
        beta_curr = {}
        for pos in train_letters:
            if i == 0:
                beta_curr[pos] = 1
            else:
                beta_curr[pos] = sum((transition_probability_table.get(
                    (pos, pos1), 10 ** -3)) * (emission_probability_table[prev_letter].get(pos1, 10 ** -3)) * (b_prev[pos1]) for pos1 in train_letters)
        temp_max = max(beta_curr.values())
        new_beta = {key: val / temp_max for key, val in beta_curr.iteritems()}
        beta.append(new_beta)
        b_prev = new_beta
        prev_letter -= 1
    # merge-forward, backward algorithm
    posterior = []
    bfw_dict = {}
    sentence_length = len(test_letters)
    j = sentence_length - 1
    for i in range(sentence_length):
        for pos in train_letters:
            bfw_dict[pos] = (beta[j][pos] * alpha[i][pos])
        posterior.append(max(bfw_dict, key=bfw_dict.get))
        bfw_dict = {}
        j -= 1
    return posterior

# loading the training and testing letters

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)


# dictionaries used to store the probabilities and counts

# initial count and probabilities
initial_probability_table = {}
initial_count_table = {}

# transition count and probabilities
succeeding_character_count_table = {}
transition_probability_table = {}

# emission probabilities
emission_probability_table = {}

try:
    read_data(train_txt_fname)
    print " Simple: " + "".join(hmm_simplified(test_letters))
    print " HMM VE: " + "".join(hmm_ve(test_letters))
    print "HMM MAP: " + "".join(hmm_viterbi(test_letters))

except IOError:
    print "File not found :("
    exit(0)
