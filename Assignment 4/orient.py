#!/usr/bin/python2.7
import sys
import heapq
import time
import math
import numpy as np

# Report

# Q1. In your report, present neatly-organized tables or graphs showing
# classification accuracies and running times as a function of the parameters you choose.

# KNN

# K 		Accuracy        Approx Time(Min)
# 1         67.2322375398       25
# 7         69.5652173913       25
# 100       70.413573701        25
# sqrt(N)   71.1558854719       25

# Adaboost

# Weak classifier   # Accuracy  Approx time(Min)
# Blue and Red      59          3
# Red               24          3

# Neural Net

# Perceptrons   Learning Rate	Running Time(Min)  Accuracy    Epoch
# 7             0.29            0.40                69.88       1000
# 8             0.29            0.45                70.47       1000
# 8             0.29            0.6                 70.33       5000
# 16            0.29            0.56                69.034      1000
# 16            0.25            0.45                67.88       1000
# 16            0.3             0.67                69.3        1000
# 25            0.29            0.765               68.822      1000
# 26            0.29            1.34                70.94       5000
# 26            0.29            4.2                 72.11       10000
# 30            0.28            4.67                70.30       10000

# Tried different learning rates between 0.1 and 0.5. 0.29 happened to give better results.
# Epoch plays an important role in bringing the weights to get accurate answers as we increased number of epochs.


# Q2. Which classifiers and which parameters would you recommend to a potential client?
# Recommendation will vary depending on the needs of the client. If the client needs faster output but with lower
# then Adaboost would work fine.
# If the client doesn't care about the training time or is ready to ignore the training time required, then Neural
# Net(with 26 perceptrons and 10000 epoch) would be the best classifier. It gives the highest accuracy in the least
# amount of time, but requires time to get trained. The accuracy can further be increased with more training data
# with more variation


# Q3. How does performance vary depending on the training dataset size, i.e. if you use just a fraction of the training data?

# KNN

# Train data    Accuracy
# 100           63.3085896076
# 1000          66.5959703075
# 10000         70.7317073171
# 36976         71.1558854719

# Adaboost

# Train Data    Accuracy
# 100           59.1728525981
# 1000          59.1728525981
# 10000         59.1728525981
# 36976         59.1728525981

# Neural Net

# Train Data    Accuracy
# 100           40.19
# 1000          64.475
# 10000         69.459
# 36976         72.216


# Q4. Show a few sample images that were classified correctly and incorrectly. Do you see any patterns to the errors?
# The training images follow a pattern. Majority of the outdoor images have blue sky at the top and darker shades of
# color like green/brown at the bottom. If the test images follow this pattern then it gets classified correctly.
# The images which are not classified correctly are the ones which don't follow this pattern. e.g sky at night image.
# The image has dark color at the top as well as the bottom.
# Using pixels as image features, classifying images becomes difficult for images which don't follow such pattern.


# Sample Images correctly classified:

# 30853946
# 50809972
# 131087914
# 172145755
# 8605521701

# Sample Images incorrectly classified:

# 10351347465 -- Black and White, hard to detect sky/water/ground
# 10684428096 -- Image doesn't follow the regular pattern of Sky(blue) at top, and ground(brown) at bottom.
# 12799838924 -- Image has brown color on top and bottom.
# 14284767631 -- Has green at the top and not at the bottom
# 14645715459 -- Sky at night. Dark at top as well as bottom.


# Notes:

# References:
# https://www.youtube.com/watch?v=262XJe2I2D0&t=1731s
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.loadtxt.html
# http://akuederle.com/create-numpy-array-with-for-loop
# Handle overflow in sigmoid - https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation
# Discussed with Akshay Naik about the possible parameters to choose.
# Discussed with Praneta Painthankar about Adaboost algorithm.


# Report End


# KNN functions begin here
# Model the KNN data and write into the model file
def train_knn(train_file, model_file):
    model_file_handler = open(model_file, "w")
    k = 0
    for line in open(train_file):
        k += 1
    model_file_handler.write(str(k) + "\n")
    for line in open(train_file):
        split_line = line.split(' ')
        model_file_handler.write(split_line[1] + " " + " ".join(split_line[2:]))


# Read from the model file and store in the train_data list
def read_model_knn(train_file):
    train_data = []
    k = 0
    for line_number, line in enumerate(open(train_file)):
        if line_number == 0:
            k = int(line)
        else:
            split_line = line.split(' ')
            train_data.append([(split_line[0]), map(int, split_line[1:])])
    return k, train_data


# Classify images using KNN classifier
def test_knn(test_file, model_file):
    knn_distances = []
    result = []
    total_test_images, total_correct_classified = 0, 0
    k, train_data = read_model_knn(model_file)
    k = int(k ** (1.0 / 2))
    output_file_handler = open("output.txt", "w")
    for line_number, line in enumerate(open(test_file)):
        split_line = line.split(' ')
        test_data = [int(split_line[1]), map(int, split_line[2:])]
        knn_vote = {}
        if line_number > 100:
            pass
        # break
        for train_image in train_data:
            temp_distance = 0
            for i, pixel in enumerate(train_image[1]):
                temp_distance += (test_data[1][i] - pixel) ** 2
            temp_distance = temp_distance ** (1 / 2.0)
            heapq.heappush(knn_distances, [temp_distance, train_image[0]])
        for d in [heapq.heappop(knn_distances) for _ in range(0, k)]:
            knn_vote[d[1]] = knn_vote.get(d[1], 0) + 1
        result.append(max(knn_vote, key=knn_vote.get))
        del knn_distances[:]
        knn_vote.clear()
        total_test_images += 1
        if int(result[-1]) == int(split_line[1]):
            total_correct_classified += 1
        output_file_handler.write(split_line[0] + " " + result[-1] + "\n")
    print "Accuracy is ", (total_correct_classified * 1.0 / total_test_images) * 100, "%"


# KNN Functions end here

# Adaboost functions begin here

# Weak classifier for Adaboost.
# Blue shade is present at the top of the image.
def blue_classifier(examples):
    blue_areas = {}
    result = {}
    for image in examples:
        blue_areas["0"] = sum([b for b in image[2][2:72:3]])
        blue_areas["180"] = sum([b for b in image[2][122:192:3]])
        blue_areas["270"] = sum([b for row in range(0, 8) for b in image[2][2 + (row * 24):9 + (row * 24):3]])
        blue_areas["90"] = sum([b for row in range(0, 8) for b in image[2][17 + (row * 24):24 + (row * 24):3]])
        classified_orientation = (max(blue_areas, key=blue_areas.get))
        result[image[0:2]] = [image[1], classified_orientation]
    return result


# Weak classifier for Adaboost.
# Red shade is present at the top of the image.
def red_classifier(examples):
    red_areas = {}
    result = {}
    for image in examples:
        red_areas["180"] = sum([b for b in image[2][0:72:3]])
        red_areas["0"] = sum([b for b in image[2][120:192:3]])
        red_areas["90"] = sum([b for row in range(0, 8) for b in image[2][(row * 24):9 + (row * 24):3]])
        red_areas["270"] = sum([b for row in range(0, 8) for b in image[2][15 + (row * 24):24 + (row * 24):3]])
        classified_orientation = (max(red_areas, key=red_areas.get))
        result[image[0:2]] = [image[1], classified_orientation]
    return result


def adaboost(examples, hypotheses_weights, orientation_first, orientation_second):
    weights = {}
    hypotheses = {}
    hypotheses["blue"] = {}
    hypotheses["red"] = {}
    color = ""
    # initialize weights by 1/N
    for image in examples:
        weights[image[0:2]] = 1.0 / len(examples)

    for k in range(2):
        total_weight = 0
        if k == 0:
            color = "blue"
            hypotheses["blue"] = blue_classifier(examples)
        elif k == 1:
            color = "red"
            hypotheses["red"] = red_classifier(examples)
        error = 0
        # calculating errors
        for key in hypotheses[color]:
            if hypotheses[color][key][0] != hypotheses[color][key][1]:
                error += weights[key]
        error_percentage = float(error) / (1 - error)
        for key in hypotheses[color]:
            if hypotheses[color][key][0] == hypotheses[color][key][1]:
                weights[key] = weights[key] * error_percentage
                total_weight += weights[key]
        for key in weights:
            weights[key] /= total_weight

        hypotheses_weights[color][orientation_first] += math.log(abs(1 / error_percentage))
        hypotheses_weights[color][orientation_second] += math.log(abs(1 / error_percentage))


def train_adaboost():
    image_orient_090 = []
    image_orient_90180 = []
    image_orient_90270 = []
    image_orient_0180 = []
    image_orient_0270 = []
    image_orient_180270 = []

    for line_number, line in enumerate(open(input_file)):
        split_line = line.split(" ")
        image = split_line[0], split_line[1], map(int, split_line[2:])

        if image[1] == "0" or image[1] == "90":
            image_orient_090.append(image)
        if image[1] == "90" or image[1] == "180":
            image_orient_90180.append(image)
        if image[1] == "180" or image[1] == "270":
            image_orient_180270.append(image)
        if image[1] == "0" or image[1] == "270":
            image_orient_0270.append(image)
        if image[1] == "0" or image[1] == "180":
            image_orient_0180.append(image)
        if image[1] == "270" or image[1] == "90":
            image_orient_90270.append(image)

    hypotheses_weights = {"blue": {}, "red": {}}

    for key in hypotheses_weights:
        hypotheses_weights[key] = {0: 0, 90: 0, 180: 0, 270: 0}

    adaboost(image_orient_090, hypotheses_weights, 0, 90)
    adaboost(image_orient_90180, hypotheses_weights, 90, 180)
    adaboost(image_orient_90270, hypotheses_weights, 90, 270)
    adaboost(image_orient_0180, hypotheses_weights, 0, 180)
    adaboost(image_orient_0270, hypotheses_weights, 0, 270)
    adaboost(image_orient_180270, hypotheses_weights, 180, 270)

    model_file_handler = open(model_file, "w")
    for key in hypotheses_weights:
        model_file_handler.write(key)
        for orientation in hypotheses_weights[key]:
            model_file_handler.write(" {0} {1}".format(orientation, hypotheses_weights[key][orientation]))
        model_file_handler.write("\n")


def test_adaboost():
    model_file_handler = open(model_file, "r")
    model = {}
    test_images = []
    output_file_handler = open("output.txt", "w")
    for line in model_file_handler.readlines():
        color_line = line.split()
        temp_dict = {}
        for k in range(1, 9, 2):
            temp_dict[color_line[k]] = color_line[k + 1]
        model[color_line[0]] = temp_dict

    for line_number, line in enumerate(open(input_file)):
        split_line = line.split(" ")
        image = split_line[0], split_line[1], map(int, split_line[2:])
        test_images.append(image)
    matching = 0
    for image in test_images:
        votes = {'0': 0, '90': 0, '180': 0, '270': 0}
        blue_result = blue_classifier([image])[image[0:2]][1]  # for the image blue
        votes[blue_result] += float(model["blue"][blue_result])
        red_result = red_classifier([image])[image[0:2]][1]  # for the image red
        votes[red_result] += float(model["red"][red_result])
        result = max(votes, key=votes.get)
        if image[1] == result:
            matching += 1
        output_file_handler.write(image[0] + " " + result + "\n")
    print "Accuracy is ", float(matching) / len(test_images) * 100, "%"

# Adaboost Functions end here


# Neural Net Functions begin here
# Sigmoid
def g_sigmoid(inj):
    inj = np.clip(inj, -500, 500)
    return 1 / (1 + np.exp(-inj))


# Derivative of Sigmoid
def g_derivative_sigmoid(inj):
    return inj * (1 - inj)


# back propagation algorithm
def training_backpropagation(inputs, outputs, len_data):
    # assign initial random weights
    np.random.seed(1)
    wij_h1 = ((np.random.uniform(-0.5, 0.5, (26, 192))))
    b1 = ((np.random.uniform(-0.5, 0.5, (26))))
    wij_op = (np.random.uniform(-0.5, 0.5, (4, 26)))
    b2 = ((np.random.uniform(-0.5, 0.5, (4))))
    # training
    for j in xrange(10000):
        # forward propagation
        hidden_layer1 = g_sigmoid(((np.dot(wij_h1, inputs)).T + b1).T)
        output_layer = g_sigmoid(((np.dot(wij_op, hidden_layer1)).T + b2).T)
        # backpropagation
        output_error = outputs - output_layer
        # calculate deltas
        output_delta = output_error * g_derivative_sigmoid(output_layer)
        hidden_delta_1 = (np.dot(wij_op.T, output_delta) * g_derivative_sigmoid(hidden_layer1))
        # update weights
        length = len_data
        if (j % 1000 == 0): length = length * 2
        b2 += (0.29 / length) * np.sum(output_delta, axis=1)
        wij_op += (0.29 / length) * np.dot(output_delta, hidden_layer1.T)
        b1 += (0.29 / length) * np.sum(hidden_delta_1, axis=1)
        wij_h1 += (0.29 / length) * np.dot(hidden_delta_1, inputs.T)
    return wij_h1, wij_op, b1, b2


def train_nnet(train_file, model_file):
    start_time = time.time()
    with open(train_file) as f:
        lines = f.readlines()
        # get inputs in numpy array
    x = np.loadtxt(train_file, delimiter=' ', usecols=range(2, 194), unpack=True)
    len_data = (float)(x.shape[1])
    arr = []
    # put outputs in numpy array
    for i, line in enumerate(lines):
        orient = line.split(' ')[1]
        if orient == '0':
            arr.append([1, 0, 0, 0])
        elif orient == '90':
            arr.append([0, 1, 0, 0])
        elif orient == '180':
            arr.append([0, 0, 1, 0])
        elif orient == '270':
            arr.append([0, 0, 0, 1])
    y = np.array(arr)
    inputs = x
    outputs = y.T
    # backprog=pagation
    wij_h1, wij_op, b1, b2 = training_backpropagation(inputs, outputs, len_data)
    # write weights into model file
    with open(model_file, 'w') as f:
        f.write('wij_h1\n')
        np.savetxt(f, wij_h1, delimiter=' ', fmt="%f")
        f.write('wij_op\n')
        np.savetxt(f, wij_op, delimiter=' ', fmt="%f")
        f.write('b1\n')
        np.savetxt(f, b1, delimiter=' ', fmt="%f")
        f.write('b2\n')
        np.savetxt(f, b2, delimiter=' ', fmt="%f")
    print("Time taken to train:", (time.time() - start_time) / 60.0)


# Classify images using Neural Network
def test_nnet(input_file, model_file):
    with open(input_file) as reading_input_file:
        lines_ip_file = reading_input_file.readlines()
    test_input = np.loadtxt(input_file, delimiter=' ', usecols=range(2, 194), unpack=True)
    len_test_data = (float)(test_input.shape[1])
    # read test file
    arr = []
    for i, line in enumerate(lines_ip_file):
        orient = line.split(' ')[1]
        if orient == '0':
            arr.append([1, 0, 0, 0])
        elif orient == '90':
            arr.append([0, 1, 0, 0])
        elif orient == '180':
            arr.append([0, 0, 1, 0])
        elif orient == '270':
            arr.append([0, 0, 0, 1])
    test_output = (np.array(arr))
    # read the trained model from model_file
    wij_h1l, wij_opl, b1l, b2l = [], [], [], []
    with open(model_file) as reading_model_file:
        lines_m_file = reading_model_file.readlines()
    for i, line in enumerate(lines_m_file):
        if i >= 1 and i <= 26:
            wij_h1l.append(map(float, line.split(' ')))
        if i >= 28 and i <= 31:
            wij_opl.append(map(float, line.split(' ')))
        if i >= 33 and i <= 58:
            b1l.append(map(float, line.split(' ')))
        if i >= 60 and i <= 63:
            b2l.append(map(float, line.split(' ')))
    wij_h1 = np.array(wij_h1l)
    wij_op = np.array(wij_opl)
    b1 = (np.array(b1l)).reshape(1, 26)
    b2 = (np.array(b2l)).reshape(1, 4)
    # forward propagation
    hidden_layer_1 = g_sigmoid(((np.dot(wij_h1, test_input)).T + b1).T)
    output_layer_test = g_sigmoid(((np.dot(wij_op, hidden_layer_1)).T + b2).T)
    # assign 1 and 0 in predicted output
    predicted_output = output_layer_test.T
    for row in predicted_output:
        threshold = np.max(row)
        row[row < threshold] = 0
        row[row >= threshold] = 1
        # get no. of hits to calculate %accuracy
    hit = 0
    for i, row1 in enumerate(test_output):
        if np.array_equal(test_output[i], predicted_output[i]):
            hit += 1
            # put images name and predicted orientation in file
    with open(input_file) as f_outputs:
        line_output = f_outputs.readlines()
    with open('output.txt', 'w') as f1:
        for row, line in zip(predicted_output, line_output):
            orient = ((np.argmax(row))) * 90
            f1.write('%s %s \n' % ((line.split(' ')[0]), (int)(orient)))
    print "Accuracy is ", (hit / len_test_data) * 100, "%"

# End Neural Functions here


program_type, input_file, model_file, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

if program_type == "train":
    if model == "nearest":
        train_knn(input_file, model_file)
    elif model == "adaboost":
        print "Training Adaboost..."
        train_adaboost()
    elif model == "nnet" or model == "best":
        print "Training Neural Net...."
        train_nnet(input_file, model_file)
    else:
        print "Enter a valid model"

if program_type == "test":
    if model == "nearest":
        print "Classifying images using KNN classifier...."
        test_knn(input_file, model_file)
    elif model == "adaboost":
        print "Classifying Images using Adaboost..."
        test_adaboost()
        pass
    elif model == "nnet" or model == "best":
        print "Classifying images using NNET classifier...."
        test_nnet(input_file, model_file)
    else:
        print "Enter a Valid model"
