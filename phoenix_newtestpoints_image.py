"""
This file is new testing file. It uses image data anpoints.

author: naserjawas
date  : 26 May 2023
"""

import os
import glob
import math
import pickle
import numpy as np
import pandas as pd
import cv2 as cv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import RBF

from sktime.dists_kernels import AggrDist
from sktime.classification.kernel_based import TimeSeriesSVC
# from sktime.classification.kernel_based import RocketClassifier
# from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.compose import ColumnConcatenator


def normalise_points(oripoints):
    scaler = MinMaxScaler()

    newpoints = []
    for p in oripoints:
        newp = []
        for b in p:
            newb =  np.array(b).reshape(-1, 1)
            newb = scaler.fit_transform(newb)
            newb = newb.reshape(1, -1)[0].tolist()
            newp.append(newb)
        newpoints.append(newp)

    return newpoints

def convertToUnequalDF(setpts):
    newsetpts = []
    for t in setpts:
        newt = np.array(t)
        newt = np.transpose(newt)
        d = []
        for t2 in newt:
            newt2 = pd.Series(t2)
            d.append(newt2)
        # newd = pd.Series(d)
        newsetpts.append(d)

    ret_df = pd.DataFrame(data=newsetpts)

    return ret_df

def convertToEqualNP(setpts, maxlen, glosslength, reshape=False):
    for i in range(len(setpts)):
        setpts[i] = zeroframepadding(setpts[i], maxlen, glosslength)

    ret_np = np.array(setpts)
    (setins, setrow, setcol) = ret_np.shape

    if reshape:
        ret_np = ret_np.reshape(-1, (setrow * setcol))
    else:
        new_ret_np = []
        for t in ret_np:
            newt = np.transpose(t)
            d = []
            for t2 in newt:
                newt2 = pd.Series(t2)
                d.append(newt2)
            # newd = pd.Series(d)
            new_ret_np.append(d)

        ret_np = pd.DataFrame(data=new_ret_np)

    return ret_np

def zeroframepadding(lstpts, maxlen, glosslength):
    lensingle = glosslength * glosslength * 3 * 5
    for i in range(len(lstpts)):
        if len(lstpts[i]) < lensingle:
            for j in range(len(lstpts[i]), lensingle):
                lstpts[i].append(0)
    zeroframe = [0] * lensingle
    for i in range(len(lstpts), maxlen):
        lstpts.append(zeroframe)

    return lstpts


if __name__ == "__main__":
    glossp = 50
    glosstype = 2
    glosslength = 10
    # drive = "./dataset/RWTHPHOENIXWeather2014"
    drive = "../dataset/RWTHPHOENIXWeather2014"
    glossdir = drive + f"/glosswithoutphase/gloss{glosstype}_files{glossp}p/"
    glossfile = []

    # load all fpoints filenames to a list variable
    print("load data...")
    for root, dirs, files in os.walk(glossdir):
        for name in dirs:
            dirname = os.path.join(root, name)
            # fpointsfile = dirname + f"/anpoints{glosslength}"
            fpointssearch = dirname + f"/anpoints{glosslength}_{glosstype}_{glossp}p*.png"
            fpointsfiles = glob.glob(fpointssearch)
            if len(fpointsfiles) > 0:
                glossfile.append(fpointssearch)

    glossfile = sorted(glossfile)
    print(f"glossfile: {len(glossfile)} anpoints image files found")

    # load pickle data into variable
    print("build dataset...")
    oldgloss = ""
    lengpoints = 0
    lenframes = []
    data = []
    points = []

    traindata = []
    testdata = []
    traintestratio = 0.75
    mindata = 10
    minlenframes = 5

    trainpts = []
    testpts = []
    trainlbl = []
    testlbl = []
    traincls = []
    testcls = []

    label = 0

    for g in glossfile:
        gdir = g.split("/")
        gfiles = sorted(glob.glob(g))
        gimgs = [cv.imread(gfile, cv.IMREAD_GRAYSCALE)
                 for gfile in gfiles]
        gpoints = [img.reshape(1, -1)[0].tolist()
                   for img in gimgs]
        lengpoints = len(gpoints)

        if oldgloss != gdir[-3] and oldgloss != "":
            lenframes = np.array([len(p) for p in points])
            lencleanframes = lenframes[lenframes > minlenframes].copy()

            # calculate mean and std dev
            if len(lencleanframes) > 0:
                mean = np.mean(lencleanframes)
                std = np.std(lencleanframes)
            else:
                mean = 0
                std = 0
            maxlenframes = mean + std

            # remove sample that has short and over-long frame sequence
            newpoints = []
            newdata = []
            for i, p in enumerate(points):
                if minlenframes < maxlenframes:
                    if len(p) > minlenframes and len(p) < maxlenframes:
                        newpoints.append(p)
                        newdata.append(oldgloss)
                else:
                    if len(p) > minlenframes:
                        newpoints.append(p)
                        newdata.append(oldgloss)
            points = newpoints
            data = newdata

            lencleanframes = np.array([len(p) for p in points])

            if len(data) >= mindata:
                split = math.ceil(len(data) * traintestratio)

                traindata = data[:split].copy()
                testdata = data[split:].copy()

                trainpoints = points[:split].copy()
                testpoints = points[split:].copy()

                trainpoints = normalise_points(trainpoints)
                testpoints = normalise_points(testpoints)

                if oldgloss == "si":
                    label = 0
                else:
                    label = 1

                trainlabel = [label for i in range(len(trainpoints))]
                testlabel = [label for i in range(len(testpoints))]

                trainclass = [oldgloss for i in range(len(trainpoints))]
                testclass = [oldgloss for i in range(len(testpoints))]

                trainpts += trainpoints
                testpts += testpoints

                trainlbl += trainlabel
                testlbl += testlabel

                traincls += trainclass
                testcls += testclass

                label += 1

            points = []
            data = []

        if lengpoints > 0:
            data.append(gdir[-3])
            points.append(gpoints)

        oldgloss = gdir[-3]

    # convert to equal dataset using numpy
    maxlen = 0
    minlen = 0
    allpts = trainpts + testpts
    for i in range(len(allpts)):
        if minlen == 0:
            minlen = len(allpts[i])
        if len(allpts[i]) > maxlen:
            maxlen = len(allpts[i])
        if len(allpts[i]) < minlen:
            minlen = len(allpts[i])
    print(f"maxlen: {maxlen}")
    print(f"minlen: {minlen}")
    X_train = convertToEqualNP(trainpts, maxlen, glosslength, False)
    X_test = convertToEqualNP(testpts, maxlen, glosslength, False)
    y_train = np.array(trainlbl)
    y_test = np.array(testlbl)
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # # convert to unequal dataset using pandas dataframe
    # X_train = convertToUnequalDF(trainpts)
    # X_test = convertToUnequalDF(testpts)
    # y_train = np.array(trainlbl)
    # y_test = np.array(testlbl)
    # print(f"X_train: {X_train.shape}")
    # print(f"X_test: {X_test.shape}")
    # print(f"y_train: {y_train.shape}")
    # print(f"y_test: {y_test.shape}")

    # start training process:
    print("training...")
    tskernel = AggrDist(RBF())
    clf = TimeSeriesSVC(kernel=tskernel)
    # clf = ColumnConcatenator() * TimeSeriesSVC(kernel=tskernel)
    # clf = ColumnConcatenator() * RocketClassifier()
    # clf = ColumnConcatenator() * DrCIF(n_estimators=100, n_intervals=5, n_jobs=4)
    clf.fit(X_train, y_train)

    # start testing process:
    print("testing...")
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("y_test:", y_test)
    print("y_pred:", y_pred)

    print(f"clf: {clf}")
    print(f"score: {score}")
