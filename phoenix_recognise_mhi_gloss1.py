"""
This file is used for simple sign recognition. It is trained with word examples and
It performs sign word recognition in one sentence video.

This file uses single classifier, one classifier for all classes.
But also uses multiple glosstype and glossp feature. The feature are:
    gloss1_files10p
    gloss2_files10p
    gloss1_files20p
    gloss2_files20p
    gloss1_files30p
    gloss2_files30p
    gloss1_files40p
    gloss2_files40p
    gloss1_files50p
    gloss2_files50p
    gloss1_files100p
    gloss2_files100p

This is the first version which do the recognition WITHOUT sign segmentation.

author: naserjawas
date: 13 Octover 2023
"""
import os
import glob
import cv2 as cv
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def save_clf(clfobj, drive, glosstype, glossp, mhitype):
    rootdir = drive + f"/gloss{glosstype}_files{glossp}p/"
    filename = rootdir + f"clf{glosstype}_{glossp}{mhitype}"
    with open(filename, 'wb') as pf:
        pickle.dump(clfobj, pf)
    print(filename, "saved...")

def load_clf(drive, glosstype, glossp, mhitype):
    rootdir = drive + f"/gloss{glosstype}_files{glossp}p/"
    filename = rootdir + f"clf{glosstype}_{glossp}{mhitype}"
    with open(filename, 'rb') as pf:
        clfobj = pickle.load(pf)
    print(filename, "loaded...")

    return clfobj

def load_filenames(drive, glosstype, glossp):
    fwdfiles = []
    bwdfiles = []
    labels = []
    rootdir = drive + f"/gloss{glosstype}_files{glossp}p/"
    for root, dirs, files in os.walk(rootdir):
        for name in dirs:
            if name.isnumeric():
                # limit the sample for training to below 10
                if int(name) > 10:
                    continue
                labels.append(root.split("/")[-1])
            dirname = os.path.join(root, name)
            fwdfile = dirname + f"/allmcmmhif{glossp}p.png"
            if os.path.exists(fwdfile):
                fwdfiles.append(fwdfile)
            bwdfile = dirname + f"/allmcmmhib{glossp}p.png"
            if os.path.exists(bwdfile):
                bwdfiles.append(bwdfile)
    print(f"gloss{glosstype}_files{glossp}p: {len(bwdfiles)}, {len(bwdfiles)}, {len(labels)}")

    return fwdfiles, bwdfiles, labels

def load_features(filenames):
    points = []
    for f in filenames:
        gimg = cv.imread(f, cv.IMREAD_GRAYSCALE)
        gpts = get_point_cloud_new(gimg, 4, 'all')
        points.append(gpts)

    return points

def load_allmcm_video(datadir, dataname, glosstype, glossp, blankframe):
    dataloc = datadir + dataname + f"/1/allmcm{glossp}p*-{glosstype}.png"
    datafiles = sorted(glob.glob(dataloc))
    dataframesori = [cv.imread(filename, cv.IMREAD_GRAYSCALE)
                     for filename in datafiles]
    dataframesori.insert(0, blankframe)

    dataframes = [cv.resize(frame, (iw, ih))
                  for frame in dataframesori ]

    return dataframes

def create_mhi(imgfiles):
    numfiles = len(imgfiles)
    mhif = np.zeros_like(imgfiles[0])
    for i, img in enumerate(imgfiles):
        mhif[img > 0] = (((i + 1) / numfiles) * 255)
    mhib = np.zeros_like(imgfiles[0])
    mhipointsb = get_point_cloud_new(mhib, 4, 'all')
    mhipointsb = np.array(mhipointsb).reshape(1, -1)
    for i, img in enumerate(reversed(imgfiles)):
        mhib[img > 0] = (((i + 1) / numfiles) * 255)
    mhipointsf = get_point_cloud_new(mhif, 4, 'all')
    mhipointsf = np.array(mhipointsf).reshape(1, -1)

    return mhipointsf, mhipointsb

def get_point_cloud_new(mhi, step, region):
    """ Get the point cloud of motion history image (mhi)

    Parameters
    ----------
    mhi : 2D numpy array np.uint8
        motion history image (opencv greyscale image).
    step: int
        step to create grids
    region: str
        contains 3 values: 'all', 'right', 'left'

    Returns
    -------
    list
        a 1D list of points.
    """
    ih, iw = mhi.shape
    if region == 'all':
        starty = 0
        startx = 0
        maxy = ih
        maxx = iw
    elif region == 'left':
        starty = 0
        startx = 0
        maxy = ih
        maxx = iw // 2
    elif region == 'right':
        starty = 0
        startx = iw // 2
        maxy = ih
        maxx = iw

    maxy -= step
    maxx -= step
    points = []
    for y in range(starty, maxy, step):
        for x in range(startx, maxx, step):
            crop = mhi[y:y+step, x:x+step]
            maxval = np.max(crop)
            points.append(maxval)
    return points

def add_results(results, classname, score, cw, ts):
    """ Add individual result into one variable that contains all results.
    """
    if ts == 0:
        classweight = cw[classname]
        score = score * classweight
    if classname in results:
        results[classname] += score
    else:
        results[classname] = score

    return results

def add_summary(summary, results, fid):
    """ Create summary from the results.
    """
    print(results)
    for classname, score in results.items():
        if classname in summary:
            # if (summary[classname][-1][0] == (fid-1) and
            #     summary[classname][-1][1] < score):
            summary[classname].append((fid, score))
        else:
            summary[classname] = [(fid, score)]

    return summary

def find_max_summary(summary, summarycount):
    """ Find maximum value from summary
    """
    # print(summary)
    sortedsumary = []
    for key, value in summary.items():
        for v in value:
            sortedsumary.append((key, v[0], v[1]))
    sortedsumary = sorted(sortedsumary, key = lambda x:x[2], reverse=False)
    # print(sortedsumary)
    lensumary = len(sortedsumary)

    topvalue = []
    for key, value in summary.items():
        # search for each individual key best strike
        indvalue = []
        value = sorted(value, key = lambda x:x[0], reverse=False)
        valuecount = summarycount[key]
        valuecount = sorted(valuecount, key = lambda x:x[0], reverse=False)
        s = None
        sc = None
        counter = 0
        totalv = 0
        # print(key)
        lenvalue = len(value)
        for i, v in enumerate(value):
            vc = valuecount[i]
            # print(v, vc)
            idg = sortedsumary.index((key, v[0], v[1]))
            idl = value.index(v)

            #XXX
            # if (v[1] / vc[1]) > 0.25:
            #    accept = True
            # else:
            #    accept = False
            accept = True

            newtotalv = ((v[1] / vc[1]) * (idg / lensumary)) + ((v[1] / vc[1]) * (idl / lenvalue))
            if s == None:
                counter = 1
                totalv = newtotalv
            else:
                if v[0] == (s[0] + 1) and v[1] > s[1] and accept:
                    counter += 1
                    totalv += newtotalv
                else:
                    indvalue.append((s[0],
                                     totalv,
                                     counter,
                                     (totalv * counter)))
                    counter = 1
                    totalv = newtotalv

            if accept:
                s = v
                sc = vc
            else:
                s = None
                sc = None
            # print(counter)

        #XXX
        # if len(indvalue) >= 0:
        #    indvalue.append((s[0],
        #                     totalv,
        #                     counter,
        #                     (totalv * counter)))
        # else:
        #    indvalue = sorted(indvalue, key = lambda x:x[3], reverse=True)
        # maxval = indvalue[0]
        # # append the individual key best strike
        # topvalue.append((key, maxval[0], maxval[1], maxval[2], maxval[3]))

        if len(indvalue) > 1:
           indvalue = sorted(indvalue, key = lambda x:x[3], reverse=True)
        if len(indvalue) > 0:
           maxval = indvalue[0]
           topvalue.append((key, maxval[0], maxval[1], maxval[2], maxval[3]))

    # sorted it to find the best of the best individual
    topvalue = sorted(topvalue, key = lambda x:x[4], reverse=True)
    print("topvalue:")
    for t in topvalue:
        print(t)

    return topvalue

def find_path(dataori, glossresult, ngrams, cw):
    # print("find_path", startpoint, endpoint)
    # loop for starting point to end point
    imgfiles = []
    imgfids = []
    f1_10 = []
    f2_10 = []
    f1_20 = []
    f2_20 = []
    f1_30 = []
    f2_30 = []
    f1_40 = []
    f2_40 = []
    f1_50 = []
    f2_50 = []
    f1_100 = []
    f2_100 = []
    results1 = {}
    results2 = {}
    summary3 = {}
    summary4 = {}
    proclen = 25
    loopstarted = True
    startpoint = 0
    while loopstarted:
        prevstartpoint = startpoint
        prevlenglossresult = len(glossresult)
        endpoint = startpoint + proclen
        summary3 = {}
        summary4 = {}

        for fid, frame in enumerate(dataori):
            if fid == 0 or fid < startpoint:
                continue
            elif fid >= endpoint:
                break

            results3 = {}
            results4 = {}
            # allmcm data processing.
            imgfids.append(fid)
            imgfiles.append(frame)
            f1_10.append(allmcm1_10[fid])
            f2_10.append(allmcm2_10[fid])
            f1_20.append(allmcm1_20[fid])
            f2_20.append(allmcm2_20[fid])
            f1_30.append(allmcm1_30[fid])
            f2_30.append(allmcm2_30[fid])
            f1_40.append(allmcm1_40[fid])
            f2_40.append(allmcm2_40[fid])
            f1_50.append(allmcm1_50[fid])
            f2_50.append(allmcm2_50[fid])
            f1_100.append(allmcm1_100[fid])
            f2_100.append(allmcm2_100[fid])

            mhi1_10f, mhi1_10b = create_mhi(f1_10)
            mhi2_10f, mhi2_10b = create_mhi(f2_10)
            mhi1_20f, mhi1_20b = create_mhi(f1_20)
            mhi2_20f, mhi2_20b = create_mhi(f2_20)
            mhi1_30f, mhi1_30b = create_mhi(f1_30)
            mhi2_30f, mhi2_30b = create_mhi(f2_30)
            mhi1_40f, mhi1_40b = create_mhi(f1_40)
            mhi2_40f, mhi2_40b = create_mhi(f2_40)
            mhi1_50f, mhi1_50b = create_mhi(f1_50)
            mhi2_50f, mhi2_50b = create_mhi(f2_50)
            mhi1_100f, mhi1_100b = create_mhi(f1_100)
            mhi2_100f, mhi2_100b = create_mhi(f2_100)

            pred1_10f = c1_10f.predict(mhi1_10f)
            pred1_10fpb = c1_10f.predict_proba(mhi1_10f)
            pred1_10b = c1_10b.predict(mhi1_10b)
            pred1_10bpb = c1_10b.predict_proba(mhi1_10b)
            pred2_10f = c1_10f.predict(mhi2_10f)
            pred2_10fpb = c1_10f.predict_proba(mhi2_10f)
            pred2_10b = c1_10b.predict(mhi2_10b)
            pred2_10bpb = c1_10b.predict_proba(mhi2_10b)
            pred1_20f = c1_20f.predict(mhi1_20f)
            pred1_20fpb = c1_20f.predict_proba(mhi1_20f)
            pred1_20b = c1_20b.predict(mhi1_20b)
            pred1_20bpb = c1_20b.predict_proba(mhi1_20b)
            pred2_20f = c1_20f.predict(mhi2_20f)
            pred2_20fpb = c1_20f.predict_proba(mhi2_20f)
            pred2_20b = c1_20b.predict(mhi2_20b)
            pred2_20bpb = c1_20b.predict_proba(mhi2_20b)
            pred1_30f = c1_30f.predict(mhi1_30f)
            pred1_30fpb = c1_30f.predict_proba(mhi1_30f)
            pred1_30b = c1_30b.predict(mhi1_30b)
            pred1_30bpb = c1_30b.predict_proba(mhi1_30b)
            pred2_30f = c1_30f.predict(mhi2_30f)
            pred2_30fpb = c1_30f.predict_proba(mhi2_30f)
            pred2_30b = c1_30b.predict(mhi2_30b)
            pred2_30bpb = c1_30b.predict_proba(mhi2_30b)
            pred1_40f = c1_40f.predict(mhi1_40f)
            pred1_40fpb = c1_40f.predict_proba(mhi1_40f)
            pred1_40b = c1_40b.predict(mhi1_40b)
            pred1_40bpb = c1_40b.predict_proba(mhi1_40b)
            pred2_40f = c1_40f.predict(mhi2_40f)
            pred2_40fpb = c1_40f.predict_proba(mhi2_40f)
            pred2_40b = c1_40b.predict(mhi2_40b)
            pred2_40bpb = c1_40b.predict_proba(mhi2_40b)
            pred1_50f = c1_50f.predict(mhi1_50f)
            pred1_50fpb = c1_50f.predict_proba(mhi1_50f)
            pred1_50b = c1_50b.predict(mhi1_50b)
            pred1_50bpb = c1_50b.predict_proba(mhi1_50b)
            pred2_50f = c1_50f.predict(mhi2_50f)
            pred2_50fpb = c1_50f.predict_proba(mhi2_50f)
            pred2_50b = c1_50b.predict(mhi2_50b)
            pred2_50bpb = c1_50b.predict_proba(mhi2_50b)
            pred1_100f = c1_100f.predict(mhi1_100f)
            pred1_100fpb = c1_100f.predict_proba(mhi1_100f)
            pred1_100b = c1_100b.predict(mhi1_100b)
            pred1_100bpb = c1_100b.predict_proba(mhi1_100b)
            pred2_100f = c1_100f.predict(mhi2_100f)
            pred2_100fpb = c1_100f.predict_proba(mhi2_100f)
            pred2_100b = c1_100b.predict(mhi2_100b)
            pred2_100bpb = c1_100b.predict_proba(mhi2_100b)

            print(fid)

            results1 = add_results(results1, le.inverse_transform(pred1_10f)[0], np.max(pred1_10fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_10f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_10f)[0], np.max(pred1_10fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_10f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_10b)[0], np.max(pred1_10bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_10b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_10b)[0], np.max(pred1_10bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_10b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_10f)[0], np.max(pred2_10fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_10f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_10f)[0], np.max(pred2_10fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_10f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_10b)[0], np.max(pred2_10bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_10b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_10b)[0], np.max(pred2_10bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_10b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_20f)[0], np.max(pred1_20fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_20f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_20f)[0], np.max(pred1_20fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_20f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_20b)[0], np.max(pred1_20bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_20b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_20b)[0], np.max(pred1_20bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_20b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_20f)[0], np.max(pred2_20fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_20f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_20f)[0], np.max(pred2_20fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_20f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_20b)[0], np.max(pred2_20bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_20b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_20b)[0], np.max(pred2_20bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_20b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_30f)[0], np.max(pred1_30fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_30f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_30f)[0], np.max(pred1_30fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_30f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_30b)[0], np.max(pred1_30bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_30b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_30b)[0], np.max(pred1_30bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_30b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_30f)[0], np.max(pred2_30fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_30f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_30f)[0], np.max(pred2_30fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_30f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_30b)[0], np.max(pred2_30bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_30b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_30b)[0], np.max(pred2_30bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_30b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_40f)[0], np.max(pred1_40fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_40f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_40f)[0], np.max(pred1_40fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_40f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_40b)[0], np.max(pred1_40bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_40b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_40b)[0], np.max(pred1_40bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_40b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_40f)[0], np.max(pred2_40fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_40f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_40f)[0], np.max(pred2_40fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_40f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_40b)[0], np.max(pred2_40bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_40b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_40b)[0], np.max(pred2_40bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_40b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_50f)[0], np.max(pred1_50fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_50f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_50f)[0], np.max(pred1_50fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_50f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_50b)[0], np.max(pred1_50bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_50b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_50b)[0], np.max(pred1_50bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_50b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_50f)[0], np.max(pred2_50fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_50f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_50f)[0], np.max(pred2_50fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_50f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_50b)[0], np.max(pred2_50bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_50b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_50b)[0], np.max(pred2_50bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_50b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_100f)[0], np.max(pred1_100fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_100f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_100f)[0], np.max(pred1_100fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_100f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred1_100b)[0], np.max(pred1_100bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred1_100b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred1_100b)[0], np.max(pred1_100bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred1_100b)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_100f)[0], np.max(pred2_100fpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_100f)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_100f)[0], np.max(pred2_100fpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_100f)[0], 1, cw, 1)

            results1 = add_results(results1, le.inverse_transform(pred2_100b)[0], np.max(pred2_100bpb), cw, 0)
            results2 = add_results(results2, le.inverse_transform(pred2_100b)[0], 1, cw, 1)
            results3 = add_results(results3, le.inverse_transform(pred2_100b)[0], np.max(pred2_100bpb), cw, 0)
            results4 = add_results(results4, le.inverse_transform(pred2_100b)[0], 1, cw, 1)

            summary3 = add_summary(summary3, results3, fid)
            summary4 = add_summary(summary4, results4, fid)

        # analyse the data for the loop.
        # try to find the highest value with
        # the longest improvement
        topvalue = find_max_summary(summary3, summary4)

        if len(glossresult) > 2:
            # get top ngram
            topngram = []
            if len(glossresult) == 0:
                for t in topvalue:
                    ngram = ngram_proba(ngrams, t[0])
                    topngram.append(ngram)
            elif len(glossresult) == 1:
                for t in topvalue:
                    ngram = ngram_proba(ngrams, glossresult[-1][0], t[0])
                    topngram.append(ngram)
            else:
                for t in topvalue:
                    ngram = ngram_proba(ngrams, glossresult[-2][0], glossresult[-1][0], t[0])
                    topngram.append(ngram)
            topngram = sorted(topngram, key=lambda x:x[1], reverse=True)
            print("topngram:")
            for t in topngram:
                print(t)

            # top value x top ngram
            newtopvalue = []
            for tv in topvalue:
                tng = list(filter(lambda x: (x[0]==tv[0]), topngram))
                if len(tng) > 0 and tng[0][1] > 0:
                    newtopvalue.append((tv[0], tv[1], tv[2]*tng[0][1], tv[3], tv[4]*tng[0][1]))

            topvalue = sorted(newtopvalue, key=lambda x:x[4], reverse=True)
            print("new topvalue:")
            for tv in topvalue:
                print(tv)

        # get top value
        if len(topvalue) >= 1:
            firstgloss = topvalue[0]
            if len(glossresult) == 0:
                glossresult.append(firstgloss)
                first = firstgloss[1]
            else:
                lastresult = glossresult[-1]
                for i, g in enumerate(topvalue):
                    if g[1] > (lastresult[1] + 3):
                        glossresult.append(g)
                        first = g[1]
                        imgfiles = []
                        imgfids = []
                        f1_10 = []
                        f2_10 = []
                        f1_20 = []
                        f2_20 = []
                        f1_30 = []
                        f2_30 = []
                        f1_40 = []
                        f2_40 = []
                        f1_50 = []
                        f2_50 = []
                        f1_100 = []
                        f2_100 = []
                        results1 = {}
                        results2 = {}
                        summary3 = {}
                        summary4 = {}
                        break
                    elif g[1] == endpoint:
                        first = g[1]
                    else:
                        first = lastresult[1]
        else:
            if len(glossresult) == 0:
                first = 0
            else:
                first = glossresult[-1][1]

        print("first:", first)
        # set new starting point.
        startpoint = first + 1
        # set new end point.
        endpoint = startpoint + proclen
        print("endpoint:", endpoint)
        print()
        print("glossresult:")
        for gr in glossresult:
            print(gr)

        if startpoint > 0 and prevlenglossresult == len(glossresult):
            skipcounter += 1
        else:
            skipcounter = 0

        startpoint += skipcounter
        endpoint += skipcounter

        if startpoint >= len(dataori):
            loopstarted = False

        c = input()

    return glossresult

def ngram_proba(ngrams, gloss1="", gloss2="", gloss3=""):
    f1 = []
    f2 = []
    f3 = []
    if gloss1 != "" and gloss2 == "" and gloss3 == "":
        f1 = list(filter(lambda x: (x[0]==gloss1), ngrams))
        # print(gloss1, len(f1), len(ngrams))
        if len(ngrams) > 0:
            return (gloss1, (len(f1)/len(ngrams)))
        else:
            return (gloss1, 0.0)
    elif gloss1 != "" and gloss2 != "" and gloss3 == "":
        f1 = list(filter(lambda x: (x[0]==gloss1), ngrams))
        f2 = list(filter(lambda x: (x[1]==gloss2), f1))
        # print(gloss2, gloss1, len(f2), len(f1))
        if len(f1) > 0:
            return (gloss2, (len(f2)/len(f1)))
        else:
            return (gloss2, 0.0)
    else:
        f1 = list(filter(lambda x: (x[0]==gloss1), ngrams))
        f2 = list(filter(lambda x: (x[1]==gloss2), f1))
        f3 = list(filter(lambda x: (x[2]==gloss3), f2))
        # print(gloss3, gloss2, gloss1, len(f3), len(f2))
        if len(f2) > 0:
            return (gloss3, (len(f3)/len(f2)))
        else:
            return (gloss3, 0.0)

def calculate_weight(classes, samples):
    weight = {}
    num_samples = len(samples)
    num_classes = len(classes)

    for c in classes:
        num_c = samples.count(c)
        weight[c] = num_samples / (num_classes * num_c)

    return weight


if __name__ == "__main__":
    train = False
    drive = "../dataset/RWTHPHOENIXWeather2014"
    g1f10f, g1f10b, g1f10l = load_filenames(drive, 1, 10)
    g2f10f, g2f10b, g2f10l = load_filenames(drive, 2, 10)
    g1f20f, g1f20b, g1f20l = load_filenames(drive, 1, 20)
    g2f20f, g2f20b, g2f20l = load_filenames(drive, 2, 20)
    g1f30f, g1f30b, g1f30l = load_filenames(drive, 1, 30)
    g2f30f, g2f30b, g2f30l = load_filenames(drive, 2, 30)
    g1f40f, g1f40b, g1f40l = load_filenames(drive, 1, 40)
    g2f40f, g2f40b, g2f40l = load_filenames(drive, 2, 40)
    g1f50f, g1f50b, g1f50l = load_filenames(drive, 1, 50)
    g2f50f, g2f50b, g2f50l = load_filenames(drive, 2, 50)
    g1f100f, g1f100b, g1f100l = load_filenames(drive, 1, 100)
    g2f100f, g2f100b, g2f100l = load_filenames(drive, 2, 100)

    le = LabelEncoder()
    le.fit(g1f100l)
    print(le.classes_)

    f1_10f = np.array(load_features(g1f10f))
    f1_10b = np.array(load_features(g1f10b))
    l1_10 = le.transform(g1f10l)
    cw = calculate_weight(list(le.classes_), list(le.inverse_transform(l1_10)))
    print(cw)

    f2_10f = np.array(load_features(g2f10f))
    f2_10b = np.array(load_features(g2f10b))
    l2_10 = le.transform(g2f10l)
    f1_20f = np.array(load_features(g1f20f))
    f1_20b = np.array(load_features(g1f20b))
    l1_20 = le.transform(g1f20l)
    f2_20f = np.array(load_features(g2f20f))
    f2_20b = np.array(load_features(g2f20b))
    l2_20 = le.transform(g2f20l)
    f1_30f = np.array(load_features(g1f30f))
    f1_30b = np.array(load_features(g1f30b))
    l1_30 = le.transform(g1f30l)
    f2_30f = np.array(load_features(g2f30f))
    f2_30b = np.array(load_features(g2f30b))
    l2_30 = le.transform(g2f30l)
    f1_40f = np.array(load_features(g1f40f))
    f1_40b = np.array(load_features(g1f40b))
    l1_40 = le.transform(g1f40l)
    f2_40f = np.array(load_features(g2f40f))
    f2_40b = np.array(load_features(g2f40b))
    l2_40 = le.transform(g2f40l)
    f1_50f = np.array(load_features(g1f50f))
    f1_50b = np.array(load_features(g1f50b))
    l1_50 = le.transform(g1f50l)
    f2_50f = np.array(load_features(g2f50f))
    f2_50b = np.array(load_features(g2f50b))
    l2_50 = le.transform(g2f50l)
    f1_100f = np.array(load_features(g1f100f))
    f1_100b = np.array(load_features(g1f100b))
    l1_100 = le.transform(g1f100l)
    f2_100f = np.array(load_features(g2f100f))
    f2_100b = np.array(load_features(g2f100b))
    l2_100 = le.transform(g2f100l)

    if train:
        c1_10f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_10b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_10f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_10b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_20f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_20b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_20f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_20b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_30f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_30b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_30f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_30b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_40f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_40b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_40f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_40b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_50f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_50b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_50f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_50b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_100f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c1_100b = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_100f = RandomForestClassifier(n_estimators=1000, random_state=0)
        c2_100b = RandomForestClassifier(n_estimators=1000, random_state=0)

        c1_10f.fit(f1_10f, l1_10)
        c1_10b.fit(f1_10b, l1_10)
        c2_10f.fit(f2_10f, l1_10)
        c2_10b.fit(f2_10b, l1_10)
        c1_20f.fit(f1_20f, l1_20)
        c1_20b.fit(f1_20b, l1_20)
        c2_20f.fit(f2_20f, l1_20)
        c2_20b.fit(f2_20b, l1_20)
        c1_30f.fit(f1_30f, l1_30)
        c1_30b.fit(f1_30b, l1_30)
        c2_30f.fit(f2_30f, l1_30)
        c2_30b.fit(f2_30b, l1_30)
        c1_40f.fit(f1_40f, l1_40)
        c1_40b.fit(f1_40b, l1_40)
        c2_40f.fit(f2_40f, l1_40)
        c2_40b.fit(f2_40b, l1_40)
        c1_50f.fit(f1_50f, l1_50)
        c1_50b.fit(f1_50b, l1_50)
        c2_50f.fit(f2_50f, l1_50)
        c2_50b.fit(f2_50b, l1_50)
        c1_100f.fit(f1_100f, l1_100)
        c1_100b.fit(f1_100b, l1_100)
        c2_100f.fit(f2_100f, l1_100)
        c2_100b.fit(f2_100b, l1_100)

        save_clf(c1_10f, drive, 1, 10, 'f')
        save_clf(c1_10b, drive, 1, 10, 'b')
        save_clf(c2_10f, drive, 2, 10, 'f')
        save_clf(c2_10b, drive, 2, 10, 'b')
        save_clf(c1_20f, drive, 1, 20, 'f')
        save_clf(c1_20b, drive, 1, 20, 'b')
        save_clf(c2_20f, drive, 2, 20, 'f')
        save_clf(c2_20b, drive, 2, 20, 'b')
        save_clf(c1_30f, drive, 1, 30, 'f')
        save_clf(c1_30b, drive, 1, 30, 'b')
        save_clf(c2_30f, drive, 2, 30, 'f')
        save_clf(c2_30b, drive, 2, 30, 'b')
        save_clf(c1_40f, drive, 1, 40, 'f')
        save_clf(c1_40b, drive, 1, 40, 'b')
        save_clf(c2_40f, drive, 2, 40, 'f')
        save_clf(c2_40b, drive, 2, 40, 'b')
        save_clf(c1_50f, drive, 1, 50, 'f')
        save_clf(c1_50b, drive, 1, 50, 'b')
        save_clf(c2_50f, drive, 2, 50, 'f')
        save_clf(c2_50b, drive, 2, 50, 'b')
        save_clf(c1_100f, drive, 1, 100, 'f')
        save_clf(c1_100b, drive, 1, 100, 'b')
        save_clf(c2_100f, drive, 2, 100, 'f')
        save_clf(c2_100b, drive, 2, 100, 'b')

    else:
        c1_10f  = load_clf(drive, 1, 10, 'f')
        c1_10b  = load_clf(drive, 1, 10, 'b')
        c2_10f  = load_clf(drive, 2, 10, 'f')
        c2_10b  = load_clf(drive, 2, 10, 'b')
        c1_20f  = load_clf(drive, 1, 20, 'f')
        c1_20b  = load_clf(drive, 1, 20, 'b')
        c2_20f  = load_clf(drive, 2, 20, 'f')
        c2_20b  = load_clf(drive, 2, 20, 'b')
        c1_30f  = load_clf(drive, 1, 30, 'f')
        c1_30b  = load_clf(drive, 1, 30, 'b')
        c2_30f  = load_clf(drive, 2, 30, 'f')
        c2_30b  = load_clf(drive, 2, 30, 'b')
        c1_40f  = load_clf(drive, 1, 40, 'f')
        c1_40b  = load_clf(drive, 1, 40, 'b')
        c2_40f  = load_clf(drive, 2, 40, 'f')
        c2_40b  = load_clf(drive, 2, 40, 'b')
        c1_50f  = load_clf(drive, 1, 50, 'f')
        c1_50b  = load_clf(drive, 1, 50, 'b')
        c2_50f  = load_clf(drive, 2, 50, 'f')
        c2_50b  = load_clf(drive, 2, 50, 'b')
        c1_100f = load_clf(drive, 1, 100, 'f')
        c1_100b = load_clf(drive, 1, 100, 'b')
        c2_100f = load_clf(drive, 2, 100, 'f')
        c2_100b = load_clf(drive, 2, 100, 'b')

    # test starts here
    print("Load video dataset...")
    drive = "../dataset/RWTHPHOENIXWeather2014"
    datadir = drive + "/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/"

    datanames = sorted(os.listdir(datadir))
    print(len(datanames), "data available...")
    iw, ih = 210, 300
    blankframe = np.zeros((iw, ih), dtype=np.uint8)

    # load ngrams language model
    with open("./phoenix_3grams", "rb") as pf:
        ngrams = pickle.load(pf)
    pf.close()

    datanamespart = datanames[3:4]
    # NEXT
    for dataname_i, dataname in enumerate(datanamespart):
        # load original data frames
        dataloc = datadir + dataname + f"/1/*-0.png"
        datafiles = sorted(glob.glob(dataloc))
        lendatafiles = len(datafiles)
        print("(", dataname_i + 1, "/", len(datanamespart),")",
              dataname, "has", lendatafiles, "frames")
        imgori = [cv.imread(filename, cv.IMREAD_COLOR)
                  for filename in datafiles]
        dataori = [cv.resize(img, (iw, ih))
                   for img in imgori]

        # load data allmcm.
        allmcm1_10 = load_allmcm_video(datadir, dataname, 1, 10, blankframe)
        allmcm2_10 = load_allmcm_video(datadir, dataname, 2, 10, blankframe)
        allmcm1_20 = load_allmcm_video(datadir, dataname, 1, 20, blankframe)
        allmcm2_20 = load_allmcm_video(datadir, dataname, 2, 20, blankframe)
        allmcm1_30 = load_allmcm_video(datadir, dataname, 1, 30, blankframe)
        allmcm2_30 = load_allmcm_video(datadir, dataname, 2, 30, blankframe)
        allmcm1_40 = load_allmcm_video(datadir, dataname, 1, 40, blankframe)
        allmcm2_40 = load_allmcm_video(datadir, dataname, 2, 40, blankframe)
        allmcm1_50 = load_allmcm_video(datadir, dataname, 1, 50, blankframe)
        allmcm2_50 = load_allmcm_video(datadir, dataname, 2, 50, blankframe)
        allmcm1_100 = load_allmcm_video(datadir, dataname, 1, 100, blankframe)
        allmcm2_100 = load_allmcm_video(datadir, dataname, 2, 100, blankframe)

        glossresult1 = []
        glossresult2 = []
        glossresult3 = []

        glossresult1 = find_path(dataori, glossresult1, ngrams, cw)

        c = input()
