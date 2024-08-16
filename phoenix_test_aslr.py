"""
This file is for testing the proposed ASLR system on Phoenix test dataset.

author: naserjawas
date:   15 August 2024
"""

import os
import pickle
import glob
import cv2 as cv
import numpy as np

def load_clf(drive, glosstype, glossp, mhitype, limitsample):
    rootdir = drive + f"/classifiers/"
    filename = rootdir + f"j{limitsample}clf{glosstype}_{glossp}{mhitype}"
    if not os.path.exists(filename):
        print("Classifier:", filename, "does not exist")
        exit()
    print(filename)
    with open(filename, 'rb') as pf:
        clfobj = pickle.load(pf)
    print(filename, "loaded...")
    print(clfobj)

    return clfobj


def load_allmcm_video(datadir, dataname, glosstype, glossp, blankframe):
    dataloc = datadir + dataname + f"/1/allmcm{glossp}p*-{glosstype}.png"
    datafiles = sorted(glob.glob(dataloc))
    dataframesori = [cv.imread(filename, cv.IMREAD_GRAYSCALE)
                     for filename in datafiles]
    dataframesori.insert(0, blankframe)
    dataframes = [cv.resize(frame, (iw, ih))
                  for frame in dataframesori]

    return dataframes


def load_filenames(drive, glosstype, glossp, limitsample):
    fwdfiles = []
    bwdfiles = []
    labels = []
    rootdir = drive + f"/new_gloss{glosstype}_files{glossp}p/"

    for root, dirs, files in os.walk(rootdir):
        for name in dirs:
            if name.isnumeric():
                # limit the sample for training to below 10
                if int(name) > limitsample:
                    continue
                parentdir = root.split("/")[-1]
                if len(parentdir) < 2:
                    continue
                if '-' in parentdir:
                    continue
                if '_' in parentdir:
                    continue
                labels.append(parentdir)
            else:
                continue
            dirname = os.path.join(root, name)
            # print(dirname)
            fwdfile = dirname + f"/allmcmmhif{glossp}p.png"
            if os.path.exists(fwdfile):
                fwdfiles.append(fwdfile)
            bwdfile = dirname + f"/allmcmmhib{glossp}p.png"
            if os.path.exists(bwdfile):
                bwdfiles.append(bwdfile)

    print(f"gloss{glosstype}_files{glossp}p: {len(fwdfiles)}, {len(bwdfiles)}, {len(labels)}")

    return fwdfiles, bwdfiles, labels


def get_point_cloud_new(mhi, step, region):
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


def add_summary(summary, results, startidx, endidx):
    """ Create summary from the results.
    """
    # print(results)
    for classname, score in results.items():
        if classname in summary:
            summary[classname].append((startidx, endidx, score))
        else:
            summary[classname] = [(startidx, endidx, score)]

    return summary


def find_max_summary(summary, summarycount, start, stop):
    """ Find maximum value from summary
    """
    numcls = 24
    # print(start, stop)
    topvalue = []
    for key, value in summary.items():
        indvalue = []
        value = sorted(value, key = lambda x: x[2], reverse=True)
        valuecount = summarycount[key]
        if len(value) > 0:
            v = value[0]
            vc = list(filter(lambda x: (x[0]==v[0] and x[1]==v[1]), valuecount))[0]
            # print(key, v, vc)
            # distance of occurance weight: v[0] to v[1]
            if ((stop - start) - (v[1] - v[0])) > 0:
                wo =  (stop - start) / ((stop - start) - (v[1] - v[0]))
            else:
                wo = 0

            # distance from start weight: v[0] to v_min
            if ((stop - start) - (start - v[0])) > 0:
                ws = ((stop - start) / ((stop - start) - (start - v[0])))
            else:
                ws = 0

            # number of classifier weight: vc[2]
            if (numcls - vc[2]) > 0:
                wc =  numcls / (numcls - vc[2])
            else:
                wc = numcls

            # calculate total value
            totalv = v[2]
            totalv = totalv * (wo + ws + wc)
            # print(key, wo, ws, wc, v[2], vc[2], totalv)
            topvalue.append((key, v[0], v[1], totalv))

    # sorted it to find the best of the best individual
    topvalue = sorted(topvalue, key = lambda x:x[3], reverse=True)
    # print("topvalue:")
    # for t in topvalue:
    #     print(t)

    return topvalue


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


def find_path(dataori, dataname, glossresult, ngrams, peaks, cw, allmcm, cf, cb):
    # loop for starting point to end point
    imgfiles = []
    imgfids = []

    f = []

    results1 = {}
    results2 = {}
    proclen = 50
    procstep = 15
    loopstarted = True
    start_i = 0
    while loopstarted:
        summary3 = {}
        summary4 = {}
        stop_i = start_i + procstep
        if stop_i > len(peaks)-1:
            stop_i = len(peaks)-1

        for i in range(start_i, stop_i):
            # print("i:", i)
            startpoint = peaks[i]
            start_j = i + 1
            # stop_j = start_j + procstep
            stop_j = stop_i + 1
            if stop_j > len(peaks):
                stop_j = len(peaks)

            for j in range(start_j, stop_j):
                # print("i-j:",i, j)
                endpoint = peaks[j]
                if (endpoint - startpoint) == 1:
                    continue
                imgfiles = []
                imgfids = []

                f = []

                results1 = {}
                results2 = {}
                results3 = {}
                results4 = {}
                # print("startpoint - endpoint:", startpoint, endpoint)

                for fid, frame in enumerate(dataori):
                    if fid < startpoint:
                        continue
                    elif fid >= endpoint:
                        break
                    # allmcm data processing.
                    imgfids.append(fid)
                    imgfiles.append(frame)

                    f.append(allmcm[fid])

                # print("imgfids:", imgfids)

                mhif, mhib = create_mhi(f)

                predf = cf.predict(mhif)
                predfpb = cf.predict_proba(mhif)
                predb = cb.predict(mhib)
                predbpb = cb.predict_proba(mhib)

                results1 = add_results(results1, le.inverse_transform(predf)[0], np.max(predfpb), cw, 0)
                results2 = add_results(results2, le.inverse_transform(predf)[0], 1, cw, 1)
                results3 = add_results(results3, le.inverse_transform(predf)[0], np.max(predfpb), cw, 0)
                results4 = add_results(results4, le.inverse_transform(predf)[0], 1, cw, 1)
                results1 = add_results(results1, le.inverse_transform(predb)[0], np.max(predbpb), cw, 0)
                results2 = add_results(results2, le.inverse_transform(predb)[0], 1, cw, 1)
                results3 = add_results(results3, le.inverse_transform(predb)[0], np.max(predbpb), cw, 0)
                results4 = add_results(results4, le.inverse_transform(predb)[0], 1, cw, 1)

                summary3 = add_summary(summary3, results3, i, j)
                summary4 = add_summary(summary4, results4, i, j)

        # print("summary3:", summary3)
        # for k,v in summary3.items():
        #     print(k, len(v))
        # print("summary4:", summary4)
        # for k,v in summary4.items():
        #     print(k, len(v))

        # analyse the data for the loop.
        # try to find the highest value with
        # the longest improvement
        topvalue = find_max_summary(summary3, summary4, start_i, stop_i)

        if len(glossresult) > 0:
        # if len(glossresult) < 0:
            # get top ngram
            topngram = []
            if len(glossresult) == 0:
                # print("1-gram")
                for t in topvalue:
                    ngram = ngram_proba(ngrams, t[0])
                    topngram.append(ngram)
            elif len(glossresult) == 1:
                # print("2-gram")
                for t in topvalue:
                    ngram = ngram_proba(ngrams, glossresult[-1][0], t[0])
                    topngram.append(ngram)
            else:
                # print("3-gram")
                for t in topvalue:
                    ngram = ngram_proba(ngrams, glossresult[-2][0], glossresult[-1][0], t[0])
                    topngram.append(ngram)
            topngram = sorted(topngram, key=lambda x:x[1], reverse=True)
            # print("topngram:")
            # for t in topngram:
            #     print(t)

            # top value x top ngram
            newtopvalue = []
            for tv in topvalue:
                tng = list(filter(lambda x: (x[0]==tv[0]), topngram))
                if len(tng) > 0 and tng[0][1] > 0:
                    newtv = tv[3]*tng[0][1]
                    newtopvalue.append((tv[0], tv[1], tv[2], newtv))

            newtopvalue = sorted(newtopvalue, key=lambda x:x[3], reverse=True)
            if len(newtopvalue) > 3:
                if newtopvalue[-1] != newtopvalue[-2] and newtopvalue[-1] != newtopvalue[-3]:
                    topvalue = newtopvalue

            # print("new topvalue:")
            # for tv in topvalue:
            #     print(tv)

        if len(topvalue) > 0:
            glossresult.append(topvalue[0])
            start_i = topvalue[0][2]
        else:
            start_i = start_i + 1

        # print()
        # print("Video:", dataname)
        # print("glossresult:")
        # for g in glossresult:
        #     print(g[0])

        if start_i >= len(peaks):
            loopstarted = False

        # c = input()

        # if c == "n":
        #     loopstarted = False

    return glossresult


if __name__ == "__main__":
    limitsample = 10
    glosstype = 1
    glossp = 40
    drive = "/content/dataset/RWTHPHOENIXWeather2014"

    cf = load_clf(drive, glosstype, glossp, 'f', limitsample)
    cb = load_clf(drive, glosstype, glossp, 'b', limitsample)

    # n-gram
    n = 3
    with open(f"/content/dataset/RWTHPHOENIXWeather2014/ngram_files/phoenix_{n}grams", "rb") as pf:
        ngrams = pickle.load(pf)
    pf.close()
    # label encoder objects
    with open(f"/content/dataset/RWTHPHOENIXWeather2014/encoded_label/label_le_obj{limitsample}", "rb") as pf:
        le = pickle.load(pf)
    pf.close()
    with open(f"/content/dataset/RWTHPHOENIXWeather2014/encoded_label/label_sample{limitsample}", "rb") as pf:
        lbl = pickle.load(pf)
    pf.close()
    with open(f"/content/dataset/RWTHPHOENIXWeather2014/encoded_label/label_weight{limitsample}", "rb") as pf:
        cw = pickle.load(pf)
    pf.close()
    # segment objects
    sp = 50
    st = 1
    with open(f"/content/dataset/RWTHPHOENIXWeather2014/segment_files/alldata_test_{sp}p_{st}", "rb") as pf:
        segments = pickle.load(pf)
    pf.close()

    datadir = drive + "/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/test/"
    datanames = sorted(os.listdir(datadir))
    if datanames[0] == ".DS_Store":
        datanames.pop(0)
    print(len(datanames), "data available...")
    iw, ih = 210, 300
    blankframe = np.zeros((iw, ih), dtype=np.uint8)

    annotationfile = drive + "/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/test"
    with open(annotationfile,"rb") as pf:
        annotations = pf.readlines()
    pf.close()

    teststart = 1
    teststop = 11
    datanamespart = datanames[teststart:teststop]

    finalresults = []
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
        peaks = segments[dataname]
        peaks.insert(0, 0)
        peaks.append(lendatafiles-1)
        allmcm = load_allmcm_video(datadir, dataname, glosstype, glossp, blankframe)
        glossresult1 = []

        glossresult1 = find_path(dataori, dataname, glossresult1, ngrams, peaks, cw, allmcm, cf, cb)
        listgloss = [g[0] for g in glossresult1]

        a = annotations[teststart + dataname_i]
        print(a)
        print(listgloss)

        finalresults.append((dataname, listgloss))

    with open('finalresults', 'wb') as pf:
        pickle.dump(finalresults, pf)
    pf.close()
