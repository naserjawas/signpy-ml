"""
This file for extracting the sign words frames data from videos.

author: naserjawas
date  : 5 October 2023
"""

import os
import glob
import cv2 as cv
import numpy as np

def addNewGlossData(oldvideofile, oldgloss, oldclassid, framefilelist, datadir,
                    glossdir, glosstype, glossp):
    if len(framefilelist) == 0:
        return framefilelist

    if oldvideofile != "":
        if oldgloss == "si":
            print(oldvideofile, oldclassid, oldgloss)
        else:
            print(oldvideofile, oldclassid-2, oldclassid-1, oldclassid,
                  oldgloss)

    glosssubdir = glossdir + oldgloss + "/"
    if not os.path.exists(glosssubdir):
        print("makedirs:", glosssubdir)
        os.makedirs(glosssubdir)
    # dirs = os.listdir(glosssubdir)
    lendirs = len(next(os.walk(glosssubdir))[1])
    glosssubsubdir = glosssubdir + str(lendirs) + "/"
    if not os.path.exists(glosssubsubdir):
        print("makedirs:", glosssubsubdir)
        os.makedirs(glosssubsubdir)

    allmcmfiles = []
    for frm in framefilelist:
        framefilesplit0 = frm.split("_")
        framefileid = framefilesplit0[-1][2:8]
        framefilesplit1 = frm.split("/")
        featurefile = datadir + "/"
        for f in framefilesplit1[:-1]:
            featurefile += f + "/"
        mainfeaturefile = featurefile + f"allmcm{glossp}p{framefileid}-{glosstype}.png"
        mainframefile = featurefile + framefilesplit1[-1]

        # copy the image frame file.
        frame = cv.imread(mainframefile, cv.IMREAD_COLOR)
        print("imwrite:", glosssubsubdir + framefilesplit1[-1])
        cv.imwrite(glosssubsubdir + framefilesplit1[-1], frame)

        # copy the image allmcm feature file.
        if os.path.exists(mainfeaturefile):
            allmcm = cv.imread(mainfeaturefile, cv.IMREAD_GRAYSCALE)
        else:
            allmcm = np.zeros((300, 210), np.uint8)
        print("imwrite:", glosssubsubdir +
              f"allmcm{glossp}p{framefileid}-{glosstype}.png")
        cv.imwrite(glosssubsubdir +
                   f"allmcm{glossp}p{framefileid}-{glosstype}.png", allmcm)
        if cv.hasNonZero(allmcm):
            allmcmfiles.append(allmcm)

    # create and save the mhi forward image file.
    numfiles = len(allmcmfiles)
    mhif = np.zeros((300, 210), np.uint8)
    for i, img in enumerate(allmcmfiles):
        mhif[img > 0] = (((i+1) / numfiles) * 255)
    print("imwrite:", glosssubsubdir + f"allmcmmhif{glossp}p.png")
    cv.imwrite(glosssubsubdir + f"allmcmmhif{glossp}p.png", mhif)

    # create and save the mhi backward image file.
    mhib = np.zeros((300, 210), np.uint8)
    for i, img in enumerate(reversed(allmcmfiles)):
        mhib[img > 0] = (((i+1) / numfiles) * 255)
    print("imwrite:", glosssubsubdir + f"allmcmmhib{glossp}p.png")
    cv.imwrite(glosssubsubdir + f"allmcmmhib{glossp}p.png", mhib)

    framefilelist = []
    print()

    return framefilelist

if __name__ == "__main__":
    glossp = 10
    glosstype = 1
    drive = "../dataset/RWTHPHOENIXWeather2014"
    datadir = drive + "/phoenix2014-release/phoenix-2014-multisigner"

    alignfile = datadir + "/annotations/automatic/train.alignment"
    classfile = datadir + "/annotations/automatic/trainingClasses.txt"

    classlist = []
    classfilereader = open(classfile, 'r').read().splitlines()
    for line in classfilereader[1:]:
        data = line.split(" ")
        datainput = (int(data[1]), data[0])
        classlist.append(datainput)

    classdict = dict(classlist)

    glossdir = drive + f"/gloss{glosstype}_files{glossp}p/"
    if not os.path.exists(glossdir):
        print("makedirs:", glossdir)
        os.makedirs(glossdir)

    glosssubdir = ""
    glosssubsubdir = ""

    oldvideofile = ""
    videofilelist = []
    videofilelimit = 1

    oldgloss = ""
    oldclassid = 0
    framefilelist = []
    alignfilereader = open(alignfile, 'r').read().splitlines()

    # targetgloss = ["__ON__", "LIEB", "ZUSCHAUER", "ABEND", "WINTER", "GESTERN",
    #                "loc-NORD", "SCHOTTLAND", "loc-REGION", "UEBERSCHWEMMUNG",
    #                "AMERIKA", "IX"]

    for line in alignfilereader:
        data = line.split(" ")
        framefile = datadir + "/" + data[0]

        if not os.path.exists(framefile):
            break

        framefilesplit = framefile.split("/")
        videofile = framefilesplit[-3]
        framefile = framefilesplit[-1]

        classid = int(data[1])
        gloss = classdict[classid]
        if gloss[-1].isnumeric():
            gloss = gloss[:-1]

        # if gloss not in targetgloss:
        #     continue

        # check if a new video file is being processed
        # break the loop if the number of processed video has reached the limit
        if oldvideofile != videofile:
           if len(videofilelist) < videofilelimit or videofilelimit <= 0:
               videofilelist.append(videofile)
           else:
               break

        if ((oldvideofile != videofile and videofile != "") or
            (oldgloss != gloss and oldgloss != "") or
            (oldgloss == gloss and oldclassid==classid+2)):
            # do #XXX,
            framefilelist = addNewGlossData(oldvideofile, oldgloss, oldclassid,
                                            framefilelist, datadir, glossdir,
                                            glosstype, glossp)

        framefilelist.append(data[0])
        oldvideofile = videofile
        oldgloss = gloss
        oldclassid = classid

    # do #XXX,here as well
    framefilelist = addNewGlossData(oldvideofile, oldgloss, oldclassid,
                                    framefilelist, datadir, glossdir,
                                    glosstype, glossp)
