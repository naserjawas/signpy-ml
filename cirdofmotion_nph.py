"""
This file generates the motion history image of cirdofextract output.

author: naserjawas
date  : 15 June 2023
"""

import os
import math
import glob
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    glossp = 50
    glosstype = 2
    #  drive = "./dataset/RWTHPHOENIXWeather2014"
    drive = "../dataset/RWTHPHOENIXWeather2014"
    glossdir = drive + f"/glosswithoutphase/gloss{glosstype}_files{glossp}p/"
    # list format
    listformat = [
                  f"anpoints10_{glosstype}_",
                  "allmcm"
                 ]

    # load all fpoints filenames to a list variable
    print("load data...")
    for root, dirs, files in os.walk(glossdir):
        for name in sorted(dirs):
            dirname = os.path.join(root, name)
            for f in listformat:
                listfiles = sorted(glob.glob(dirname + f"/{f}{glossp}p*.png"))
                if listfiles != []:
                    imgfiles = [cv.imread(f, cv.IMREAD_GRAYSCALE)
                                for f in listfiles]

                    numfiles = len(listfiles)
                    #  mhi = np.zeros_like(imgfiles[0])
                    #  for i, img in enumerate(imgfiles):
                    #      mhi[img > 0] = (((i+1) / numfiles) * 255)
                    mhib = np.zeros_like(imgfiles[0])
                    for i, img in enumerate(reversed(imgfiles)):
                        mhib[img > 0] = (((i+1) / numfiles) * 255)
                        

                    #  cv.imshow("mhi", mhi)
                    #  cv.imshow("mhib", mhib)
                    # print(dirname + f"/{f}mhi{glossp}p.png")
                    #  cv.imwrite(dirname + f"/{f}mhi{glossp}p.png", mhi)
                    print(dirname + f"/{f}mhib{glossp}p.png")
                    cv.imwrite(dirname + f"/{f}mhib{glossp}p.png", mhib)

                    #  k = cv.waitKey(0) & 0xFF
                    #  if k == ord('q'):
                    #     cv.destroyAllWindows()
                    #     exit()
