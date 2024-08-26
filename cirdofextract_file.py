"""
This file implements the separate CIRDOF extraction procedure.
In this file, the CIRDOF is calculated using and averaging value
from multiple frames.

This file is used to save the allmcm into an image file.

author: naserjawas
date: 29 October 2022
"""

import os
import glob
import cv2 as cv
import numpy as np
import math
import pickle

from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def display_multi_window(list_of_images, title, current_index, window_length):
    """
    Display images in multiple window.

    Parameters
    ----------
    list_of_images: an array of images to display.
    title: the title of the window.
    current_index: the latest index of the image array to display.
    window_length: how many images to display.

    Returns
    -------
    None
    """
    display_index = 0
    start_index = current_index - window_length
    for i in range(start_index, current_index + 1):
        cv.imshow("signpy: " + title + str(display_index), list_of_images[i])
        display_index += 1

def display_single_window(list_of_images, title, current_index, window_length):
    """
    Display images in single window.

    Parameters
    ----------
    list_of_images: an array of images to display.
    title: the title of the window.
    current_index: the latest index of the image array to display.
    window_length: how many images to display.

    Returns
    -------
    images: series of images as one image.
    """
    start_index = current_index - window_length
    for i in range(start_index, current_index + 1):
        img = list_of_images[i].copy()
        cv.putText(img, str(i+1), (10, 20),
                    cv.FONT_HERSHEY_PLAIN, 1, 255)
        if i == start_index:
            images = img
        else:
            images = np.concatenate((images, img), axis=1)
    cv.imshow(title, images)

    return images

def calc_optical_flow(prev_img, next_img, mode, prevof):
    """
    Calculate the optical flow of two images.

    Parameters
    ----------
    prev_img: the first image.
    next_img: the second image.
    mode:
        0: low-scale.
        1: high-scale.

    Returns
    -------
    mag: magnitude image.
    ang: angular image.
    """
    ### This part is for using Ferneback Optical Flow
    params0 = dict(numLevels=1,          # 1,
                   pyrScale=0.125,       # 0.125,
                   fastPyramids=True,    # True,
                   winSize=1,            # 1,
                   numIters=1,           # 1,
                   polyN=3,              # 5,
                   polySigma=1.1,        # 1.2,
                   flags=0)
    params1 = dict(numLevels=3,          # 3,
                   pyrScale=0.5,         # 0.5,
                   fastPyramids=True,    # True,
                   winSize=11,           # 15,
                   numIters=3,           # 3,
                   polyN=7,              # 5,
                   polySigma=1.5,        # 1.2,
                   flags=0)
    if mode == 0:
        of = cv.FarnebackOpticalFlow_create(**params0)
    else:
        of = cv.FarnebackOpticalFlow_create(**params1)
    flow = of.calc(prev_img, next_img, prevof)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    ###

    ### This part is using DIS Optical FLow
    # of = cv.DISOpticalFlow_create(cv.DISOpticalFlow_PRESET_MEDIUM)
    # flow = of.calc(prev_img, next_img, prevof)
    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    ###

    return mag, ang, flow


def create_ofimg(of_mag, of_ang):
    """
    Create the optical flow image from its magnitude and angle.

    Parameters
    ----------
    of_mag: optical flow magnitude.
    of_ang: optical flow angular.

    Returns
    -------
    of_bgr: optical flow image in bgr format.
    """
    ih, iw = of_mag.shape
    of_hsv = np.zeros((ih, iw, 3), dtype=np.uint8)
    of_hsv[..., 0] = np.abs(of_ang - 180)
    of_hsv[..., 1] = 255
    of_hsv[..., 2] = cv.normalize(of_mag, None, 0, 255, cv.NORM_MINMAX)
    of_bgr = cv.cvtColor(of_hsv, cv.COLOR_HSV2BGR)

    return of_bgr


def create_needle(of_mag, of_ang):
    """
    Create the optical flow needle diagram from its magnitude and angle.

    Parameters
    ----------
    of_mag: optical flow magnitude.
    of_ang: optical flow angular.

    Returns
    -------
    of_needle: optical flow needle diagram.
    """
    ih, iw = of_mag.shape
    of_needle = np.ones((ih, iw), dtype=np.uint8) * 255
    for i in range(0, ih, 5):
        for j in range(0, iw, 5):
            r = of_mag[i, j]
            # theta = np.abs(of_ang[i, j] - 180)
            theta = of_ang[i, j]

            x = r * math.cos(math.radians(theta))
            if not math.isinf(x):
                x = int(x)
            else:
                x = 0

            y = r * math.sin(math.radians(theta))
            if not math.isinf(y):
                y = int(y)
            else:
                y = 0

            #  if x != 0 or y != 0:
            #      cv.line(of_needle, (j, i), (j+x, i+y), 0, 1)
            cv.line(of_needle, (j, i), (j+x, i+y), 0, 1)

    return of_needle


def create_cirdof(of_bgr, k):
    """
    Create cirdof from optical flow image.

    Parameters
    ----------
    of_bgr: optical flow image in bgr format.
    k: number of clusters for k-means clustering.

    Returns
    -------
    cirdof: cirdof image.
    masks: cirdof separated based on the colours.
    """
    r = np.array(of_bgr, dtype=np.float64) / 255
    h, w, d = r.shape
    r = np.reshape(r, (w * h, d))

    sample = shuffle(r, random_state=0)[:1000]
    clusters = KMeans(n_clusters=k).fit(sample)
    labels = clusters.predict(r)

    codebook = clusters.cluster_centers_
    masks = []
    for c in codebook:
        mask = np.zeros((h, w), dtype=np.uint8)
        masks.append(mask)

    # codebook_d = codebook.shape[1]
    quan = np.zeros((h, w, d))
    label_id = 0
    for i in range(h):
        for j in range(w):
            quan[i][j] = codebook[labels[label_id]]
            masks[labels[label_id]][i][j] = 255
            label_id += 1
    cirdof = np.array(quan * 255, dtype=np.uint8)

    # set the cirdof image background to 0 (black)
    nonzero = [np.count_nonzero(mask) for mask in masks]
    max_nonzero = np.argmax(np.array(nonzero))
    cirdof[masks[max_nonzero] == 255] = 0
    masks.pop(max_nonzero)

    return cirdof, masks


def remove_background_mask(masks):
    """
    Remove the background mask from list of mask.

    Parameters
    ----------
    masks: a list of mask.

    Returns
    -------
    masks: a list of mask after background mask deleted.
    """
    nonzero = [np.count_nonzero(mask) for mask in masks]
    max_nonzero = np.argmax(np.array(nonzero))
    masks.pop(max_nonzero)

    return masks


def create_clusters(arr, k):
    """
    Create clusters from 2d array.

    Parameters
    ----------
    arr: input 2D array
    k: number of cluster

    Returns
    -------
    cluster: 2D array of clusters
    masks: the list of masks that are decomposed from cluster
    """
    arr = np.nan_to_num(arr)
    h, w = arr.shape
    d = 1
    r = np.reshape(arr, (h * w, d))

    sample = shuffle(r, random_state=0)[:1000]
    clusters = KMeans(n_clusters=k).fit(sample)
    labels = clusters.predict(r)

    codebook = clusters.cluster_centers_
    masks = []
    for c in codebook:
        mask = np.zeros((h, w), dtype=np.uint8)
        masks.append(mask)

    # codebook_d = codebook.shape[1]
    quan = np.zeros((h, w))
    label_id = 0
    for i in range(h):
        for j in range(w):
            quan[i][j] = codebook[labels[label_id]]
            masks[labels[label_id]][i][j] = 255
            label_id += 1
    cluster = np.array(quan * 255, dtype=np.uint8)

    return cluster, masks


def refine_masks(masks, fxc, fyc):
    """
    Refine the masks using contour detection.

    Parameters
    ----------
    masks: the original masks from create_clusters.

    Returns
    -------
    newmasks: the refined masks.
    cirdofbw: the flatten masks.
    """
    newmasks = []

    # separate the mask using contour detection
    for mid, mask in enumerate(masks):
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            facepass = True
            if fxc > 0 and fyc > 0:
                if cv.pointPolygonTest(contours[c], (fxc, fyc), False) > 0:
                    facepass = False

            # FIXME: this (25 * 25) must be from face detection
            if cv.contourArea(contours[c]) > (25 * 25) and facepass:
                newmask = np.zeros_like(mask)
                cv.drawContours(newmask, contours, c, 255, -1)
                newmasks.append(newmask)

    if len(newmasks) > 0:
        newmasks = sorted(newmasks, key=lambda v: cv.countNonZero(v),
                          reverse=True)

    return newmasks


def create_cirdofbw(masks, ih, iw):
    """
    Create cirdofbw from contour masks that are generated by refine_masks.

    Parameters
    ----------
    masks: a list of mask.

    Returns
    -------
    cirdofbw: an image that shows a flatten version from the list of masks.
    """
    cirdofbw = np.zeros((ih, iw), dtype=np.uint8)
    # build cirdofbw from newmasks
    for mid, mask in enumerate(masks):
        cirdofbw += mask
    cirdofbw[cirdofbw > 0] = 255

    return cirdofbw


def get_skinmask(cascade_classifier, image,
                 lowerhsv, upperhsv, lowerycrcb, upperycrcb, fxc, fyc):
    """
    Get the skinmask

    Parameters
    ----------
    cascade_classifier: face detector
    image: image in RGB format
    lower: the lower colour threshold
    upper: the upper colour threshold
    fxc: face centre x coordinate
    fyc: face centre y coordinate

    Returns
    -------
    skinmask: skin masks in one image.
    skinmasks: skin masks in a list.
    facemask: face mask in one image.
    lower: the new lower colour threshold.
    upper: the new upper colour threshold.
    fxc: the new face centre x coordinate.
    fyc: the new face centre y coordinate.
    """
    imagehsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    imageycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    imagegrey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    facemask = np.zeros_like(imagegrey)
    skinmask = np.zeros_like(imagegrey)
    skinmaskycrcb = np.zeros_like(imagegrey)
    skinmaskhsv = np.zeros_like(imagegrey)
    (fx0, fy0, fw, fh) = get_face(cascade_classifier, imagegrey)
    if fx0 > 0 and fy0 > 0 and fw > 0 and fh > 0:
        newfxc = int(fx0 + (fw // 2))
        newfyc = int(fy0 + (fh // 2))
        # FIXME: this (25 * 25) must be from face detection
        if fxc != 0 and fyc != 0:
            if math.fabs(newfxc - fxc) < 10:
                fxc = newfxc
            if math.fabs(newfyc - fyc) < 10:
                fyc = newfyc
        else:
            fxc = newfxc
            fyc = newfyc

        sx0, sy0, sx1, sy1 = calc_rect(fxc, fyc, (fw // 4))

        coloursample = imageycrcb[sy0:sy1, sx0:sx1]
        mean, stddev = cv.meanStdDev(coloursample)
        mean = mean.astype(int).reshape(1, 3)
        stddev = stddev.astype(int).reshape(1, 3)
        lowerycrcb = mean - stddev * 3
        upperycrcb = mean + stddev * 3
        skinmaskycrcb = cv.inRange(imageycrcb, lowerycrcb, upperycrcb)
        skinmaskycrcb[skinmaskycrcb > 0] = 1

        coloursample = imagehsv[sy0:sy1, sx0:sx1]
        mean, stddev = cv.meanStdDev(coloursample)
        mean = mean.astype(int).reshape(1, 3)
        stddev = stddev.astype(int).reshape(1, 3)
        lowerhsv = mean - stddev * 3
        upperhsv = mean + stddev * 3
        skinmaskhsv = cv.inRange(imagehsv, lowerhsv, upperhsv)
        skinmaskhsv[skinmaskhsv > 0] = 1

    skinmask = skinmaskycrcb + skinmaskhsv
    skinmask[skinmask > 0] = 255

    # redraw contourmask from skinmask
    contourmask = np.zeros_like(imagegrey)
    contours, hierarchy = cv.findContours(skinmask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        # FIXME: this (25 * 25) must be from face detection
        if cv.contourArea(contours[c]) > (25 * 25):
            cv.drawContours(contourmask, contours, c, 255, -1)

    # decompose skinmask
    skinmasks = []
    contours, hierarchy = cv.findContours(contourmask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    newskinmask = np.zeros_like(skinmask)
    for c in range(len(contours)):
        facepass = True
        if fxc > 0 and fyc > 0:
            if cv.pointPolygonTest(contours[c], (fxc, fyc), False) > 0:
                facepass = False
                cv.drawContours(facemask, contours, c, 255, -1)

        if facepass:
            newmask = np.zeros_like(skinmask)
            cv.drawContours(newmask, contours, c, 255, -1)
            skinmasks.append(newmask)

        cv.drawContours(newskinmask, contours, c, 255, -1)

    skinmask = newskinmask

    return (skinmask, skinmasks, facemask,
            lowerhsv, upperhsv, lowerycrcb, upperycrcb, fxc, fyc)


def combine_masks(magmasks, angmasks, skinmasks0, skinmasks1,
                  facemasks, magarr, multiclustermag):
    """
    Combine the masks to produce the hand candidate location.
    NOTE: it replaces the compare_masks and extract_highmag_masks function.

    Parameters
    ----------
    magmasks: magnitude masks.
    angmasks: angular masks.
    skinmasks0: skin masks from t.
    skinmasks1: skin masks from t+1.
    facemasks: face masks residue
    magarr: original magnitude array.
    multiclustermag: multicluster magnitude array.

    Returns
    -------
    retmasks: return combined masks.
    probmask: probability mask
    """
    retmasks = []
    magangmasks = []
    for magid, magmask in enumerate(magmasks):
        for angid, angmask in enumerate(angmasks):
            intersection = magmask * angmask
            intersectcount = cv.countNonZero(intersection)
            # FIXME: this (25 * 25) must be from face detection
            if intersectcount > (25 * 25):
                intersection[intersection > 0] = 255
                magangmasks.append(intersection)

    highmags = []
    for magangid, magangmask in enumerate(magangmasks):
        magangmask[magangmask > 0] = 1
        magval = magangmask * magarr
        summagval = np.sum(magval)
        highmags.append((magangid, summagval))
        magangmask[magangmask > 0] = 255

    highmags = sorted(highmags, key=lambda v: v[1], reverse=True)

    highmagmasks = []
    for hm in highmags:
        mask = magangmasks[hm[0]]
        if highmagmasks == []:
            highmagmasks.append(mask)
        else:
            intersectcount = 0
            for highmagmask in highmagmasks:
                intersectionmask = highmagmask * mask
                if cv.countNonZero(intersectionmask) > 0:
                    intersectcount += 1
            if intersectcount == 0:
                highmagmasks.append(mask)

    # filter retmasks with skin
    # create a combination mask from all skinmasks
    allskinmasks = np.zeros_like(magarr, dtype=np.uint8)
    for sid, skinmask in enumerate(skinmasks0):
        allskinmasks[skinmask > 0] += 1
    for sid, skinmask in enumerate(skinmasks1):
        allskinmasks[skinmask > 0] += 1
    allskinmasks[allskinmasks > 0] = 1

    # check only if there is a skin area detected
    if cv.countNonZero(allskinmasks) > 0:
        for hm in highmagmasks:
            check = allskinmasks * hm
            if cv.countNonZero(check) > 0:
                retmasks.append(hm)
    else:
        retmasks = highmagmasks.copy()

    # create probability mask
    probmask = np.zeros_like(magarr)
    n = len(retmasks)
    for rid, retmask in enumerate(retmasks):
        probmask[retmask > 0] = (n-rid)

    for sid, skinmask in enumerate(skinmasks0):
        probmask[skinmask > 0] += 1

    for sid, skinmask in enumerate(skinmasks1):
        probmask[skinmask > 0] += 1

    for fid, facemask in enumerate(facemasks):
        probmask[facemask > 0] += 1

    # use multiclustermag
    t = 0.75 * 255
    probmask[multiclustermag > t] += 1

    # probmask = probmask * minmax_scale(np.nan_to_num(of_mag))
    probmask = (probmask / np.max(probmask)) * 255
    probmask = probmask.astype(np.uint8)

    return retmasks, probmask


def get_face(cascade_classifier, image):
    """
    Get the face location in an image using cascade classifier.

    Parameters
    ----------
    cascade_classifier:
    image:

    Returns
    -------
    face_x0
    face_y0
    face_width
    face_height
    """
    faces = cascade_classifier.detectMultiScale(image)
    if len(faces) != 0:
        fx, fy, fw, fh = faces[0]
    else:
        fx, fy, fw, fh = 0, 0, 0, 0

    return fx, fy, fw, fh


def calc_rect(xc, yc, span):
    """
    Calculate rectangle (x0, y0) and (x1, y1) from centre (xc, yc)

    Parameters
    ----------
    xc,yc: centre coordinate
    span: the length from centre to side of rectangle.

    Returns
    -------
    x0, y0: first coordinate
    x1, y1: second coordinate
    """
    x0 = xc - span
    y0 = yc - span
    x1 = xc + span
    y1 = yc + span

    return x0, y0, x1, y1


def multiscaleoperation(dataarray, kset=[2, 3, 4], rmbg=True, redraw=True):
    """
    Do multiscale operation with different clusters number

    Parameters
    ----------
    dataarray: original array to be clustered.
    kset: set of number of cluster for clustering the data.
    rmbg: boolean flag for remove_background_mask.
    redraw: boolean flag for redraw contours.

    Returns
    -------
    allmaskbin: clustering result in a form of probability mask.
    """
    allmasksbin = np.zeros_like(dataarray, dtype=np.uint8)
    for k in kset:
        allmask = np.zeros_like(dataarray, dtype=np.uint8)
        cluster, masks = create_clusters(dataarray, k)
        if rmbg:
            masks = remove_background_mask(masks)
        # redraw contour
        for mask in masks:
            contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL,
                                                  cv.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                # FIXME: this (25 * 25) must be from face detection
                if cv.contourArea(contours[c]) > (25 * 25):
                    if redraw:
                        newmask = np.zeros_like(mask)
                        cv.drawContours(newmask, contours, c, 255, -1)
                        allmasksbin[newmask > 0] += 1
                        allmask[newmask > 0] += 1
                    else:
                        allmasksbin[mask > 0] += 1
                        allmask[mask > 0] += 1
        allmask = (allmask / np.max(allmask)) * 255
        allmask = allmask.astype(np.uint8)
        # cv.imwrite(f"allmask_{k}.jpeg", allmask)

    allmasksbin = (allmasksbin / np.max(allmasksbin)) * 255
    allmasksbin = allmasksbin.astype(np.uint8)
    # cv.imwrite(f"allmask_234.jpeg", allmasksbin)

    return allmasksbin


def analyseskinmask(skinmask, fxc, fyc, of_mag, of_ang):
    """
    Analyse skin mask using moments and face location and magnitude and angle.

    Parameters
    ----------
    skinmask: skinmasks
    fxc: face x coordinate
    fyc: face y coordinate
    of_mag: magnitude array
    of_ang: angle array

    Returns
    -------
    skinmaskcolour: new skin mask

    """
    skinmaskcolour = cv.cvtColor(skinmask, cv.COLOR_GRAY2BGR)
    cv.circle(skinmaskcolour, (fxc, fyc), 2, (0, 0, 255), -1)

    contours, hierarchy = cv.findContours(skinmask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        if cv.pointPolygonTest(contours[c], (fxc, fyc), False) < 0:
            tempmask = np.zeros_like(skinmask)
            cv.drawContours(tempmask, contours, c, 1, -1)
            tempmag = tempmask * of_mag
            tempang = tempmask * of_ang
            meanmag = np.sum(tempmag) / cv.countNonZero(tempmask)
            meanang = np.sum(tempang) / cv.countNonZero(tempmask)
            x = meanmag * math.cos(math.radians(meanang))
            y = meanmag * math.sin(math.radians(meanang))
            if not math.isinf(x) and not math.isnan(x):
                x = int(x)
            else:
                x = 0
            if not math.isinf(y) and not math.isnan(y):
                y = int(y)
            else:
                y = 0

            mm = cv.moments(contours[c])
            mcx = int(mm['m10'] / mm['m00'])
            mcy = int(mm['m01'] / mm['m00'])
            cv.circle(skinmaskcolour, (mcx, mcy), 2, (255, 0, 0), -1)

            cv.line(skinmaskcolour, (mcx, mcy), (mcx+x, mcy+y), (255, 0, 0), 1)

    return skinmaskcolour


def cleanmask(mask, minx, miny):
    """
    Draw masks from contour to clean the masks.

    Parameters
    ----------
    mask: the mask image
    minx: the minimum width of each contour
    miny: the minimum height of each contour.
    Returns
    -------
    newmask: the cleaned mask
    """
    newmask = np.zeros_like(mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        if cv.contourArea(contours[c]) > (minx * miny):
            cv.drawContours(newmask, contours, c, 255, -1)

    return newmask


def get_facemask(facemask0, facemask1, facemask, facemaskbin):
    """
    Get facemask in each frame.

    Parameters
    ----------
    facemask0: facemask from frame0
    facemask1: facemask from frame1
    facemask: facemask from previous cycle
    facemaskbin: facemask as a binary image.
    Returns
    -------
    facemasks: list of facemask
    facemask: facemask result
    facemaskbin: facemask as a binary image.
    faceprobbin: the probability of face area.
    """
    facemaskbin[facemask0 > 0] += 1
    facemaskbin[facemask1 > 0] += 1

    faceprob = (facemaskbin / np.max(facemaskbin))
    faceprobbin = np.ones_like(faceprob, dtype=np.uint8)
    faceprobbin[faceprob > 0.5] = 0

    kernel = np.ones((3, 3), np.uint8)
    faceprobbin = cv.morphologyEx(faceprobbin, cv.MORPH_ERODE, kernel,
                                  iterations=5)

    facemask0 = facemask0 * faceprobbin
    facemask0[facemask0 > 0] = 255
    facemask0 = cleanmask(facemask0, 25, 25)

    facemask1 = facemask1 * faceprobbin
    facemask1[facemask1 > 0] = 255
    facemask1 = cleanmask(facemask1, 25, 25)

    facemasks = [facemask0, facemask1]

    facemask = (facemaskbin / np.max(facemaskbin)) * 255
    facemask = facemask.astype(np.uint8)

    return facemasks, facemask, facemaskbin, faceprobbin


def refine_skinmask(skinmask0, skinmask1, faceprobbin):
    """
    Refine skinmask using face probability.

    Parameters
    ----------
    skinmask0: skinmask from frame0
    skinmask1: skinmask from frame1
    faceprobbin: probability mask of face area.
    Returns
    -------
    newskinmask: refined skinmask
    """
    newskinmask0 = skinmask0 * faceprobbin
    newskinmask0 = cleanmask(newskinmask0, 25, 25)
    newskinmask1 = skinmask1 * faceprobbin
    newskinmask1 = cleanmask(newskinmask1, 25, 25)

    newskinmask = np.zeros_like(skinmask0)
    newskinmask = newskinmask0 + newskinmask1
    newskinmask[newskinmask > 0] = 255

    return newskinmask


def refine_multiclustermag(multiclustermag):
    """
    Refine multi cluster magnitude by contour detection.

    Parameters
    ----------
    multiclustermag: original multiclustermag
    Returns
    -------
    newmultimag: refined multiclustermag
    """
    newmultimag = np.zeros_like(multiclustermag)
    tu = np.unique(multiclustermag)
    for tid, t in enumerate(tu):
        tmask = multiclustermag.copy()
        tmask[tmask <= t] = 0
        tmask[tmask > t] = 255
        contours, hierarchy = cv.findContours(tmask, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        for c in range(len(contours)):
            cx, cy, cw, ch = cv.boundingRect(contours[c])
            if (cw/iw) > 0.75 or (ch/ih) > 0.75:
                cv.rectangle(tmask, (cx, cy), (cx+cw, cy+ch), 255, 1)
            else:
                newmask = np.zeros_like(tmask)
                cv.drawContours(newmask, contours, c, 1, -1)
                newmultimag[newmask > 0] += 1

    newmultimag = (newmultimag / np.max(newmultimag)) * 255
    newmultimag = newmultimag.astype(np.uint8)

    return newmultimag


def refine_newmultimag(newmultimag, sctr, allmctr, pctr):
    """
    Refine multimag with contours from current and previous multimag.

    Parameters
    ----------
    newmultimag: multimag from current frame.
    sctr: skin contour
    allmctr: mag contours from current frame.
    pctr: mag contour from previous frame.
    Returns
    -------
    newmultimag: refined multimag
    """
    sc = np.zeros_like(newmultimag)
    if len(sctr) > 0:
        # print("using skinmask")
        for c0 in range(len(sctr)):
            sc0 = np.zeros_like(newmultimag)
            cv.drawContours(sc0, sctr, c0, 1, -1)
            sc0area = cv.countNonZero(sc0)
            ratiolist = []

            for c1 in range(len(allmctr)):
                mc1 = np.zeros_like(newmultimag)
                cv.drawContours(mc1, allmctr, c1, 1, -1)
                mc1area = cv.countNonZero(mc1)
                scmc = sc0 * mc1
                scmcarea = cv.countNonZero(scmc)
                ratio = ((0.3 * (scmcarea / sc0area)) +
                         (0.7 * (scmcarea / mc1area)))
                ratiolist.append((c1, ratio))

            if len(ratiolist) > 0:
                ratiolist = sorted(ratiolist, key=lambda v: v[1], reverse=True)
                c1 = ratiolist[0][0]
                cv.drawContours(sc, allmctr, c1, 1, -1)
    else:
        # print("NOT using skinmask")
        for c1 in range(len(allmctr)-1, -1, -1):
            newsc = np.zeros_like(newmultimag)
            cv.drawContours(newsc, allmctr, c1, 1, -1)
            testsc = (sc + newsc) * newsc
            ratio = 0.0
            for c2 in range(len(pctr)):
                pc2 = np.zeros_like(newmultimag)
                cv.drawContours(pc2, pctr, c2, 1, -1)
                pc2area = cv.countNonZero(pc2)
                pcmc = pc2 * newsc
                pcmcarea = cv.countNonZero(pcmc)
                ratio = pcmcarea / pc2area

            if np.max(testsc) <= 3 and ratio > 0.7:
                # print(np.max(testsc), ratio)
                sc = sc + newsc

    newmultimag = sc.copy()
    newmultimag[newmultimag > 0] = 255

    return newmultimag


def extractContour(f0, fp1, fm1, iw, ih):
    """
    Analyse the contour from 3 frames.

    data: [ctr, cctr]
        ctr: contours
        cctr: [mcx, mcy, x, y, magmean, angmean]

    Parameters
    ----------
    f0 : middle frame data.
    fp1 : plus 1 frame data.
    fm1 : minus 1 frame data.
    Returns
    -------
    fr: return middle frame data.
    """
    outimg = np.zeros((ih, iw), dtype=np.uint8)
    outimg2 = np.zeros((ih, iw), dtype=np.uint8)

    firstintersect = []
    secondintersect = []
    intersectmag = []
    frm = []
    # calculate the f0 directions from fm1
    for j, m in enumerate(fm1[1]):
        mx = m[0]
        my = m[1]
        # print("m:", m)
        img_m = np.zeros((ih, iw), dtype=np.uint8)
        cv.drawContours(img_m, fm1[0], j, 1, -1)

        for i, x in enumerate(f0[1]):
            xx = x[0]
            xy = x[1]
            # calculate angle direction
            xt = math.degrees(math.atan2((xy-my), (xx-mx)))
            if xt < 0:
                xt = 360 + xt
            # calculate precentage angle difference between real and data
            diffxt = (math.fabs(xt-m[5]) / 360) * 100

            # calculate magnitude difference
            xu = math.sqrt(math.pow((xx-mx), 2) + math.pow((xy-my), 2))
            # calculate precentage magnitude difference between real and data
            diffxu = (xu-m[4]) / m[4]
            # print("mag m:", xu, m[4], diffxu)

            # check union area
            # img_0 = np.zeros((ih, iw), dtype=np.uint8)
            # cv.drawContours(img_0, f0[0], i, 1, -1)
            # um0 = img_m * img_0
            # nz_um0 = np.count_nonzero(um0)
            # if nz_um0 > 0:
            #     firstintersect.append(i)
            #     intersectmag.append([m[4], x[4]])
            #-----------------

            # if the percentage difference less than 12.5 %
            if diffxt < 12.5 and x[4] >= 1 and diffxu <= 2:
                # print("[", xx, xy, "]", xt, diffxt, xu, diffxu)
                frm.append([f0[0][i], f0[1][i]])
                cv.drawContours(outimg, f0[0], i, 255, -1)

    frp = []
    for j, p in enumerate(fp1[1]):
        px = p[0]
        py = p[1]
        # print("p:", p)
        img_p = np.zeros((ih, iw), dtype=np.uint8)
        cv.drawContours(img_p, fp1[0], j, 1, -1)

        for i, x in enumerate(f0[1]):
            xx = x[0]
            xy = x[1]
            # calculate angle direction
            xt = math.degrees(math.atan2((xy-py), (xx-px)))
            if xt < 0:
                xt = 360 + xt
            if xt < 180:
                xt = xt + 180
            else:
                xt = xt - 180
            # calculate precentage angle difference between real and data
            diffxt = (math.fabs(xt-p[5]) / 360) * 100

            # calculate magnitude difference
            xu = math.sqrt(math.pow((xx-px), 2) + math.pow((xy-py), 2))
            # calculate precentage magnitude difference between real and data
            diffxu = (xu-x[4]) / x[4]
            # print("mag f:", xu, x[4], diffxu)

            # check union area
            # img_0 = np.zeros((ih, iw), dtype=np.uint8)
            # cv.drawContours(img_0, f0[0], i, 1, -1)
            # up0 = img_p * img_0
            # nz_up0 = np.count_nonzero(up0)
            # if nz_up0 > 0:
            #     if i in firstintersect and i not in secondintersect:
            #         id0 = firstintersect.index(i)
            #         secondintersect.append(i)
            #         intersectmag[id0].append(p[4])
            #-----------------

            # if the percentage difference less than 12.5 %
            if diffxt < 12.5 and x[4] >= 1 and diffxu <= 2:
                # print("[", xx, xy, "]", xt, diffxt, xu, diffxu)
                frp.append([f0[0][i], f0[1][i]])
                cv.drawContours(outimg, f0[0], i, 255, -1)

    # check union area
    # print(secondintersect)
    # for id0, i in enumerate(secondintersect):
    #     print(i, intersectmag[id0])
    #     cv.drawContours(outimg2, f0[0], i, 255, -1)
    # cv.imshow("outimg2", outimg2)
    #-----------------
    # NOTE: on the "check union area"
    #       This part of the code is used to check the
    #       continuity between frames. But it still needs
    #       good condition to catch before using this feature.

    return outimg


def create_allbin(bins, start, stop):
    allbin = np.zeros_like(bins[0])
    for i, b in enumerate(bins[start:stop]):
        if np.count_nonzero(b) == 0 and i > 1:
            allbin[allbin == i-1] = (i + 1)
        else:
            allbin[b > 0] = (i + 1)
    n = stop-start
    for i, b in enumerate(bins[start:stop]):
        allbin[allbin == (i + 1)] = int(((i + 1) / n) * 255)

    return allbin


def get_point_cloud(mag, mask, n, d):
    pointmap = np.zeros_like(mag, dtype=np.uint8)
    points = []

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        magcopy = mag.copy()
        # minval = np.min(magcopy)
        minval = 0
        cmap = np.zeros_like(mask)
        cv.drawContours(cmap, contours, c, 1, -1)
        magcopy = magcopy * cmap
        numpoints = 0
        while(numpoints < n):
            includeit = True
            maxval = np.max(magcopy)
            maxloc = np.unravel_index(np.argmax(magcopy), magcopy.shape)

            if maxval == minval:
                break

            for p in points:
                z = math.sqrt(math.pow((p[0]-maxloc[0]), 2) +
                              math.pow((p[1]-maxloc[1]), 2))
                if z < (d - (d - (numpoints / d))):
                    includeit = False

            if includeit:
                points.append(maxloc)
                numpoints += 1
                y, x = maxloc
                # cv.circle(mask, (x, y), 1, 200, -1)

            magcopy[maxloc] = minval

    return points


def get_point_cloud2(mag, mask, n, d):
    pointmap = np.zeros_like(mag, dtype=np.uint8)
    points = []

    magcopy = mag.copy()
    # minval = np.min(magcopy)
    minval = 0
    cmap = np.zeros_like(mask)
    numpoints = 0
    while(numpoints < n):
        includeit = True
        maxval = np.max(magcopy)
        maxloc = np.unravel_index(np.argmax(magcopy), magcopy.shape)

        if maxval == minval:
            break

        for p in points:
            z = math.sqrt(math.pow((p[0]-maxloc[0]), 2) +
                            math.pow((p[1]-maxloc[1]), 2))
            if z < (d - (d - (numpoints / d))):
                includeit = False

        if includeit:
            points.append(maxloc)
            numpoints += 1
            y, x = maxloc
            cv.circle(mask, (x, y), 1, 200, -1)

        magcopy[maxloc] = minval

    return mask


def get_point_cloud3(mag, ang, probmask, n):
    pointmap = np.zeros_like(mag, dtype=np.uint8)
    points = []

    magcopy = mag.copy()
    probval = np.unique(probmask).tolist()
    lendata = len(probval) - 1
    sumdata = sum(list(range(len(probval))))

    # calculate number of points to generate in each level.
    numofpoints = [int(((i+1) / sumdata) * n) for i in range(lendata)]
    d = sum(numofpoints) - n
    if d < 0:
        numofpoints[-1] += abs(d)
    d = sum(numofpoints) - n

    # calculate the coefficient of distance for each level.
    dd = [i for i in range(len(probval))]
    dd = dd[::-1]
    dd = dd[:-1]

    for i, p in enumerate(probval[1:]):
        mask = probmask.copy()
        mask[mask != probval[i+1]] = 0
        mask[mask == probval[i+1]] = 1

        magmasked = mag * mask
        minval = 0
        curnumofpoints = 0

        while(curnumofpoints < numofpoints[i]):
            includeit = True
            maxval = np.max(magmasked)
            maxloc = np.unravel_index(np.argmax(magmasked), magmasked.shape)
            y, x = maxloc
            magyx = mag[y, x]
            angyx = ang[y, x]

            if maxval == minval:
                break

            for p in points:
                z = math.sqrt(math.pow((p[0] - y), 2) +
                              math.pow((p[1] - x), 2))
                if z < (dd[i] * 3):
                    includeit = False

            if includeit:
                points.append((y, x, magyx, angyx))
                # print((y, x, magyx, angyx))
                curnumofpoints += 1
                # cv.circle(probmask, (x, y), 1, 0, -1)

            magmasked[y, x] = minval
    # print(len(points), "points generated")

    # return probmask
    return points

def get_point_cloud4(probmask, n, fx0, fy0, fw, fh):
    fxc = int(fx0 + math.ceil(fw / 2))
    fyc = int(fy0 + math.ceil(fh / 2))

    bx0 = fx0 - fw
    by0 = fy0 - math.ceil(fh / 2)
    bx1 = bx0 + int(fw * 3)
    by1 = by0 + int(fh * 5)

    mask = probmask.copy()
    maxy, maxx = mask.shape
    numx = n * 3
    numy = n * 5

    stepx = math.ceil(fw / n)
    stepy = math.ceil(fh / n)

    if bx0 < 0:
        bx0 = 0
    if by0 < 0:
        by0 = 0
    if bx1 > (maxx - stepx):
        bx1 = maxx - stepx - 1
    if by1 > (maxy - stepy):
        by1 = maxy - stepy - 1

    ibox = 0
    points = []
    county = 0
    for y in range(by0, by1, stepy):
        countx = 0
        for x in range(bx0, bx1, stepx):
            # print("x, y", x, y, x+stepx, y+stepy, ibox, stepx, stepy)
            # cv.putText(mask, str(ibox), (x+5,y+15), cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.circle(mask, (x, y), 2, 255, -1)
            # cv.rectangle(mask, (x, y), (x+stepx, y+stepy), 255, 1)
            smask = mask[y:y+stepy, x:x+stepx]
            usmask = np.max(smask)
            # print(smask.shape, usmask)
            points.append(usmask)
            ibox += 1
            countx += 1
        if countx <= numx:
            for cx in range(countx, numx):
                points.append(0)
                ibox += 1
        county += 1
    if county <= numy:
        for cy in range(county, numy):
            for cx in range(numx):
                points.append(0)
                ibox += 1

    # cv.rectangle(mask, (fx0, fy0), (fx0+fw, fy0+fh), 255, 1)
    # cv.rectangle(mask, (bx0, by0), (bx1, by1), 255, 1)
    # cv.imshow("probmask", mask)
    # print(len(points), "points generated")

    return points

def get_point_cloud5(probmask, magarray, n, fx0, fy0, fw, fh):
    fxc = int(fx0 + (fw // 2))
    fyc = int(fy0 + (fh // 2))

    bx0 = fx0 - fw
    by0 = fy0 - int(fh // 2)
    bx1 = bx0 + fw * 3
    by1 = by0 + int(fh * 5)

    mask = probmask.copy()
    ibox = 0
    points = []
    for y in range(by0, by1, int(fh//n)):
        if (y + int(fh//n)) > by1:
            continue

        for x in range(bx0, bx1, int(fw//n)):
            if (x + int(fw//n)) > bx1:
                continue

            # print("x,y", x, y, x+fw, y+fh, ibox)
            # cv.putText(mask, str(ibox), (x+5,y+15), cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.circle(mask, (x, y), 2, 255, -1)
            smask = mask[y:y+fh, x:x+fw].copy()
            mmask = magarray[y:y+fh, x:x+fw].copy()
            usmask = np.unique(smask)
            if len(usmask) > 0:
                t = usmask[-1]
                smask[smask < t] = 0
                smask[smask >= t] = 1
                mmask[smask != 1] = 0
                magmean = np.mean(mmask)
            else:
                magmean = 0.0
            points.append(magmean)
            # print(len(usmask))
            ibox += 1

    # cv.rectangle(mask, (fx0, fy0), (fx0+fw, fy0+fh), 255, 1)
    # cv.rectangle(mask, (bx0, by0), (bx1, by1), 255, 1)
    # cv.imshow("probmask", mask)
    # print(len(points), "points generated")

    return points



if __name__ == "__main__":
    print("Load dataset...")
    # drive = "/run/media/naser/EXT0/RWTHPHOENIXWeather2014"
    # drive = "/home/u1899243/PhDResearch/workingdir/dataset/RWTHPHOENIXWeather2014"
    #  drive = "./dataset/RWTHPHOENIXWeather2014"
    drive = "../dataset/RWTHPHOENIXWeather2014"
    datadir = drive + "/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/ch5/"

    datanames = sorted(os.listdir(datadir))
    if datanames[0] == ".DS_Store":
        datanames = datanames[1:]
    print(len(datanames), "data available...")
    iw, ih = 210, 300

    # print("Load face detector...")
    cascade_file = "./data/haarcascade_frontalface_default.xml"
    cascade_classifier = cv.CascadeClassifier()
    cascade_classifier.load(cv.samples.findFile(cascade_file))

    fbhsv = []    # face bottom colour
    fthsv = []    # face top colour
    fbycrcb = []  # face bottom colour
    ftycrcb = []  # face top colour
    fxc = 0       # face x centre
    fyc = 0       # face y centre
    prevfx0, prevfy0, prevfw, prevfh = 0, 0, 0, 0

    # first trial
    # datanamespart = datanames[:100]
    # datanamespart = datanames[100:250]
    # datanamespart = datanames[250:500]
    # second trial

    # datanamespart = datanames[0:250]
    # datanamespart = datanames[250:500]
    # datanamespart = datanames[500:750]
    # datanamespart = datanames[750:1000]

    # datanamespart = datanames[:500]
    # datanamespart = datanames[500:1000]
    # datanamespart = datanames[1000:1500]
    # datanamespart = datanames[1500:2000]
    # datanamespart = datanames[2000:3000]
    # datanamespart = datanames[3000:4000]
    # datanamespart = datanames[4000:5000]
    # datanamespart = datanames[5000:]
    # # datanamespart = datanames[5:6] # example of error result on thesis.
    # # datanamespart = datanames[6:7]
    # # datanamespart = datanames[13:14]
    # # datanamespart = datanames[10:11] # example of good result on thesis.
    datanamespart = datanames[2:3]
    # NEXT
    for dataname_i, dataname in enumerate(datanamespart):
        dataloc = datadir + dataname + "/1/*-0.png"
        datafiles = sorted(glob.glob(dataloc))
        lendatafiles = len(datafiles)
        print("(", dataname_i + 1, "/", len(datanamespart),")",
              dataname, "has", lendatafiles, "frames")
        dataframes = [cv.imread(filename, cv.IMREAD_COLOR)
                      for filename in datafiles]
        facemask = np.zeros((ih, iw), dtype=np.float32)
        facemaskbin = np.zeros((ih, iw), dtype=np.uint16)

        # load tracking points
        tpoints = []
        picklefile = datadir + dataname + "/1/tpoints1"
        with open(picklefile, "rb") as pf:
            tpoints = pickle.load(pf)
            # print("loadpickle:", loadpickle)
        pf.close()
        print(picklefile, "loaded.")

        prevskinmask = []
        prevmultimag = []
        prevmultimagflt = []
        prevmultimagdata = []
        prevbins = []
        prevof = []
        prevmcm = []
        prevsummag = []
        prevsumpxl = []
        prevratpxl = []

        fpoints = []
        fx0, fy0, fw, fh = 0, 0, 0, 0
        lfx0 = []
        lfy0 = []
        lfw = []
        lfh = []

        for fid, frame in enumerate(dataframes):
            # print(f"fid: {fid} / {lendatafiles}")
            if fid == 0:
                continue
            frame = cv.resize(frame, (iw, ih))
            frame0 = cv.resize(dataframes[fid-1], (iw, ih))
            frame1 = cv.resize(dataframes[fid], (iw, ih))
            grey0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
            grey1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            # cv.imwrite("frame1new.jpeg", frame1)

            # tracking point
            tpoint = tpoints[fid-1]
            # tracking mask
            tmask = np.zeros((ih, iw), dtype=np.uint8)
            tmask0 = np.zeros((ih, iw), dtype=np.uint8)
            tmask1 = np.zeros((ih, iw), dtype=np.uint8)
            if tpoint[1] != []:
                for i in range(3, 0, -1):
                    for t in tpoint[1]:
                        if t[1] != [] and t[0] == 0:
                            cv.circle(tmask, t[1], (i*25), int(1/i * 255), -1)
                            cv.circle(tmask0, t[1], (i*25), int(1/i * 255), -1)
                            x, y = t[1]
                            x0 = x - 25
                            y0 = y - 25
                            x1 = x + 25
                            y1 = y + 25
                            # cv.rectangle(frame1, (x0, y0), (x1, y1), (0, 0, 255), 1)
                        if t[1] != [] and t[0] == 1:
                            cv.circle(tmask, t[1], (i*25), int(1/i * 255), -1)
                            cv.circle(tmask1, t[1], (i*25), int(1/i * 255), -1)
                            x, y = t[1]
                            x0 = x - 25
                            y0 = y - 25
                            x1 = x + 25
                            y1 = y + 25
                            # cv.rectangle(frame1, (x0, y0), (x1, y1), (0, 0, 255), 1)

            # calculate optical flow
            pof = None
            if len(prevof) > 0:
                pof = prevof[-1]
            else:
                pof = None
            of_mag, of_ang, of = calc_optical_flow(grey0, grey1, 1, pof)
            prevof.append(of)

            # create optical flow image (BGR format)
            of_bgr = create_ofimg(of_mag, of_ang)

            # create CIRDOF and its masks
            cirdof, masks = create_cirdof(of_bgr, 3)

            # refine masks and create CIRDOFbw
            newmasks = refine_masks(masks, fxc, fyc)
            cirdofbw = create_cirdofbw(newmasks, ih, iw)

            # create needle diagram
            needle = create_needle(of_mag, of_ang)

            # XXX: new idea
            # create cluster image from mag and ang separately
            clustermag, magmasks = create_clusters(of_mag, 2)
            magmasks = remove_background_mask(magmasks)
            clusterang, angmasks = create_clusters(of_ang, 4)

            # multiscale processing
            multiclustermag = multiscaleoperation(of_mag, [2, 3, 4])
            multiclustermag0 = multiclustermag.copy()
            # t = 0.75 * 255
            # multiclustermag[multiclustermag < t] = 0

            # calculate face location and size
            (nfx0, nfy0, nfw, nfh) = get_face(cascade_classifier, grey0)
            # print("new:", nfx0, nfy0, nfw, nfh)
            if nfx0 > 0 and nfy0 > 0 and nfw > 0 and nfh > 0:
                lfx0.append(nfx0)
                lfy0.append(nfy0)
                lfw.append(nfw)
                lfh.append(nfh)
                fx0 = int(sum(lfx0) // len(lfx0))
                fy0 = int(sum(lfy0) // len(lfy0))
                fw = int(sum(lfw) // len(lfw))
                fh = int(sum(lfh) // len(lfh))
                # print("updated:", fx0, fy0, fw, fh)
            elif len(lfx0) > 0 and len(lfy0) > 0 and len(lfw) > 0 and len(lfh) > 0:
                fx0 = int(sum(lfx0) // len(lfx0))
                fy0 = int(sum(lfy0) // len(lfy0))
                fw = int(sum(lfw) // len(lfw))
                fh = int(sum(lfh) // len(lfh))
                # print("current:", fx0, fy0, fw, fh)
            elif prevfx0 > 0 and prevfy0 > 0 and prevfw > 0 and prevfh > 0:
                fx0 = prevfx0
                fy0 = prevfy0
                fw = prevfw
                fh = prevfh
                # print("prev:", prevfx0, prevfy0, prevfw, prevfh)
            else:
                # print(fid, "is skipped.")
                continue

            # calculate average mcm from n previous frames.
            # current frame is not included.
            avgmcm = np.zeros((ih, iw), dtype=np.uint8)
            avgmcmall = np.zeros((ih, iw), dtype=np.uint8)
            avgmcmbin = np.zeros((ih, iw), dtype=np.uint8)
            avgnum = 3
            # threshval = 0.5
            if len(prevmcm) >= avgnum:
                lastmcm = prevmcm[-avgnum:]
                for i, mcm in enumerate(lastmcm):
                    # this gives weight to previous layers.
                    avgmcmall += (mcm * ((i + 1) / avgnum)).astype(np.uint8)
                avgval = np.unique(avgmcmall)
                lenavgval = len(avgval)
                if lenavgval > 1:
                    lenavgval -= 1
                for i, val in enumerate(avgval):
                    # if i > (threshval * lenavgval):
                    avgmcmbin[avgmcmall == val] = i
                    avgmcm[avgmcmall == val] = int((i / lenavgval) * 255)

            # cv.imwrite("multiclustermag_before.jpeg", multiclustermag)
            summag = np.sum(of_mag)
            sumpxl = np.count_nonzero(multiclustermag)
            ratpxl = sumpxl / (iw * ih)
            # print(f"sum of magnitude: {summag}")
            # print(f"num of pixels: {sumpxl}")
            # print(f"ratio: {ratpxl}")
            if len(prevsummag) > 1:
                dsummag = summag - prevsummag[-1]
                psummag = (dsummag / summag)
                dsumpxl = sumpxl - prevsumpxl[-1]
                if sumpxl == 0:
                    print(f"xxx0: frame-{fid} sumpxl == 0")
                    sumpxl = dsumpxl
                if dsumpxl > 0:
                    psumpxl = (dsumpxl / sumpxl)
                else:
                    psumpxl = 0
                facepxl = 3 * fh * fw
                # print(f"dsumpxl: {dsumpxl}, facepxl: {facepxl}")
                # print(f"psummag: {psummag}, psumpxl:{psumpxl}")
                if psummag < -0.25 or psumpxl > 0.25 or dsumpxl > facepxl:
                    # print("LARGE SLOW AREA IS DETECTED.")
                    # print("searching for new multiclustermag...")
                    # print()
                    mmaskbin2 = multiclustermag.copy()
                    mval = np.unique(mmaskbin2)
                    i = 0
                    while(psumpxl > 0.25 and i < len(mval)):
                        mmaskbin2 = multiclustermag.copy()
                        mmaskbin2[mmaskbin2 < mval[i]] = 0
                        sumpxl = np.count_nonzero(mmaskbin2)
                        dsumpxl = sumpxl - prevsumpxl[-2]
                        if sumpxl == 0:
                            print(f"xxx1: frame-{fid} sumpxl == 0")
                            sumpxl = dsumpxl
                        if dsumpxl > 0:
                            psumpxl = (dsumpxl / sumpxl)
                        else:
                            psumpxl = 0
                        i += 1
                    multiclustermag = mmaskbin2.copy()
                    sumpxl = np.count_nonzero(multiclustermag)
                    ratpxl = sumpxl / (iw * ih)

            prevsummag.append(summag)
            prevsumpxl.append(sumpxl)
            prevratpxl.append(ratpxl)

            # cv.imwrite("multiclustermag_after.jpeg", multiclustermag)
            # mcm mask bin
            mmaskbin = np.zeros((ih, iw), dtype=np.uint8)
            mval = np.unique(multiclustermag)
            for i, val in enumerate(mval):
                mmaskbin[multiclustermag == val] = i
            prevmcm.append(mmaskbin)

            # tracking mask bin
            tmaskbin = np.zeros((ih, iw), dtype=np.uint8)
            tval = np.unique(tmask)
            for i, val in enumerate(tval):
                tmaskbin[tmask == val] = i

            # combine masks
            mtmaskfull = np.zeros((ih, iw), dtype=np.uint8)
            mtmaskpart = np.zeros((ih, iw), dtype=np.uint8)
            mtmaskfulladd = tmaskbin + mmaskbin
            mtmaskpartadd = tmaskbin + mmaskbin
            mtval = np.unique(mtmaskfulladd)
            lenmtval = len(mtval)
            if lenmtval > 1:
                lenmtval -= 1
            threshval = 0.5
            for i, val in enumerate(mtval):
                mtmaskfull[mtmaskfulladd == val] = int((i / lenmtval) * 255)
                if i > (threshval * lenmtval):
                    mtmaskpart[mtmaskpartadd == val] = int((i / lenmtval) * 255)
                    # mtmaskpart[mtmaskpartadd == val] = 255

            avgtmask = np.zeros((ih, iw), dtype=np.uint8)
            avgtmaskadd = tmaskbin + avgmcmbin
            avgtval = np.unique(avgtmaskadd)
            lenavgtval = len(avgtval)
            if lenavgtval > 1:
                lenavgtval -= 1
            for i, val in enumerate(avgtval):
                avgtmask[avgtmaskadd == val] = int((i / lenavgtval) * 255)

            # start combining
            # 1. output1 = tracking mask x current mcm

            tmask0mcm = np.zeros((ih, iw), dtype=np.uint8)
            tmask0bin = np.zeros((ih, iw), dtype=np.uint8)
            tval0 = np.unique(tmask0)
            print(tval0)
            for i, val in enumerate(tval0):
                tmask0bin[tmask0 == val]= i
            mcmval = np.unique(multiclustermag0)
            print(mcmval)
            mcmlayer = np.zeros((ih, iw), dtype=np.uint8)
            for i in range(len(mcmval)-1, 0, -1):
                mcmlayer[multiclustermag0 == mcmval[i]] = 1
                tmcm0 = mcmlayer * tmask0bin
                print(mcmval[i])
                if np.count_nonzero(tmcm0) > 0:
                    tmask0mcm[multiclustermag0 >= mcmval[i]] = 255
                    break
            # cv.imshow("tmask0mcm", tmask0mcm)
            # cv.imwrite("tmask0mcm.jpeg", tmask0mcm)
            tmask0mcmbin = tmask0mcm.copy()
            tmask0mcmbin[tmask0mcm == 255] = 1


            tmask1mcm = np.zeros((ih, iw), dtype=np.uint8)
            tmask1bin = np.zeros((ih, iw), dtype=np.uint8)
            tval1 = np.unique(tmask1)
            for i, val in enumerate(tval1):
                tmask1bin[tmask1 == val]= i
            mcmval = np.unique(multiclustermag0)
            mcmlayer = np.zeros((ih, iw), dtype=np.uint8)
            for i in range(len(mcmval)-1, 0, -1):
                mcmlayer[multiclustermag0 == mcmval[i]] = 1
                tmcm1 = mcmlayer * tmask1bin
                if np.count_nonzero(tmcm1) > 0:
                    tmask1mcm[multiclustermag0 >= mcmval[i]] = 255
                    break
            # cv.imshow("tmask1mcm", tmask1mcm)
            # cv.imwrite("tmask1mcm.jpeg", tmask1mcm)
            tmask1mcmbin = tmask1mcm.copy()
            tmask1mcmbin[tmask1mcm == 255] = 1

            # 2. output2 = previous mcm x current mcm
            allmcmnew = np.zeros((ih, iw), dtype=np.uint8)
            allmcm = avgmcmbin + mmaskbin + tmask0mcmbin + tmask1mcmbin
            maxmcm = np.max(allmcm)
            allmcmstd = allmcm / maxmcm
            allmcmsave = allmcmstd.copy()
            # allmcmstd[allmcmstd <= 0.5] = 0
            # allmcmval = np.unique(allmcm)
            # print(allmcmval)
            # lenallmcmval = len(allmcmval)
            # if lenallmcmval > 1:
            #     lenallmcmval -= 1
            # for i, val in enumerate(allmcmval):
            #     allmcmnew[allmcm == val] = int((i / lenallmcmval) * 255)
            allmcmnew = (allmcmstd * 255).astype(np.uint8)
            cv.imshow("allmcmnew", allmcmnew)
            # cv.imwrite("allmcmnew.jpeg", allmcmnew)
            # cv.imshow("allmcmstd", allmcmstd)
            allmcmfile = datadir + dataname + f"/1/allmcm50p{fid:06d}-1.png"
            #  cv.imwrite(allmcmfile, allmcmnew)



            # 3. final mcm: output1 x output2
            # finish combining
            # save to the collection of final mcm

            # XXX
            # # # refine masks in newmagmasks and newangmasks
            # # newmagmasks = refine_masks(magmasks, fxc, fyc)
            # # newangmasks = refine_masks(angmasks, fxc, fyc)
            # #
            # # # get skin mask and decompose it
            # # (skinmask0, skinmasks0, facemask0,
            # #  fbhsv, fthsv, fbycrcb, ftycrcb,
            # #  fxc, fyc) = get_skinmask(cascade_classifier, frame0,
            # #                           fbhsv, fthsv, fbycrcb, ftycrcb, fxc, fyc)
            # #
            # # (skinmask1, skinmasks1, facemask1,
            # #  fbhsv, fthsv, fbycrcb, ftycrcb,
            # #  fxc, fyc) = get_skinmask(cascade_classifier, frame1,
            # #                           fbhsv, fthsv, fbycrcb, ftycrcb, fxc, fyc)
            # #
            # # # get facemask
            # # (facemasks, facemask,
            # #  facemaskbin, faceprobbin) = get_facemask(facemask0, facemask1,
            # #                                           facemask, facemaskbin)
            # #
            # # # find the hand candidate location from mag, ang, skin masks
            # # allmasks, probmask = combine_masks(newmagmasks, newangmasks,
            # #                                    skinmasks0, skinmasks1,
            # #                                    facemasks, of_mag,
            # #                                    multiclustermag)
            # XXX

            # probmask = get_point_cloud3(of_mag, probmask, 25)
            # points = get_point_cloud3(of_mag, of_ang, probmask, 25)
            # fpoints.append([fid, points])
            # for (y, x, magyx, angyx) in points:
            #     cv.circle(probmask, (x, y), 1, 0, -1)
            #     cv.circle(frame0, (x, y), 1, (255, 255, 255), -1)

            # points = get_point_cloud4(probmask, 2, fx0, fy0, fw, fh)
            # points = get_point_cloud4(multiclustermag, 2, fx0, fy0, fw, fh)
            # points = get_point_cloud4(allmcm, 10, fx0, fy0, fw, fh)
            # points = get_point_cloud5(probmask, of_mag, 2, fx0, fy0, fw, fh)
            # fpoints.append([fid, points])
            # print(fpoints)
            # exit()


            # # XXX: testing new idea for getting the hand
            #
            # newskinmask = refine_skinmask(skinmask0, skinmask1, faceprobbin)
            # newmultimag = refine_multiclustermag(multiclustermag)
            #
            # # select the right amount of newmultimag that covers the skin
            # # get skinmask contour
            # sctr, shrchy = cv.findContours(newskinmask, cv.RETR_EXTERNAL,
            #                                cv.CHAIN_APPROX_SIMPLE)
            #
            # # get multimag contour and save all into one tupple
            # tu = np.unique(newmultimag)
            # tmaskori = newmultimag.copy()
            # allmctr = ()
            # for tid, t in enumerate(tu):
            #     tmask = tmaskori.copy()
            #     tmask[tmask <= t] = 0
            #     tmask[tmask > t] = 255
            #     mctr, mhrchy = cv.findContours(tmask, cv.RETR_EXTERNAL,
            #                                    cv.CHAIN_APPROX_SIMPLE)
            #     allmctr = allmctr + mctr
            #
            # # get prevmultimag contour
            # pctr = ()
            # if len(prevmultimag) > 0:
            #     pctr, phrchy = cv.findContours(prevmultimag[-1],
            #                                    cv.RETR_EXTERNAL,
            #                                    cv.CHAIN_APPROX_SIMPLE)
            #
            # # start analysing
            # newmultimag = refine_newmultimag(newmultimag, sctr, allmctr, pctr)
            #
            # # here are start the idea of week153
            # # 1. detect rotated rectangle of the contour.
            # newmultimag2 = refine_multiclustermag(multiclustermag)
            # newmultimag2top = newmultimag2.copy()
            # newmultimag2top[newmultimag2top > 0] = 255
            # newmultimag2flt = np.zeros_like(multiclustermag)
            # ctr, hrchy = cv.findContours(newmultimag2top, cv.RETR_EXTERNAL,
            #                              cv.CHAIN_APPROX_SIMPLE)
            # for c in range(len(ctr)):
            #     ctrmask = np.zeros_like(multiclustermag)
            #     cv.drawContours(ctrmask, ctr, c, 1, -1)
            #     ctrmulti = multiclustermag * ctrmask
            #     tu = np.unique(ctrmulti)
            #     t = tu[-1]
            #     newmultimag2flt[ctrmulti >= t] = 255
            #
            # ctr, hrchy = cv.findContours(newmultimag2flt, cv.RETR_EXTERNAL,
            #                              cv.CHAIN_APPROX_SIMPLE)
            #
            # cctr = []
            # for c in range(len(ctr)):
            #     # 2. draw the rotated rectangle.
            #     rect = cv.minAreaRect(ctr[c])
            #     box = cv.boxPoints(rect)
            #     box = np.int0(box)
            #     cv.drawContours(newmultimag2flt, [box], 0, 255, 1)
            #     # 3. detect the average angle
            #     cmap = np.zeros_like(newmultimag2flt)
            #     cv.drawContours(cmap, ctr, c, 1, -1)
            #     angcmap = of_ang * cmap
            #     magcmap = of_mag * cmap
            #     nzcmap = cv.countNonZero(cmap)
            #     angmean = np.sum(angcmap) / nzcmap
            #     magmean = np.sum(magcmap) / nzcmap
            #     x = magmean * math.cos(math.radians(angmean))
            #     y = magmean * math.sin(math.radians(angmean))
            #     if not math.isinf(x) and not math.isnan(x):
            #         x = int(x)
            #     else:
            #         x = 0
            #     if not math.isinf(y) and not math.isnan(y):
            #         y = int(y)
            #     else:
            #         y = 0
            #     mm = cv.moments(ctr[c])
            #     m01 = mm['m01']
            #     m10 = mm['m10']
            #     m00 = mm['m00'] + 0.00001
            #     mcx = int(m10 / m00)
            #     mcy = int(m01 / m00)
            #     # cv.circle(newmultimag2flt, (mcx, mcy), 2, 127, -1)
            #     # cv.line(newmultimag2flt, (mcx, mcy), (mcx+(x*2), mcy+(y*2)), 127, 2)
            #     # cv.rectangle(newmultimag2flt, (0, 0), (iw-1, ih-1), 255, 1)
            #     cctr.append([mcx, mcy, x, y, magmean, angmean])
            #
            # # get the highest cluster from multi cluster magnitude
            # highmultimagmask = multiclustermag.copy()
            # highmultimagmask[highmultimagmask < 255] = 0
            # frame0gray = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
            # frame1gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            # frame0gray[highmultimagmask > 0] = 255
            # frame1gray[highmultimagmask > 0] = 255
            #
            # # save for next process
            # prevskinmask.append(newskinmask)
            # prevmultimag.append(newmultimag)
            # prevmultimagflt.append(newmultimag2flt)
            # prevmultimagdata.append([ctr, cctr])
            # # ---------------------
            #
            # if len(prevmultimagdata) > 2:
            #     # get t-0
            #     t0 = prevmultimagdata[-2]
            #
            #     # get t-1
            #     tm1 = prevmultimagdata[-3]
            #
            #     # get t+1
            #     tp1 = prevmultimagdata[-1]
            #
            #     # print("Frame:", fid)
            #     outimg = extractContour(t0, tp1, tm1, iw, ih)
            # else:
            #     outimg = np.zeros((ih, iw), dtype=np.uint8)
            # # cv.imshow("outimg", outimg)
            # prevbins.append(outimg)
            #
            # n = 5
            # if len(prevbins) >= n:
            #     start = len(prevbins)-n
            #     stop = len(prevbins)
            #     allbin = create_allbin(prevbins, start, stop)
            #     # cv.imshow("allbin", allbin)
            # # ---------------
            # fnori = datafiles[fid-1]
            # fnorisplit = fnori.split("/")
            # fnlast = fnorisplit[-1]
            # fnlastsplit = fnlast.split("-")
            # fnwrite = ""
            # for fn in fnorisplit[:-1]:
            #     fnwrite += fn + "/"
            # # fnwrite += fnlastsplit[0] + "-1.png"
            # # print(fnwrite)
            # # cv.imwrite(fnwrite, outimg)
            #
            #
            # skinmaskcolour0 = analyseskinmask(skinmask0, fxc, fyc,
            #                                   of_mag, of_ang)
            # skinmaskcolour1 = analyseskinmask(skinmask1, fxc, fyc,
            #                                   of_mag, of_ang)
            #
            # # test max points:
            # # points = get_point_cloud(of_mag, newmultimag, 50, 5)
            # # fpoints.append([fid, points])

            # frame0gray = get_point_cloud2(of_mag, frame0gray, 50, 5)


            # # put text information
            # cv.putText(frame0, str(fid-1), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            # cv.putText(frame1, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            # cv.putText(of_bgr, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # cv.putText(cirdofbw, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(needle, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 0)
            # cv.putText(clustermag, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(clusterang, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(multiclustermag, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # # cv.putText(skinmaskcolour0, str(fid-1), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # cv.putText(skinmaskcolour1, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            # cv.putText(facemask0, str(fid-1), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(facemask1, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(facemask, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # cv.putText(probmask, str(fid), (10, 20),
            #            cv.FONT_HERSHEY_PLAIN, 1, 255)
            # if tpoint[1] != []:
            #     for t in tpoint[1]:
            #         if t[1] != []:
            #             cv.circle(frame0, t[1], 2, (0, 0, 255), -1)
            #             cv.circle(multiclustermag, t[1], 1, 0, 1)
            #             cv.circle(multiclustermag, t[1], 2, 255, 1)
            #             cv.circle(multiclustermag, t[1], 3, 0, 1)
            #             cv.circle(multiclustermag, t[1], 31, 0, 1)
            #             cv.circle(multiclustermag, t[1], 32, 255, 1)
            #             cv.circle(multiclustermag, t[1], 33, 0, 1)

            # display
            cv.imshow("frame0", frame0)
            # cv.imshow("frame0", frame0gray)
            cv.imshow("frame1", frame1)
            # cv.imshow("frame1", frame1gray)
            # cv.imshow("needle", needle)
            # cv.imshow("of_bgr", of_bgr)
            # cv.imshow("clustermag", clustermag)
            # cv.imshow("clusterang", clusterang)
            cv.imshow("original multiclustermag", multiclustermag0)
            cv.imshow("multiclustermag", multiclustermag)
            cv.imshow("avgmcm", avgmcm)
            # cv.imshow("avgmcmbin", avgmcmbin)
            # cv.imshow("avgtmask", avgtmask)
            # cv.imshow("newmultimag", newmultimag)
            # cv.imshow("newmultimag2flt", newmultimag2flt)
            # cv.imshow("skinmaskcolour0", skinmaskcolour0)
            # cv.imshow("skinmaskcolour1", skinmaskcolour1)
            # cv.imshow("newskinmask", newskinmask)
            # cv.imshow("facemask0", facemask0)
            # cv.imshow("facemask1", facemask1)
            # cv.imshow("facemask", facemask)
            # cv.imshow("probmask", probmask)
            # for fid, f in enumerate(facemasks):
            #     cv.imshow("facemasks-"+str(fid), f)
            # cv.imshow("facemask", facemask)
            # cv.imshow("facemaskbin", facemaskbin*255)
            # cv.imshow("faceprobbin", faceprobbin*255)
            # if len(prevmultimagflt) > 2:
            #     display_single_window(prevmultimagflt, "lst multimag", fid-1, 2)
            cv.imshow("tmask", tmask)
            cv.imshow("tmask0", tmask0)
            cv.imshow("tmask1", tmask1)
            # cv.imshow("mtmaskfull", mtmaskfull)
            # cv.imshow("mtmaskpart", mtmaskpart)

            # winx0 = 1920
            # winy0 = 0
            # winw = 370
            # winh = 385
            # cv.moveWindow("frame0", winx0 + 0 * winw, winy0 + 0 * winh)
            # cv.moveWindow("frame1", winx0 + 0 * winw, winy0 + 1 * winh)

            k = cv.waitKey(0) & 0xFF
            if k == ord('q'):
                cv.destroyAllWindows()
                exit()
            elif k == ord('s'):
                # cv.imwrite("frame0.jpeg", frame0)
                # cv.imwrite("frame1.jpeg", frame1)
                # cv.imwrite("needle.jpeg", needle)
                # cv.imwrite("of_bgr.jpeg", of_bgr)
                # cv.imwrite("multiclustermagori.jpeg", multiclustermag0)
                # cv.imwrite("multiclustermag.jpeg", multiclustermag)
                # cv.imwrite("avgmcm.jpeg", avgmcm)
                # cv.imwrite("tmask.jpeg", tmask)
                # cv.imwrite("tmask0.jpeg", tmask0)
                # cv.imwrite("tmask1.jpeg", tmask1)
                llist = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                for ll in llist:
                    allmcmsave = allmcmstd.copy()
                    ts = (100-ll) / 100
                    print(ll)
                    allmcmsave[allmcmstd <= ts] = 0
                    allmcmdis = (allmcmsave * 255).astype(np.uint8)
                    cv.imwrite(f"../allmcm{ll}p{fid:06d}-1.png", allmcmdis)

        # picklefile = datadir + dataname + "/1/anpoints10_1"
        # with open(picklefile, "wb") as pf:
        #     pickle.dump(fpoints, pf)
        # pf.close()
        # print(picklefile, "saved.")
        # prevfx0, prevfy0, prevfw, prevfh = fx0, fy0, fw, fh

        # with open(picklefile, "rb") as pf:
        #     loadpickle = pickle.load(pf)
        #     print("loadpickle:", loadpickle)
        # pf.close()
        # print(picklefile, "loaded.")
