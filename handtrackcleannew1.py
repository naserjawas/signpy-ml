"""
This is a copy of handtrackcleannew.py version

In this file, the hand tracking algorithm is rewritten to clean the code for
paper revision.

This file is inteded to use the RWTH-BOSTON-50 dataset.

author: naserjawas
date: 20 March 2022
"""
import pickle
import argparse
import glob
import math
import statistics
import csv
import cv2 as cv
import numpy as np
import xml.etree.cElementTree as et

from statistics import NormalDist
from sklearn.cluster import KMeans, OPTICS
from sklearn.utils import shuffle
from sklearn.svm import OneClassSVM
from PIL import Image


def display_multi_window(list_of_images, title, current_index, window_length):
    """
    Display images in multiple window.

    parameters:
        list_of_images: an array of images to display.
        title: the title of the window.
        current_index: the latest index of the image array to display.
        window_length: how many images to display.
    """
    display_index = 0
    start_index = current_index - window_length
    for i in range(start_index, current_index + 1):
        cv.imshow("signpy: " + title + str(display_index), list_of_images[i])
        display_index += 1

def display_single_window(list_of_images, title, current_index, window_length):
    """
    Display images in single window.

    parameters:
        list_of_images: an array of images to display.
        title: the title of the window.
        current_index: the latest index of the image array to display.
        window_length: how many images to display.
    """
    start_index = current_index - window_length
    for i in range(start_index, current_index + 1):
        if i == start_index:
            images = list_of_images[i].copy()
        else:
            images = np.concatenate((images, list_of_images[i]), axis=1)
    cv.imshow(title, images)
    cv.moveWindow(title, 0, 0)

    return images

def get_face(cascade_classifier, image):
    """
    Get the face location in an image using cascade classifier.

    parameters:
        cascade_classifier:
        image:

    returns:
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

def calc_optical_flow(prev_img, next_img, mode):
    """
    Calculate the optical flow of two images.

    parameters:
        prev_img: the first image.
        next_img: the second image.
        mode:
            0: low-scale.
            1: high-scale.

    returns:
        mag: magnitude image.
        ang: angular image.
    """
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
                   winSize=15,           # 15,
                   numIters=3,           # 3,
                   polyN=7,              # 5,
                   polySigma=1.5,        # 1.2,
                   flags=0)
    if mode == 0:
        of = cv.FarnebackOpticalFlow_create(**params0)
    else:
        of = cv.FarnebackOpticalFlow_create(**params1)
    flow = of.calc(prev_img, next_img, None)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    return mag, ang

def calc_euclidean(x0, y0, x1, y1):
    """
    Calculate euclidean distance from two points.

    parameters:
        x0
        y0
        x1
        y1

    returns:
        r: distance.
    """
    dx = x1 - x0
    dy = y1 - y0
    r2 = (dx**2) + (dy**2)
    r = math.sqrt(r2)

    return r

def compile_feature(cluster_data, flow0_data, flow1_data, span=3):
    """
    Compile feature for tracking.

    parameters:
        cluster_data = mask_id,
                       X0_0 (x, y),
                       X1_0 (colours),
                       X2_0 (mag),
                       X3_0 (ang)
        flow0_data (mag, ang)
        flow1_data (mag, ang)

    returns:
        features = (centre, sum_mag, mean_ang, std_ang,
                    mean_dist, std_dist, num_of_points, centre mag)
    """
    features = []
    flow0_mag, flow0_ang = flow0_data
    flow1_mag, flow1_ang = flow1_data

    for c in cluster_data:
        if len(c[1]) > 0:
            centre = np.mean(c[1], axis=0, dtype=np.uint32).tolist()
            sum_mag = np.sum(c[3], dtype=np.float64)
            dists = [calc_euclidean(centre[0], centre[1], X0_0[0], X0_0[1])
                     for X0_0 in c[1]]
            dists = np.array(dists)
            mean_dist = np.mean(dists)
            std_dist = np.std(dists)
            num_of_points = len(c[1])
            x0 = centre[0] - span
            y0 = centre[1] - span
            x1 = centre[0] + span + 1
            y1 = centre[1] + span + 1
            centre_ang = flow1_ang[y0:y1, x0:x1]
            mean_ang = np.mean(centre_ang)
            std_ang = np.std(centre_ang)
            centre_mag = flow0_mag[centre[1], centre[0]]
            features.append((centre, sum_mag, mean_ang, std_ang,
                             mean_dist, std_dist, num_of_points,
                             centre_mag))
    return features

def find_last_record(records, counter):
    """
    Find 1 previous record.
    Note: this method is useful when in the records all the data are saved,
    including the empty ones.

    parameters:
        records: a list of records
        counter: number of record from the last with 0 means the last.

    returns:
        last_id, last
    """
    last_id = 0
    last = []
    for i in range(len(records)-1, -1, -1):
        last_id = i
        last = records[last_id]
        if len(last) > 0:
            if counter > 0:
                counter -= 1
                last_id = 0
                last = []
            else:
                break

    return last_id, last

def find_last_records(records, counter):
    """
    Find n previous records.

    paramters:
        records: a list of records
        counter: how many record to return

    returns:
        last n records in a list
        centre points of the last n records
    """
    lasts = []
    centres = []
    centre = []
    return_counter = 0

    for i in range(len(records)-1, -1, -1):
        last = records[i]
        if len(last) > 0 and return_counter < counter:
            lasts.append(last)
            centres.append(last[0][0])
            return_counter += 1
            if return_counter >= counter:
                break

    if len(centres) > 0:
        centre = np.mean(centres, axis=0, dtype=np.uint32).tolist()

    return lasts, centre

def get_candidate(candidates, id_candidate, pop):
    """
    Get a candidate using its id and remove it from candidate list.

    parameter:
        candidates: a list of candidates.
        id_candidate: the id of the candidate to get.
        pop: boolean, true for pop or false for not pop.

    returns:
        candidate: the single candidate.
        candidates: the list of remain candidates.
    """
    if id_candidate < len(candidates):
        candidate = candidates[id_candidate]
        if pop:
            candidates.pop(id_candidate)
    else:
        candidate = None

    return candidate, candidates

def get_furthest_candidate(candidates, ref, limit=0):
    """
    Get the furthest candidate from reference value.

    parameter:
        candidates: a list of candidates
        ref: reference candidate

    returns:
        candidate: the single candidate.
        candidates: the list of remain candidates.
    """
    candidate = ()
    if len(candidates) > 0:
        dists = []
        refx0, refy0 = ref[0]
        for candidate_id, candidate in enumerate(candidates):
            cx0, cy0 = candidate[0]
            dist = calc_euclidean(refx0, refy0, cx0, cy0)
            dists.append((candidate_id, dist))
        dists = sorted(dists, key=lambda v:v[1], reverse=True)
        candidate_id, dist = dists[0]
        # set limit and return the furthest away but bellow a certain limit.
        if limit > 0 and dist > limit:
            candidate = candidates[candidate_id]
            candidates.pop(candidate_id)

    return candidate, candidates

def get_nearest_candidate(candidates, ref, limit=0):
    """
    Get the nearest candidate from reference value.

    parameter:
        candidates: a list of candidates
        ref: reference candidate

    returns:
        candidate: the single candidate.
        candidates: the list of remain candidates.
    """
    candidate = ()
    if len(candidates) > 0:
        dists = []
        refx0, refy0 = ref[0]
        for candidate_id, candidate in enumerate(candidates):
            cx0, cy0 = candidate[0]
            dist = calc_euclidean(refx0, refy0, cx0, cy0)
            dists.append((candidate_id, dist))
        dists = sorted(dists, key=lambda v:v[1], reverse=False)
        candidate_id, dist = dists[0]
        # set limit and return the nearest and bellow a certain limit.
        if limit > 0 and dist < limit:
            candidate = candidates[candidate_id]
            candidates.pop(candidate_id)

    return candidate, candidates

def check_assignment(records1, current1, records2, current2,
                     candidates, face_xc, face_yc, face_d):
    """
    Check the first tracking assignment for:
    1. if it is first data: no overlap with face.
    2. if it is not first data: must overlap with previous to prevent jumping

    parameter:
        records1: the records to be checked
        current1: the current tracking assignment to be checked
        records2: the records for reference.
        current2: the current tracking assignment for reference.
        candidates: list of the candidates
        face_xc: x coordinate of face centre
        face_yc: y coordinate of face centre

    returns:
        current1:
        current2:
        candidates:
    """
    if len(records1) == 0:
        # first data: must not overlap with face
        if len(current1) > 0:
            x1, y1 = current1[0][0]
            dist1 = calc_euclidean(x1, y1, face_xc, face_yc)
            if dist1 < face_d:
                current1 = []
    else:
        # non first: must overlap with previous to prevent jumping
        # 1. find previous from records1
        r1_id, r1 = find_last_record(records1, 0)
        if len(current1) > 0:
            xr1, yr1 = r1[0][0]
            xc1, yc1 = current1[0][0]
            dist1 = calc_euclidean(xr1, yr1, xc1, yc1)
            # 2. if dist previous from current is more than face_d
            if dist1 > (2 * face_d):
                if len(records2) > 0:
                    r2_id, r2 = find_last_record(records2, 0)
                    xr2, yr2 = r2[0][0]
                    dist2 = calc_euclidean(xr2, yr2, xc1, yc1)
                    distf = calc_euclidean(face_xc, face_yc, xc1, yc1)
                    # if dist2 < face_d and dist2 < distf:
                    if dist2 < face_d:
                        if len(current2) > 0:
                            candidates.append(current2[0])
                        current2 = current1.copy()
                    else:
                        candidates.append(current1[0])
                    current1 = []
                    current1.append(r1[0])

    return current1, current2, candidates

def compare_candidates(candidates, current, records, limit):
    """
    compare the initial tracking assignment with the available candidates.

    parameter:
        candidates: list of candidates
        current: current assignment
        records: list of previous records
        limit: limit distance (usually set to face_d)

    returns:
        candidates: list of candidates
        current: new current assignment
    """
    prev_id, prev = find_last_record(records, 0)
    xcr, ycr = current[0][0]
    xpr, ypr = prev[0][0]
    distcr = calc_euclidean(xcr, ycr, xpr, ypr)

    dists = []
    for c_id, c in enumerate(candidates):
        xc, yc = c[0]
        dist = calc_euclidean(xcr, ycr, xc, yc)
        if dist > 0 and dist < limit:
            dists.append([c_id, dist])
    if len(dists) > 0:
        dists = sorted(dists, key=lambda v:v[1], reverse=False)
        c_id = dists[0][0]
        minc = candidates[c_id]
        if minc[1] > current[0][1] or minc[7] > current[0][7]:
            candidates.pop(c_id)
            candidates.append(current[0])
            current[0] = minc

    return candidates, current

def record_zscore(mag_array, summag_records, zscore_records, span=3):
    """
    calculate and record the zscore

    parameters:
        mag_array: the magnitude array from optical flow
        summag_records: record of sum of magnitude from previous frame.
        zscore_records: record of zscore from previous frame.

    return:
        summag_records: updated summag_records
        zscore_records: updated
    """
    summag = np.sum(mag_array[np.isfinite(mag_array)])
    summag_records.append(summag)
    if len(summag_records) > span:
        nd = NormalDist.from_samples(summag_records[-span:])
        if nd._sigma != 0:
            zscore = nd.zscore(summag)
        else:
            zscore = 0
        zscore_records.append(round(zscore))
    else:
        zscore_records.append(0)

    return summag_records, zscore_records

def zscore_projection(zscore_records, tracking_records):
    """
    define and calculate the tracking projection using zscore records.

    parameters:
        zscore_records: the previous frames zscore records
        tracking_records: the previous frames tracking records

    return:
        projection: the projection compiled feature.
    """
    projection = []
    r1_id, r1 = find_last_record(tracking_records, 0)
    r2_id, r2 = find_last_record(tracking_records, 1)

    zscore3, zscore2, zscore1 = zscore_records[-3:]
    if zscore1 < zscore2 and zscore1 < zscore3:
        xr1, yr1 = r1[0][0]
        xr2, yr2 = r2[0][0]
        dx = xr1 - xr2
        dy = yr1 - yr2
        xr0 = xr1 + dx
        yr0 = yr1 + dy
        feature = ([xr0, yr0], 0, 0, 0, 0, 0, 0, 0)
        projection.append(feature)
    elif zscore1 == zscore2 and zscore1 == zscore3 and zscore1 < 0:
        projection = r1.copy()

    return projection

def averaging_projection(records, current):
    """
    projection with averaging the tracking information between two frames.

    parameters:
        records: a list of tracking records.
        current: the current frame tracking data.

    return:
        projection
    """
    projection = []
    if (len(records) > 2 and current != [] and
        records[-1] != [] and records[-2] != []):
        r1_id, r1 = find_last_record(records, 0)
        r2_id, r2 = find_last_record(records, 1)
        x0, y0 = current[0][0]
        x1, y1 = r1[0][0]
        x2, y2 = r2[0][0]
        dx = x0 - x2
        dy = y0 - y2
        mx = dx / 2
        my = dy / 2
        newx1 = int(x2 + mx)
        newy1 = int(y2 + my)
        projection.append((newx1, newy1))

    return projection

def calc_rect(xc, yc, span):
    """
    calculate rectangle (x0, y0) and (x1, y1) from centre (xc, yc)

    parameters:
        xc,yc: centre coordinate
        span: the length from centre to side of rectangle.

    return:
        x0, y0: first coordinate
        x1, y1: second coordinate
    """
    x0 = xc - span
    y0 = yc - span
    x1 = xc + span
    y1 = yc + span

    return x0, y0, x1, y1

def check_stationary(current, records, counter):
    """
    gives the status of stationary records (boolean)

    parameters:
        current: current position
        records: hand position records
        counter: number of occurence

    return:
        status: boolean
    """
    return_counter = 0
    checked = False
    for i in range(len(records)-1, -1, -1):
        last = records[i]
        if len(last) > 0 and return_counter < counter:
            if last[0][0] == current[0][0]:
                return_counter += 1
                if return_counter >= counter:
                    checked = True
                    break
            else:
                checked = False
                break

    return checked

def contrast_stretching(img, alpha, beta):
    newimg = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                newimg[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)

    return newimg

def calc_optical_flow_new(prev_img, next_img, mode, prevof):
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
                   winSize=15,           # 15,
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
                    else:
                        allmasksbin[mask > 0] += 1

    allmasksbin = (allmasksbin / np.max(allmasksbin)) * 255
    allmasksbin = allmasksbin.astype(np.uint8)

    return allmasksbin


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="This program tracks hand(s) in sign language video.")
    parser.add_argument("-p", "--path", help="path to the video", dest="path", required=True)
    args = parser.parse_args()
    print(args.path)

    # Load dataset
    dataset_path = args.path

    dir_name = args.path
    video_name = dir_name.split("/")[-2]
    save_name = dir_name.split("/")[-1]
    # save_name = save_name.replace("_", "")
    # print("video_name:", save_name)
    file_path = dir_name + "*-0.png"
    filenames = sorted(glob.glob(file_path))
    if len(filenames) == 0:
        print("file " + save_name + " is not found")
        exit()
    images = [cv.imread(filename, cv.IMREAD_COLOR)
              for filename in filenames]
    print("Load", len(images), "images... OK")

    # settings
    ih, iw = 0, 0
    win_length = 5
    win_main = math.floor(win_length / 2)
    range_w = 2
    range_h = 2
    # range_pixel = 10
    sample_indexes = []
    # image scaling
    scale = 100

    # load ground truth of initial frame
    groundtruth_file = "./rwth-boston-50/handmarks/rwth-boston-50.hand-groundtruth-annotation-first-frames.xml"
    xml_tree = et.parse(groundtruth_file)
    xml_root = xml_tree.getroot()
    file_paths = file_path.split("/")
    groundtruth = xml_root.findall("./video/[@name='" +file_paths[-2]+"/"+file_paths[-1] +"']")
    init_position = []
    for gt in groundtruth[0][0]:
       gt_n = int(gt.get("n"))
       # NOTE: ground truth for all width as Lim 2017
       gt_x = int(int(gt.get("x")) * scale / 100)
       gt_y = int(int(gt.get("y")) * scale / 100)
       # NOTE: ground truth for cropped area as dataset suggestion
       # gt_x = int(int(gt.get("x")) * scale / 100) - 70
       # gt_y = int(int(gt.get("y")) * scale / 100) - 10
       init_position.append((gt_n, gt_x, gt_y))

    # load ground truth
    # groundtruth_file = dir_name + "gt.csv"
    # gt_all = []
    # with open(groundtruth_file, newline="") as csvfile:
    #    reader = csv.reader(csvfile)
    #    for row in reader:
    #        row = list(map(int, row))
    #        # # NOTE: ground truth for all width as Lim 2017
    #        # row[1] = int(row[1] * scale / 100)
    #        # row[2] = int(row[2] * scale / 100)
    #        # row[3] = int(row[3] * scale / 100)
    #        # row[4] = int(row[4] * scale / 100)
    #        # # NOTE: ground truth for cropped area as dataset suggestion
    #        # row[1] = int(row[1] * scale / 100) - 70
    #        # row[2] = int(row[2] * scale / 100) - 10
    #        # row[3] = int(row[3] * scale / 100) - 70
    #        # row[4] = int(row[4] * scale / 100) - 10
    #        gt_all.append(row)
    # print("Load", len(gt_all), "groundtruth data... OK")

    # face variables
    face_detected = False
    cascade_file = "./data/haarcascade_frontalface_default.xml"
    cascade_classifier = cv.CascadeClassifier()
    cascade_classifier.load(cv.samples.findFile(cascade_file))
    face_x0 = 0
    face_y0 = 0
    face_x1 = 0
    face_y1 = 0
    face_xc = 0
    face_yc = 0
    face_w = 0
    face_h = 0
    face_d = 0

    # skin variables
    upper = 0
    lower = 0
    upper0 = 0
    lower0 = 0
    upper1 = 0
    lower1 = 0

    # Body variables
    body_x0 = 0
    body_y0 = 0
    body_x1 = 0
    body_y1 = 0

    # One Class SVM
    c0 = OneClassSVM(nu=0.1, coef0=1)

    # record variables
    drawer_images = []
    prev_colour_images = []
    prev_grey_images = []
    flow0_bgrs = []
    flow1_bgrs = []             # previously flow1_images
    flow0_bws = []              # previously flow0_images
    flow1_bws = []
    flow1_cirdofs = []          # previously flow1_quans
    flow1_cirdofbws = []        # previously flow1_foregrounds
    flow1_masks_records = []
    flow1_needles = []
    flow1_mags = []
    feature_records = []
    primary_records = []
    secondary_records = []
    a_summag_records = []
    p_summag_records = []
    s_summag_records = []
    a_zscore_records = []
    p_zscore_records = []
    s_zscore_records = []
    bins = []
    skin_masks = []
    # XXX:MCM
    mcms = []
    prevof0 = []
    prevof1 = []
    prevof2 = []
    tpoints = []

    #final result records:
    results_records = []

    #hand candidate vs ground truth
    hcgt = []

    frame_id = -1
    save_index = 0
    save_window_index = 0
    for image in images:
        frame_id = frame_id + 1
        ih, iw = image.shape[:2]

        # NOTE: RWTH-BOSTON-50 specific configuration as in Lim 2017:
        image = image[0:242, 0:]
        ih, iw = image.shape[:2]
        # NOTE: RWTH-BOSTON-50 specific configuration as in the dataset readme:
        # image = image[10:175, 70:265]
        # ih, iw = image.shape[:2]

        # image preparation
        flow0_mag = np.zeros((ih, iw), dtype=np.float64)
        flow0_ang = np.zeros((ih, iw), dtype=np.float64)
        flow0_hsv = np.zeros((ih, iw, 3), dtype=np.uint8)
        flow0_bgr = np.zeros((ih, iw, 3), dtype=np.uint8)
        flow0_bw = np.zeros((ih, iw), dtype=np.uint8)

        flow1_mag = np.zeros((ih, iw), dtype=np.float64)
        flow1_ang = np.zeros((ih, iw), dtype=np.float64)
        flow1_hsv = np.zeros((ih, iw, 3), dtype=np.uint8)
        flow1_bgr = np.zeros((ih, iw, 3), dtype=np.uint8)
        flow1_bw = np.zeros((ih, iw), dtype=np.uint8)

        flow1_cirdof = np.zeros((ih, iw, 3), dtype=np.uint8) # previously flow1_quan
        flow1_cirdofbw = np.zeros((ih, iw), dtype=np.uint8)  # previously flow1_th
        flow1_needle = np.ones((ih, iw), dtype=np.uint8) * 255


        bin0 = np.zeros((ih, iw), dtype=np.uint8)
        bin1 = np.zeros((ih, iw), dtype=np.uint8)
        bin01 = np.zeros((ih, iw), dtype=np.uint8)

        skin_mask = np.zeros((ih, iw), dtype=np.uint8)
        skin_mask0 = np.zeros((ih, iw), dtype=np.uint8)
        skin_mask1 = np.zeros((ih, iw), dtype=np.uint8)

        body_mask = np.zeros((ih, iw), dtype=np.uint8)

        # XXX:MCM
        multiclustermag = np.zeros((ih, iw), dtype=np.uint8)

        flow1_masks = []
        cluster_data = []
        feature_current = []
        feature_previous = []
        current_primary = []
        current_secondary = []
        projection_primary = []
        projection_secondary = []
        averaging_primary = []
        averaging_secondary = []

        # print("\n-------------------------------------------------------------------------------")
        print("Frame Number:", save_index)

        # Pre-Processing
        drawer_image = image.copy()
        colour_image = image.copy()
        grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur_colour_image = cv.medianBlur(colour_image, 7)
        # blur_colour_image = contrast_stretching(blur_colour_image, 1, 1)
        blur_grey_image = cv.medianBlur(grey_image, 7)

        cv.putText(drawer_image, str(save_index), (10, 20),
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # Processing
        # 1. Processing: get the signer's face
        #                initial location of the signers, skin colour.
        if not face_detected:
            (gf_x0, gf_y0, gf_w, gf_h) = get_face(cascade_classifier, image)
            if gf_x0 > 0 and gf_y0 > 0 and gf_w > 0 and gf_h > 0:
                face_detected = True
                face_w = gf_w
                face_h = gf_h
                face_x0 = gf_x0
                face_y0 = gf_y0
                face_x1 = gf_x0 + gf_w
                face_y1 = gf_y0 + gf_h
                face_xc = gf_x0 + (gf_w // 2)
                face_yc = gf_y0 + (gf_h // 2)
                face_d = math.sqrt(gf_w**2 + gf_h**2) / 2
                # range_w = face_w // range_pixel
                # range_h = face_h // range_pixel
                body_x0 = gf_x0 - gf_w
                body_x1 = body_x0 + (3 * gf_w)
                body_y0 = gf_y0 - int(0.5 * gf_h)
                body_y1 = body_y0 + int(4.5 * gf_h)

                # create face colour sample for One Class SVM
                sx0, sy0, sx1, sy1 = calc_rect(face_xc, face_yc, (face_w // 4))
                # colour_sample = blur_colour_image[sy0:sy1, sx0:sx1]

                # mean, stddev = cv.meanStdDev(colour_sample)
                # mean = mean.astype(int).reshape(1, 3)
                # stddev = stddev.astype(int).reshape(1, 3)
                # lower = mean - stddev * 3
                # upper = mean + stddev * 3
                # print(lower, upper)

                # colour_sample = colour_sample.reshape(-1, 3)
                # c0.fit(colour_sample)
                # print("Initialisation Face colour classifier... OK")
                grey_sample = blur_grey_image[sy0:sy1, sx0:sx1]

                mean, stddev = cv.meanStdDev(grey_sample)
                mean = mean.astype(int).reshape(1, 1)
                stddev = stddev.astype(int).reshape(1, 1)
                lower = mean - stddev * 3
                upper = mean + stddev * 3
                # print(lower, upper)

                grey_sample = grey_sample.reshape(-1, 1)
                c0.fit(grey_sample)
                # print("Initialisation Face grey classifier... OK")

                # build sample index
                for ix in range(0, iw - range_w, range_w):
                    for iy in range(0, ih - range_h, range_h):
                        sample_indexes.append((ix, iy))
                sample_indexes = set(sample_indexes)
                sample_indexes = sorted(sample_indexes,
                                        key=lambda v: (v[1], v[0]),
                                        reverse=False)

        else:
            (gf_x0, gf_y0, gf_w, gf_h) = get_face(cascade_classifier, image)
            d_gf_w = math.fabs(face_w - gf_w)
            d_gf_h = math.fabs(face_h - gf_h)
            if d_gf_w < int(0.25 * face_w) and d_gf_h < int(0.25 * face_h):
                face_detected = True
                face_w = gf_w
                face_h = gf_h
                face_x0 = gf_x0
                face_y0 = gf_y0
                face_x1 = gf_x0 + gf_w
                face_y1 = gf_y0 + gf_h
                face_xc = gf_x0 + (gf_w // 2)
                face_yc = gf_y0 + (gf_h // 2)
                face_d = math.sqrt(gf_w**2 + gf_h**2) / 2
                # range_w = face_w // range_pixel
                # range_h = face_h // range_pixel
                body_x0 = gf_x0 - gf_w
                body_x1 = body_x0 + (3 * gf_w)
                body_y0 = gf_y0 - int(0.5 * gf_h)
                body_y1 = body_y0 + int(4.5 * gf_h)
                # create face colour sample for One Class SVM
                sx0, sy0, sx1, sy1 = calc_rect(face_xc, face_yc, (face_w // 4))
                # colour_sample = blur_colour_image[sy0:sy1, sx0:sx1]

                # mean, stddev = cv.meanStdDev(colour_sample)
                # mean = mean.astype(int).reshape(1, 3)
                # stddev = stddev.astype(int).reshape(1, 3)
                # lower = mean - stddev * 3
                # upper = mean + stddev * 3
                # print(lower, upper)

                # colour_sample = colour_sample.reshape(-1, 3)
                # c0.fit(colour_sample)
                # print("Reinitialisation Face colour classifier... OK")

                grey_sample = blur_grey_image[sy0:sy1, sx0:sx1]
                mean, stddev = cv.meanStdDev(grey_sample)
                mean = mean.astype(int).reshape(1, 1)
                stddev = stddev.astype(int).reshape(1, 1)
                lower = mean - stddev * 3
                upper = mean + stddev * 3

                grey_sample = grey_sample.reshape(-1, 1)
                c0.fit(grey_sample)
                # print("Initialisation Face grey classifier... OK")

                grey_sample_0 = []
                grey_sample_1 = []
                if len(results_records) > 0:
                    for r in results_records[-1]:
                        if r[1] != []:
                            rx0, ry0 = r[1]
                            sx0, sy0, sx1, sy1 = calc_rect(rx0, ry0,
                                                           (face_w // 4))
                            if r[0] == 0:
                                grey_sample_0 = blur_grey_image[sy0:sy1,
                                                                sx0:sx1]
                                try:
                                    mean, stddev = cv.meanStdDev(grey_sample_0)
                                except:
                                    mean, stddev = np.array([[0]]), np.array([[0]])

                                mean = mean.astype(int).reshape(1, 1)
                                stddev = stddev.astype(int).reshape(1, 1)
                                lower0 = mean - stddev * 3
                                upper0 = mean + stddev * 3

                            elif r[0] == 1:
                                grey_sample_1 = blur_grey_image[sy0:sy1,
                                                                sx0:sx1]
                                try:
                                    mean, stddev = cv.meanStdDev(grey_sample_1)
                                except:
                                    mean, stddev = np.array([[0]]), np.array([[0]])

                                mean = mean.astype(int).reshape(1, 1)
                                stddev = stddev.astype(int).reshape(1, 1)
                                lower1 = mean - stddev * 3
                                upper1 = mean + stddev * 3

        # 2. Processing: feature extraction
        #                get high and low scale optical flow
        if face_detected and len(prev_grey_images) > 0:
            # flow0: low-scale optical flow
            pof = None
            if len(prevof0) > 0:
                pof = prevof0[-1]
            else:
                pof = None
            flow0_mag, flow0_ang, of = calc_optical_flow_new(prev_grey_images[-1],
                                                             grey_image, 0, pof)
            prevof0.append(of)
            flow0_hsv[..., 0] = flow0_ang / 2
            flow0_hsv[..., 1] = 255
            flow0_hsv[..., 2] = cv.normalize(flow0_mag, None, 0, 255,
                                             cv.NORM_MINMAX)
            flow0_bgr = cv.cvtColor(flow0_hsv, cv.COLOR_HSV2BGR)
            flow0_bw = cv.cvtColor(flow0_bgr, cv.COLOR_BGR2GRAY)
            flow0_bw[flow0_bw > 0] = 255
            if cv.countNonZero(flow0_bw) == 0 and len(flow0_bws) > 0:
                flow0_bw = flow0_bws[-1]

            # remove optical flow outside body
            body_mask[body_y0:body_y1, body_x0:body_x1] = 1
            # flow0_bw = flow0_bw * body_mask

            # calculate zscore from magnitude of low-scale optical flow
            # all frame zscore
            (a_summag_records,
             a_zscore_records) = record_zscore(flow0_mag,
                                               a_summag_records,
                                               a_zscore_records,
                                               3)

            # primary hand zscore
            if len(primary_records) > 1 and primary_records[-1] != []:
                pr1_id, pr1 = find_last_record(primary_records, 0)
                xpr, ypr = pr1[0][0]
                xp0, yp0, xp1, yp1 = calc_rect(xpr, ypr, (face_w // 2))
                p_mag = flow0_mag[yp0:yp1, xp0:xp1].copy()
                (p_summag_records,
                 p_zscore_records) = record_zscore(p_mag,
                                                   p_summag_records,
                                                   p_zscore_records, 3)

            # secondary hand zscore
            if len(secondary_records) > 1 and secondary_records[-1] != []:
                sr1_id, sr1 = find_last_record(secondary_records, 0)
                xsr, ysr = sr1[0][0]
                xs0, ys0, xs1, ys1 = calc_rect(xsr, ysr, (face_w // 2))
                s_mag = flow0_mag[ys0:ys1, xs0:xs1].copy()
                (s_summag_records,
                 s_zscore_records) = record_zscore(s_mag,
                                                   s_summag_records,
                                                   s_zscore_records, 3)
            # print("Zscore calculation... OK")

            # flow1: high-scale optical flow
            pof = None
            if len(prevof1) > 0:
                pof = prevof1[-1]
            else:
                pof = None
            flow1_mag, flow1_ang, of = calc_optical_flow_new(prev_grey_images[-1],
                                                             grey_image, 1, pof)
            prevof1.append(of)
            flow1_hsv[..., 0] = flow1_ang / 2
            flow1_hsv[..., 1] = 255
            flow1_hsv[..., 2] = cv.normalize(np.power(flow1_mag, 2), None, 0, 255,
                                             cv.NORM_MINMAX)
            flow1_bgr = cv.cvtColor(flow1_hsv, cv.COLOR_HSV2BGR)
            flow1_bw = cv.cvtColor(flow1_bgr, cv.COLOR_BGR2GRAY)
            flow1_bw[flow1_bw > 0] = 255
            if cv.countNonZero(flow1_bw) == 0 and len(flow1_bws) > 0:
                flow1_bw = flow1_bws[-1]

            # genereate the needle diagram
            for i in range(0, iw, 5):
               for j in range(0, ih, 5):
                   # cv.circle(flow1_needle, (i, j), 1, 0, -1)
                   r = flow1_mag[j, i] * 2
                   theta = flow1_ang[j, i] / 2
                   x = int(r * math.cos(math.radians(theta)))
                   y = int(r * math.sin(math.radians(theta)))
                   if x > 0 or y > 0:
                       cv.line(flow1_needle, (i, j), (i+x, j+y), 0, 1)

            # remove optical flow outside body
            # flow1_bw = flow1_bw * body_mask

            # clustering the high-scale optical flow
            flow1_k = 3
            flow1_r = np.array(flow1_bgr, dtype=np.float64) / 255
            flow1_w, flow1_h, flow1_d = flow1_r.shape
            flow1_r = np.reshape(flow1_r, (flow1_w * flow1_h, flow1_d))

            flow1_sample = shuffle(flow1_r, random_state=0)[:1000]
            flow1_clusters = KMeans(n_clusters=flow1_k).fit(flow1_sample)
            flow1_labels = flow1_clusters.predict(flow1_r)

            flow1_codebook = flow1_clusters.cluster_centers_
            for c in flow1_codebook:
                mask = np.zeros((flow1_w, flow1_h), dtype=np.uint8)
                flow1_masks.append(mask)

            codebook_d = flow1_codebook.shape[1]
            quan = np.zeros((flow1_w, flow1_h, codebook_d))
            label_idx = 0
            for i in range(flow1_w):
                for j in range(flow1_h):
                    quan[i][j] = flow1_codebook[flow1_labels[label_idx]]
                    flow1_masks[flow1_labels[label_idx]][i][j] = 255
                    label_idx += 1
            flow1_cirdof = np.array(quan * 255, dtype=np.uint8)

            # set the background area in flow1_cirdof to 0
            flow1_nonzero = [np.count_nonzero(mask) for mask in flow1_masks]
            flow1_max_nonzero = np.argmax(np.array(flow1_nonzero))
            flow1_cirdof[flow1_masks[flow1_max_nonzero] == 255] = 0
            flow1_masks.pop(flow1_max_nonzero)

            # separate the mask using contour
            flow1_masks_new = []
            for mask_id, mask in enumerate(flow1_masks):
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL,
                                                      cv.CHAIN_APPROX_SIMPLE)
                for c in range(len(contours)):
                    min_w = face_w // 2
                    min_h = face_h // 2
                    min_area = min_w * min_h
                    if cv.contourArea(contours[c]) > min_area:
                        mask_new = np.zeros_like(mask)
                        cv.drawContours(mask_new, contours, c, 255, -1)
                        flow1_masks_new.append(mask_new)

            flow1_masks_new = sorted(flow1_masks_new,
                                     key=lambda v:cv.countNonZero(v),
                                     reverse=True)
            flow1_masks = flow1_masks_new

            # Build flow1_cirdofbw
            for mask_id, mask in enumerate(flow1_masks):
                flow1_cirdofbw += mask
            flow1_cirdofbw[flow1_cirdofbw > 0] = 255

            # print("build CIRDOF... OK")

            # XXX:MCM
            # calculate optical flow
            pof = None
            if len(prevof2) > 0:
                pof = prevof2[-1]
            else:
                pof = None
            of_mag, of_ang, of = calc_optical_flow_new(prev_grey_images[-1],
                                                       grey_image, 1, pof)
            prevof2.append(of)
            multiclustermag = multiscaleoperation(of_mag, [2, 3, 4])
            # flow1_cirdofbw = multiclustermag.copy()
            # t = 0.1 * 255
            # flow1_cirdofbw[flow1_cirdofbw < t] = 0
            # flow1_cirdofbw[flow1_cirdofbw >= t] = 255
            # flow1_cirdofbw[flow1_cirdofbw > 0] = 255

            # Build skin mask
            # NOTE: the skin mask is need to be rebuilded when the movement
            # are detected.

            # skin_mask0 and skin_mask1 are produced using colour thresholding.
            # The colour reference is taken from tracked hands.
            skin_mask0 = cv.inRange(blur_grey_image, lower0, upper0)
            skin_mask1 = cv.inRange(blur_grey_image, lower1, upper1)

            # skin_mask2 is produced using oneclasssvm classifier.
            # The colour reference is taken from the face area.
            flat_grey_image = blur_grey_image.reshape(-1, 1)
            skin_mask2 = c0.predict(flat_grey_image)
            skin_mask2 = skin_mask2.reshape(ih, iw)
            skin_mask2[skin_mask2 > 0] = 255
            skin_mask2[skin_mask2 <= 0] = 0
            skin_mask2 = skin_mask2.astype(np.uint8)

            # This is the original skin_mask that is produced using colour
            # thresholding.
            skin_mask = cv.inRange(blur_grey_image, lower, upper)

            # creating the mask that shows intersection area between the
            # contour mask from the optical flow and the skin mask.
            # bin0 is the skin mask.
            # bin1 is the contour mask.
            # bin01 is the intersection mask.
            bin0[skin_mask2 > 0] = 1
            bin1[flow1_cirdofbw > 0] = 1
            bin01 = bin0 * bin1
            bin01[bin01 > 0] = 255

            # inspecting the zscore
            # NOTE: there are two types of ideas using zscore:
            # 1. rebuilding flow1_cirdofbw
            # if len(p_zscore_records) > 1 and len(s_zscore_records) > 1:
            #     if (p_zscore_records[-1] < p_zscore_records[-2] or
            #         s_zscore_records[-1] < s_zscore_records[-2]):
            #         flow1_masks += flow1_masks_records[-1]
            #         # Rebuild flow1_cirdofbw
            #         for mask_id, mask in enumerate(flow1_masks_records[-1]):
            #             flow1_cirdofbw += mask

            # 2. projection using zscore
            # if len(p_zscore_records) > 2:
            #     projection_primary = zscore_projection(p_zscore_records,
            #                                            primary_records)
            #     # print("primary projection calculation... OK")
            #
            # if len(s_zscore_records) > 2:
            #     projection_secondary = zscore_projection(s_zscore_records,
            #                                              secondary_records)
            #     # print("secondary projection calculation... OK")

        # 3. Processing: feature extraction:
        #                skin colour feature matching within optical flow mask
        if face_detected:
            if len(feature_records) > 0:
                feature_previous = feature_records[-1]
            else:
                feature_previous = []

            for mask_id, mask in enumerate(flow1_masks):
                coordinate_samples = []
                for x0, y0 in sample_indexes:
                    x1 = x0 + range_w
                    y1 = y0 + range_h
                    cx = x0 + range_w // 2
                    cy = y0 + range_h // 2

                    if x1 < iw and y1 < ih and mask[cy, cx] == 255:
                        flow0_sample = flow0_bw[y0:y1, x0:x1].copy()
                        non_zero = cv.countNonZero(flow0_sample)
                        non_zero = non_zero / (range_w * range_h)
                        if (non_zero > 0 and
                            (cx < face_x0 or cx > face_x1 or
                            cy < face_y0 or cy > face_y1)):
                            # NOTE: this feature_records checking needs to be
                            #       evaluated when this block is tabbed.
                            if len(feature_records) > 0:
                                for fp in feature_previous:
                                    fpx0, fpy0 = fp[0]
                                    dist = calc_euclidean(fpx0, fpy0, cx, cy)
                                    if dist < (2 * face_d):
                                        coordinate_samples.append([cx, cy])
                                        break
                            else:
                                coordinate_samples.append([cx, cy])

                if len(coordinate_samples) > 0:
                    points = np.array(coordinate_samples)
                    points = points.reshape(-1, 2)
                    cx = points[..., 0]
                    cy = points[..., 1]
                    mag = np.vstack(flow0_mag[cy.ravel(), cx.ravel()])
                    ang = np.vstack(flow0_ang[cy.ravel(), cx.ravel()])
                    # colours = np.vstack(blur_colour_image[cy.ravel(), cx.ravel()])
                    colours = np.vstack(blur_grey_image[cy.ravel(), cx.ravel()])

                    # data
                    X0 = points.astype(np.uint32)
                    X1 = colours
                    X2 = mag
                    X3 = ang

                    # reshape
                    X0 = X0.reshape(-1, 2)
                    # X1 = X1.reshape(-1, 3)
                    X1 = X1.reshape(-1, 1)
                    X2 = X2.reshape(-1, 1)
                    X3 = X3.reshape(-1, 1)
                    labels0 = c0.predict(X1)

                    # refinement clustering result with the optical flow map
                    for label_id, label in enumerate(labels0):
                        if label != -1:
                            X0x, X0y = X0[label_id]
                            if flow1_cirdofbw[X0y, X0x] == 0:
                                labels0[label_id] = -1

                    # removing the negative sample
                    X0_0 = X0[labels0 != -1]
                    X1_0 = X1[labels0 != -1]
                    X2_0 = X2[labels0 != -1]
                    X3_0 = X3[labels0 != -1]

                    # XXX
                    # for X0x, X0y in X0:
                    #     cv.circle(drawer_image, (X0x, X0y), 1, (0, 0, 0), -1)
                    # for X0x, X0y in X0_0:
                    #     cv.circle(drawer_image, (X0x, X0y), 1, (255, 255, 255), -1)

                    # rebuild the data X0_0, X1_0, X2_0, and X3_0 using OPTICS
                    if len(X0_0) > 2:
                        c1 = OPTICS(min_samples=2, algorithm='ball_tree',
                                    cluster_method='xi', xi=0.05).fit(X0_0)
                        if len(c1.cluster_hierarchy_) > 1:
                            c1_h = [e - s for s, e in c1.cluster_hierarchy_]
                            c1_h_sort = sorted(c1_h, key=lambda v:v, reverse=True)
                            c1_h_top = c1_h_sort[1]
                            c1_h_top_idx = np.where(c1_h == c1_h_top)
                            s, e = c1.cluster_hierarchy_[c1_h_top_idx[0][0]]

                            X0_0 = X0_0[c1.ordering_][s: e + 1]
                            # X1_0 = [blur_colour_image[y, x] for x, y in X0_0]
                            # X1_0 = np.array(X1_0).reshape(-1, 3)
                            X1_0 = [blur_grey_image[y, x] for x, y in X0_0]
                            X1_0 = np.array(X1_0).reshape(-1, 1)
                            X2_0 = [flow0_mag[y, x] for x, y in X0_0]
                            X2_0 = np.array(X2_0).reshape(-1, 1)
                            X3_0 = [flow0_ang[y, x] for x, y in X0_0]
                            X3_0 = np.array(X3_0).reshape(-1, 1)

                    # XXX
                    # for x, y in X0_0:
                    #     cv.circle(drawer_image, (x, y), 1, (255, 0, 0), -1)

                    if len(X0_0) > 0:
                        cluster_data.append((mask_id, X0_0, X1_0, X2_0, X3_0))

        # 4. Processing: Tracking:
        if face_detected:
            # 4.1 Get the initial location based on the available features.
            features = compile_feature(cluster_data,
                                       (flow0_mag, flow0_ang),
                                       (flow1_mag, flow1_ang))
            candidates = []
            if len(features) > 0:
                # candidates is sorted by sum of magnitude
                candidates = sorted(features, key=lambda v:v[1], reverse=True)
                pr1_id, pr1 = find_last_record(primary_records, 0)
                sr1_id, sr1 = find_last_record(secondary_records, 0)
                pr5, cpr5 = find_last_records(primary_records, 5)
                sr5, csr5 = find_last_records(secondary_records, 5)

                # NOTE: For early tracking data, assign the highest magnitude
                #       for primary and the furthest away from the primary for
                #       the secondary.
                #       - Try to fix it with fixed assignment.
                #       - Also this assignment will contain also empty list.
                if len(pr5) == 0 or len(sr5) == 0:
                    cp, candidates = get_candidate(candidates, 0, True)
                else:
                    cp, candidates = get_candidate(candidates, 0, False)
                    if pr1[0][1] > cp[0][1] or sr1[0][1] > cp[0][1]:
                        cp, candidates = get_nearest_candidate(candidates,
                                                               pr1[0], face_d)
                        if cp == ():
                            cp, candidates, get_nearest_candidate(candidates,
                                                                  cpr5, face_d)
                    else:
                        cp, candidates = get_candidate(candidates, 0, True)

                cs, candidates = get_furthest_candidate(candidates, cp, face_d)

                # add to current_primary and current_secondary
                if cp != ():
                    current_primary.append(cp)
                if cs != ():
                    current_secondary.append(cs)

                # print("First Tracking Phase... OK")
                # -------------------------------------------------------------
                # NOTE: data example:
                # -------------------------------------------------------------
                # primary_records:
                # [[([137, 241], 2.91, 284.96, 10.58, 0.0, 0.0, 1, 2.91)],
                #  [([133, 222], 813.05, 270.46, 2.73, 7.35, 3.45, 68, 32.47)]]
                # current_primary:
                # [([131, 188], 1454.88, 269.42, 0.20, 9.76, 3.45, 82, 15.22)]
                # -------------------------------------------------------------

            # 4.2 tracking refinement: check first data
            # if first data: no overlap with face
            # if not first data: must overlap with previous
            (current_primary, current_secondary,
             candidates) = check_assignment(primary_records,
                                            current_primary,
                                            secondary_records,
                                            current_secondary,
                                            candidates,
                                            face_xc, face_yc, face_d)

            (current_secondary, current_primary,
             candidates) = check_assignment(secondary_records,
                                            current_secondary,
                                            primary_records,
                                            current_primary,
                                            candidates,
                                            face_xc, face_yc,face_d)
            # print("First Refinement Phase... OK")

            # 4.3 tracking refinement: check swapping problem
            # only calculated when there are two current data.
            if len(current_primary) > 0 and len(current_secondary) > 0:
                xp, yp = current_primary[0][0]
                xs, ys = current_secondary[0][0]

                if len(primary_records) > 0 and len(secondary_records) == 0:
                    # matching with primary record only
                    pr1_id, pr1 = find_last_record(primary_records, 0)
                    if len(pr1) > 0:
                        xpr1, ypr1 = pr1[0][0]
                        distpp = calc_euclidean(xpr1, ypr1, xp, yp)
                        distps = calc_euclidean(xpr1, ypr1, xs, ys)
                        if distpp > distps:
                            (current_primary,
                            current_secondary) = (current_secondary,
                                                current_primary)

                elif len(primary_records) == 0 and len(secondary_records) > 0:
                    # matching with secondary record only
                    sr1_id, sr1 = find_last_record(secondary_records, 0)
                    if len(sr1) > 0:
                        xsr1, ysr1 = sr1[0][0]
                        distsp = calc_euclidean(xsr1, ysr1, xp, yp)
                        distss = calc_euclidean(xsr1, ysr1, xs, ys)
                        if distsp < distss:
                            (current_primary,
                            current_secondary) = (current_secondary,
                                                current_primary)

                else:
                    # matching with both primary and secondary records.
                    pr1_id, pr1 = find_last_record(primary_records, 0)
                    sr1_id, sr1 = find_last_record(secondary_records, 0)
                    if len(pr1) > 0 and len(sr1) > 0:
                        xpr1, ypr1 = pr1[0][0]
                        xsr1, ysr1 = sr1[0][0]
                        distpp = calc_euclidean(xpr1, ypr1, xp, yp)
                        distps = calc_euclidean(xpr1, ypr1, xs, ys)
                        distsp = calc_euclidean(xsr1, ysr1, xp, yp)
                        distss = calc_euclidean(xsr1, ysr1, xs, ys)
                        dist0 = distpp + distss
                        dist1 = distps + distsp
                        if dist0 > dist1:
                            (current_primary,
                            current_secondary) = (current_secondary,
                                                current_primary)
            # print("Second Refinement Phase... OK")

            # 4.4 tracking refinement: checking for better candidate
            # print("Candidate(s) available:", len(candidates))

            if len(current_primary) > 0 and len(primary_records) > 0:
                (candidates,
                 current_primary) = compare_candidates(candidates,
                                                       current_primary,
                                                       primary_records,
                                                       face_d)
            if len(current_secondary) > 0 and len(secondary_records) > 0:
                (candidates,
                 current_secondary) = compare_candidates(candidates,
                                                         current_secondary,
                                                         secondary_records,
                                                         face_d)
            # print("Third Refinement Phase... OK")

        # # 5. zscore calculation and projection based on zscore
        # if len(projection_primary) > 0:
        #     px, py = projection_primary[0][0]
        #     if (px >= body_x0 and px <= body_x1 and
        #         py >= body_y0 and py <= body_y1):
        #         current_primary = projection_primary.copy()
        #         # print("primary projection applied... OK")
        #
        # if len(projection_secondary) > 0:
        #     px, py = projection_secondary[0][0]
        #     if (px >= body_x0 and px <= body_x1 and
        #         py >= body_y0 and py <= body_y1):
        #         current_secondary = projection_secondary.copy()
        #         # print("secondary projection applied... OK")

        # # 6. projection based on previous frame and next frame
        # # ---------------------------------------------------------------------
        # # NOTE: projection for next frame
        # # ---------------------------------------------------------------------
        # # this part is to calculate average movement between 2 frames.
        # # the first frame is the current frame, the second frame is
        # # current frame - 2 frames. The projection is used as the tracking
        # # result for current frame - 1 frame.
        # # ---------------------------------------------------------------------
        # averaging_primary = averaging_projection(primary_records,
        #                                          current_primary)
        # averaging_secondary = averaging_projection(secondary_records,
        #                                            current_secondary)

        # # 7. final decision
        # # final decision is taken from averaging primary
        # # the secondary result id is 0 and the primary result id is 1
        # if (len(averaging_primary) > 0 and
        #     not check_stationary(current_primary, primary_records, 5)):
        #     result_primary = (1, averaging_primary[0])
        # else:
        #     result_primary = (1, [])
        #
        # if (len(averaging_secondary) > 0 and
        #     not check_stationary(current_secondary, secondary_records, 5)):
        #     result_secondary = (0, averaging_secondary[0])
        # else:
        #     result_secondary = (0, [])
        # result = [result_secondary, result_primary]
        if len(current_primary) > 0:
            result_primary = (1, current_primary[0][0])
        else:
            result_primary = (1, [])
        if len(current_secondary) > 0:
            result_secondary = (0, current_secondary[0][0])
        else:
            result_secondary = (0, [])
        result = [result_secondary, result_primary]

        # XXX
        # 8. Drawing
        # 8.1 Drawing face boundary
        # cv.rectangle(drawer_image, (face_x0, face_y0), (face_x1, face_y1),
        #              (0, 0, 255), 1)
        # 8.2 Drawing body boundary
        # cv.rectangle(drawer_image, (body_x0, body_y0), (body_x1, body_y1),
        #              (0, 0, 255), 1)
        # 8.3 Drawing tracking Result
        # primary hand
        # for hand in current_primary:
        #     hxc, hyc = hand[0]
        #     hx0, hy0, hx1, hy1 = calc_rect(hxc, hyc, (face_w // 2))
        #     cv.rectangle(drawer_image, (hx0, hy0), (hx1, hy1),
        #                  (0, 0, 255), 1)
        # Secondary hand
        # for hand in current_secondary:
        #     hxc, hyc = hand[0]
        #     hx0, hy0, hx1, hy1 = calc_rect(hxc, hyc, (face_w // 2))
        #     cv.rectangle(drawer_image, (hx0, hy0), (hx1, hy1),
        #                  (0, 0, 255), 1)
        # 8.4 Drawing averaging result
        # primary hand
        # for hand in averaging_primary:
        #     hxc, hyc = hand
        #     hx0, hy0, hx1, hy1 = calc_rect(hxc, hyc, (face_w // 2))
        #     cv.rectangle(drawer_images[-1], (hx0, hy0), (hx1, hy1),
        #                  (0, 255, 0), 1)
        # Secondary hand
        # for hand in averaging_secondary:
        #     hxc, hyc = hand
        #     hx0, hy0, hx1, hy1 = calc_rect(hxc, hyc, (face_w // 2))
        #     cv.rectangle(drawer_images[-1], (hx0, hy0), (hx1, hy1),
        #                  (0, 255, 0), 1)
        # 8.5 Drawing ground truth
        # g = gt_all[frame_id]
        # gx0, gy0, gx1, gy1 = calc_rect(g[1], g[2], (face_w // 2))
        # # cv.rectangle(drawer_image, (gx0, gy0), (gx1, gy1), (255, 255, 255), 1)
        # cv.circle(drawer_image, (g[1], g[2]), 3, (255, 0, 0), -1)
        # print("gt:", g[1], g[2])
        # gx0, gy0, gx1, gy1 = calc_rect(g[3], g[4], (face_w // 2))
        # cv.rectangle(drawer_image, (gx0, gy0), (gx1, gy1), (255, 255, 255), 1)
        # cv.circle(drawer_image, (g[3], g[4]), 3, (255, 0, 0), -1)
        # print("gt:", g[3], g[4])
        # 8.5 Drawing initial position
        # if frame_id == 0:
        #     g = init_position[0]
        #     gx0, gy0, gx1, gy1 = calc_rect(g[1], g[2], (face_w // 2))
        #     cv.rectangle(drawer_image, (gx0, gy0), (gx1, gy1), (0, 0, 255), 1)
        #     g = init_position[1]
        #     gx0, gy0, gx1, gy1 = calc_rect(g[1], g[2], (face_w // 2))
        #     cv.rectangle(drawer_image, (gx0, gy0), (gx1, gy1), (0, 0, 255), 1)
        # 8.6 Drawing results
        for r in result:
            if r[1] != []:
                rx, ry = r[1]
                rx0, ry0, rx1, ry1 = calc_rect(rx, ry, (face_w // 2))
                cv.rectangle(drawer_images[-1], (rx0, ry0), (rx1, ry1),
                            (0, 0, 255), 1)
                cv.circle(drawer_images[-1], (rx, ry), 1, (0, 0, 255), -1)
        # 8.7 Back to features
        # if face_detected:
        #     for f in features:
        #         print("hc:", f[0])
        #         hc = f[0]
        #         hcgt0 = calc_euclidean(hc[0], hc[1], g[1], g[2])
        #         hcgt1 = calc_euclidean(hc[0], hc[1], g[3], g[4])
        #         if hcgt0 > hcgt1:
        #             hcgt0, hcgt1 = hcgt1, hcgt0
        #         print("hc:", hc, hcgt0, hcgt1)
        #         hcgt.append((frame_id, hc[0], hc[1], hcgt0, hcgt1))
        #         cv.circle(drawer_image, (hc[0], hc[1]), 3, (0, 0, 255), -1)

        # End
        drawer_images.append(drawer_image)
        prev_colour_images.append(colour_image)
        prev_grey_images.append(grey_image)
        flow0_bgrs.append(flow0_bgr)
        flow1_bgrs.append(flow1_bgr)
        flow0_bws.append(flow0_bw)
        flow1_bws.append(flow1_bw)

        flow1_cirdofs.append(flow1_cirdof)
        flow1_cirdofbws.append(flow1_cirdofbw)
        flow1_masks_records.append(flow1_masks)
        flow1_needles.append(flow1_needle)
        flow1_mags.append(flow1_mag)

        bins.append(bin01)
        skin_masks.append(skin_mask)
        # XXX:MCM
        mcms.append(multiclustermag)
        if save_index > 0:
            tpoint = result.copy()
            tpoint.append((2, [face_xc, face_yc]))
            # print(save_index-1, tpoint)
            tpoints.append([save_index-1, tpoint])

        # if frame_id == 0:
        #     g = init_position[1]
        #     init_feature = [([g[1], g[2]], 0, 0, 0, 0, 0, 0, 0)]
        #     primary_records.append(init_feature)
        #     g = init_position[0]
        #     init_feature = [([g[1], g[2]], 0, 0, 0, 0, 0, 0, 0)]
        #     secondary_records.append(init_feature)

        if current_primary != []:
            primary_records.append(current_primary)

        if current_secondary != []:
            secondary_records.append(current_secondary)

        if feature_current != []:
            feature_records.append(feature_current)

        # result is recorded frame-1 from current frame
        if frame_id > 0:
            results_records.append(result)

        # cv.imshow("original", drawer_image)
        # Display
        if save_index >= (win_length-1):
            # cv.imshow("output", drawer_images[-2])
            # cv.imshow("cirdof", flow1_cirdofs[-2])
            # cv.imshow("needle", flow1_needles[-2])
            # cv.imshow("contour mask", flow1_cirdofbws[-2])
            # cv.imshow("skin mask", skin_masks[-2])
            # cv.imshow("bin", bins[-2])
            # XXX:MCM
            # cv.imshow("mcm", mcms[-2])

            # cv.imshow("contour mask", flow1_cirdofbw)
            # cv.imshow("original", drawer_image)
            # cv.imshow("skin_mask", skin_mask)
            # cv.imshow("skin_mask0", skin_mask0)
            # cv.imshow("skin_mask1", skin_mask1)
            # cv.imshow("skin_mask2", skin_mask2)
            # cv.imshow("bin", bin01)

            # imgrgb0 = cv.cvtColor(drawer_images[-2], cv.COLOR_BGR2RGB)
            # imgjpeg = Image.fromarray(imgrgb0)
            # imgjpeg.save("v50_output_"+str(save_index-1)+".jpeg", "JPEG",
            #              dpi=(1000,1000), quality=100)
            # imgrgb1 = cv.cvtColor(flow1_cirdofs[-2], cv.COLOR_BGR2RGB)
            # imgjpeg = Image.fromarray(imgrgb1)
            # imgjpeg.save("v50_cirdof_"+str(save_index-1)+".jpeg", "JPEG",
            #              dpi=(1000,1000), quality=100)
            # save_image_0 = display_single_window(drawer_images,
            #                                      "drawer_images",
            #                                      save_index, win_length-1)
            # save_image_1 = display_single_window(flow0_bws,
            #                                      "flow0_bws",
            #                                      save_index, win_length-1)
            # save_image_2 = display_single_window(flow1_bgrs,
            #                                      "flow1_bgrs",
            #                                      save_index, win_length-1)
            # save_image_3 = display_single_window(flow1_cirdofs,
            #                                      "flow1_cirdofs",
            #                                      save_index, win_length-1)
            # save_image_4 = display_single_window(flow1_cirdofbws,
            #                                      "flow1_cirdofbws",
            #                                      save_index, win_length-1)

            # cv.moveWindow("output", 0, 0)
            # cv.moveWindow("cirdof", 0, 352)
            # cv.moveWindow("needle", 370, 352)
            # cv.moveWindow("contour mask", 0, 672)
            # cv.moveWindow("skin mask", 370, 672)
            # cv.moveWindow("bin", 740, 672)

            # NOTE: uncomment this block for saving the window
            # if (save_index % win_length) == (win_length-1):
            if save_index % 4 == 0:
                # cv.imwrite("drawer_images2_"+str(save_window_index)+".png",
                #            save_image_0)
                # cv.imwrite("flow_images_"+str(save_window_index)+".png",
                #            save_image_1)
                # cv.imwrite("flow1_images_"+str(save_window_index)+".png",
                #            save_image_2)
                # cv.imwrite("otsu_images_"+str(save_window_index)+".png",
                #            save_image_3)
                # cv.imwrite("quan_images_"+str(save_window_index)+".png",
                #            save_image_4)
                save_window_index = save_window_index + 1

        save_index += 1
        # k = cv.waitKey(0) & 0xFF
        # if k == ord('q'):
        #    cv.destroyAllWindows()
        #    break

    picklefile = dir_name + "tpoints1"
    with open(picklefile, "wb") as pf:
        pickle.dump(tpoints, pf)
    pf.close()
    print(picklefile, "saved.")

    # with open(picklefile, "rb") as pf:
    #     loadpickle = pickle.load(pf)
    #     print("loadpickle:", loadpickle)
    # pf.close()
    # print(picklefile, "loaded.")


    # # create allcirdifbw image
    # allcirdofbw = np.zeros((ih, iw), dtype=np.uint8)
    # # for i, cirdofbw in enumerate(flow1_cirdofbws[:-1]):
    # #     cirdofbw[cirdofbw > 0] = 1
    # #     allcirdofbw += cirdofbw
    # for i, cirdofbw in enumerate(flow1_cirdofbws[:-1]):
    #     allcirdofbw[cirdofbw > 0] = (i+1)
    # n = len(flow1_cirdofbws[:-1])
    # for i, cirdofbw in enumerate(flow1_cirdofbws[:-1]):
    #     allcirdofbw[allcirdofbw == (i+1)] = int(((i+1) / n) * 255)
    # allcirdofbw = cv.cvtColor(allcirdofbw, cv.COLOR_GRAY2RGB)
    #
    # # create allbin image
    # allbin = np.zeros((ih, iw), dtype=np.uint8)
    # allbin2 = np.zeros((ih, iw), dtype=np.uint8)
    # for i, bin in enumerate(bins[:-1]):
    #     allbin[bin > 0] = (i+1)
    # n = len(bins[:-1])
    # for i, bin in enumerate(bins[:-1]):
    #     allbin[allbin == (i+1)] = int(((i+1) / n) * 255)
    # allbin = cv.cvtColor(allbin, cv.COLOR_GRAY2RGB)
    #
    # # create allprevgrey image
    # allprevgrey = np.zeros((ih, iw), dtype=np.float64)
    # n = len(prev_grey_images[:-1])
    # for i, prevgrey in enumerate(prev_grey_images[:-1]):
    #     prevgrey = prevgrey / n
    #     allprevgrey += prevgrey
    # allprevgrey = allprevgrey.astype(np.uint8)
    #
    # # create result_plot
    # result_plot = np.ones((ih, iw, 3), dtype=np.uint8)
    # result_plot = result_plot * 255
    # n_results = len(results_records)
    # r0 = []
    # r1 = []
    # for i, result in enumerate(results_records):
    #     for r in result:
    #         if r[0] == 0 and r[1] != []:
    #             cv.circle(result_plot, r[1], 2, (0, 0, 255), -1)
    #             # cv.circle(allcirdofbw, r[1], 2, (0, 0, 255), -1)
    #             r0.append(r[1])
    #         elif r[0] == 1 and r[1] != []:
    #             cv.circle(result_plot, r[1], 2, (255, 0, 0), -1)
    #             # cv.circle(allcirdofbw, r[1], 2, (255, 0, 0), -1)
    #             r1.append(r[1])
    # for i in range(len(r0)-1):
    #     # cv.line(allbin, r0[i], r0[i+1], (0, 0, 255), 3)
    #     cv.line(allbin2, r0[i], r0[i+1], int(((i+1) / n) * 255), 5)
    # for i in range(len(r1)-1):
    #     # cv.line(allbin, r1[i], r1[i+1], (255, 0, 0), 3)
    #     cv.line(allbin2, r1[i], r1[i+1], int(((i+1) / n) * 255), 5)

    # cv.destroyAllWindows()
    # cv.imshow("result_plot", result_plot)
    # cv.imshow("allcirdofbw", allcirdofbw)
    # cv.imshow("allbin", allbin)
    # cv.imshow("allbin2", allbin2)
    # cv.imshow("allprevgrey", allprevgrey)
    # cv.moveWindow("result_plot", 0, 0)
    # cv.moveWindow("allcirdofbw", 370, 0)
    # cv.moveWindow("allbin", 740, 0)
    # cv.moveWindow("allbin2", 740, 352)
    # cv.moveWindow("allprevgrey", 1110, 0)
    # cv.waitKey(0) & 0xFF
    # cv.imwrite(save_name + "allbin.png", allbin)
    # cv.imwrite(save_name + "allbin2.png", allbin2)
    # cv.imwrite(save_name + "allcirdofbw.png", allcirdofbw)
    # cv.imwrite(save_name + "allprevgrey.png", allprevgrey)
    # print("r0:", r0)
    # print("r1:", r1)

    # save all cirdof
    # for i, cirdofbw in enumerate(flow1_cirdofbws):
    #     cv.imwrite(save_name + "cirdofbw" + str(i).zfill(2) + ".png", cirdofbw)

    # # output_dir = dir_name.split("/")
    # # output_filename = output_dir[-1]
    # # print("./" + output_filename + ".csv")
    # # with open("./" + output_filename + ".csv", mode="w") as individualcsvfile:
    # #     individualwriter = csv.writer(individualcsvfile, delimiter=",")
    # #     for hc in hcgt:
    # #         individualwriter.writerow(hc)
    # #         individualcsvfile.flush()
    # # individualcsvfile.close()

    # # calculate tracking accuracy
    # # calculate COL
    # # calculate TER
    # tertresh = 20
    #
    # col00, col11, col01, col10 = None, None, None, None
    # ter00, ter11, ter01, ter10 = None, None, None, None
    #
    # allcol00, allcol11, allcol01, allcol10 = [], [], [], []
    # allter00, allter11, allter01, allter10 = [], [], [], []
    #
    # output_dir = dir_name.split("/")
    # output_filename = output_dir[-1]
    # with open("./" + output_filename + ".csv", mode="w") as individualcsvfile:
    #     individualwriter = csv.writer(individualcsvfile, delimiter=",")
    #     for result_id, result in enumerate(results_records):
    #         g = gt_all[result_id]
    #         g0x, g0y = g[1], g[2]
    #         g1x, g1y = g[3], g[4]
    #
    #         if len(result[0][1]):
    #             r0x, r0y = result[0][1]
    #             col00 = calc_euclidean(r0x, r0y, g0x, g0y)
    #             col01 = calc_euclidean(r0x, r0y, g1x, g1y)
    #             if col00 > tertresh:
    #                 ter00 = 1
    #             else:
    #                 ter00 = 0
    #             if col01 > tertresh:
    #                 ter01 = 1
    #             else:
    #                 ter01 = 0
    #         else:
    #             r0x, r0y = 0, 0
    #             col00 = None
    #             col01 = None
    #             ter00 = None
    #             ter01 = None
    #
    #         if len(result[1][1]):
    #             r1x, r1y = result[1][1]
    #             col11 = calc_euclidean(r1x, r1y, g1x, g1y)
    #             col10 = calc_euclidean(r1x, r1y, g0x, g0y)
    #             if col11 > tertresh:
    #                 ter11 = 1
    #             else:
    #                 ter11 = 0
    #             if col10 > tertresh:
    #                 ter10 = 1
    #             else:
    #                 ter10 = 0
    #         else:
    #             r1x, r1y = 0, 0
    #             col11 = None
    #             col10 = None
    #             ter11 = None
    #             ter10 = None
    #
    #         if col00 is not None:
    #             allcol00.append(col00)
    #             allter00.append(ter00)
    #         if col01 is not None:
    #             allcol01.append(col01)
    #             allter01.append(ter01)
    #         if col11 is not None:
    #             allcol11.append(col11)
    #             allter11.append(ter11)
    #         if col10 is not None:
    #             allcol10.append(col10)
    #             allter10.append(ter10)
    #
    #         individualwriter.writerow([result_id, col00, col01, col11, col10,
    #                                               ter00, ter01, ter11, ter10])
    #         individualcsvfile.flush()
    #         # print(result_id, "recorded")
    #
    #     meancol00, meancol01, meancol11, meancol10 = 0, 0, 0, 0
    #     # print("\nResults:")
    #     if len(allcol00) > 0:
    #         meancol00 = statistics.mean(allcol00)
    #         # print("- meancol00 (primary noswap):", meancol00)
    #     if len(allcol01) > 0:
    #         meancol01 = statistics.mean(allcol01)
    #         # print("- meancol01 (primary swap):", meancol01)
    #     if len(allcol11) > 0:
    #         meancol11 = statistics.mean(allcol11)
    #         # print("- meancol11 (secondary noswap):", meancol11)
    #     if len(allcol10) > 0:
    #         meancol10 = statistics.mean(allcol10)
    #         # print("- meancol10 (secondary swap):", meancol10)
    #
    #     meanter00, meanter01, meanter11, meanter10 = 0, 0, 0, 0
    #     if len(allter00) > 0:
    #         meanter00 = statistics.mean(allter00)
    #         # print("- meanter00 (primary no swap):", meanter00)
    #     if len(allter01) > 0:
    #         meanter01 = statistics.mean(allter01)
    #         # print("- meanter01 (primary swap):", meanter01)
    #     if len(allter11) > 0:
    #         meanter11 = statistics.mean(allter11)
    #         # print("- meanter11 (secondary no swap):", meanter11)
    #     if len(allter10) > 0:
    #         meanter10 = statistics.mean(allter10)
    #         # print("- meanter10 (secondary swap):", meanter10)
    #
    #     individualwriter.writerow(["mean", meancol00, meancol01, meancol11, meancol10,
    #                                        meanter00, meanter01, meanter11, meanter10])
    #     individualcsvfile.flush()
    #     print(video_name, "done...")
    # individualcsvfile.close()
