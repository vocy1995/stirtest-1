""" calculates error of model on STIR labelled dataset.
Averages nearest endpoint error over clips"""
import cv2
import numpy as np
import json
import sys
from tqdm import tqdm
from collections import defaultdict
import itertools
from scipy.interpolate import Rbf
from STIRLoader import STIRLoader
import random
import torch
import argparse
from scipy.spatial import KDTree
from pathlib import Path
import logging
from testutil import *
import csrt
import mft
import raft
import MFT.utils.vis_utils as vu
import os
from lightglue import LightGlue, SuperPoint, DISK, ALIKED
from lightglue.utils import load_image, rbd
from dataclasses import dataclass 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

match_on = []
match_list = None
reset_query = False
@dataclass 
class threshold:
    op: int = 120
    radius: int = 45
    distance_gap: int = 45
    num_points: int = 5
    straight: int = 4
    follow: int = 20

@dataclass 
class match_threshold:
    depth_confidence: float = 0.97
    filter_threshold: float = 0.12
    max_num_keypoints: int = 2048
    

modeldict = {"MFT": mft.MFTTracker,
           "CSRT": csrt.CSRTMultiple,
           "RAFT": raft.RAFTTracker,}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_opencv(image):
    """ " converts N im tensor to numpy in BGR"""
    image = image.squeeze(0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonsuffix",
        type=str,
        default="test",
        help="output suffix for json",
    )
    parser.add_argument(
        "--modeltype",
        type=str,
        default="RAFT",
        help="CSRT, MFT or RAFT",
    )
    parser.add_argument(
        "--ontestingset",
        type=int,
        default="1",
        help="whether on the testing set. Testing set provides no access to end segs.",
    )
    parser.add_argument(
        "--showvis",
        type=int,
        default="1",
        help="whether to show vis",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default="8",
        help="number of sequences to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default="0",
        help="random seed",
    )
    args = parser.parse_args()
    args.batch_size = 1  # do not change, only one for running
    return args


def drawpoints(im, points, color):
    global match_on
    global match_list
    for idx, pt in enumerate(points[:, :]):
        pt = pt.astype(int)
        im = cv2.circle(im, tuple(pt), 3, color, thickness=1)
        im = cv2.circle(im, tuple(pt), 12, color, thickness=3)
        if match_on[idx]:
            im = cv2.circle(im, tuple(pt), threshold.radius, [0, 255, 0], thickness=2)
            for match in match_list[idx]:
                match = match.astype(int)
                im = cv2.circle(im, tuple(match), 3, [255, 0, 255], thickness=2,lineType=16)
    return im

def tensor_image_to_torch(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize the image tensor and reorder the dimensions.
    Assumes the input image tensor has shape (batch_size, channels, height, width).
    """
    image = image.permute(2, 0, 1)
    # Normalize pixel values to [0, 1]
    image_normalized = image / 255.0
    return image_normalized.float()

def match_point(image0, image1,extractor, matcher):
    #point extraction
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # point matching
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return m_kpts0, m_kpts1


def trackanddisplay(
    startpointlist,
    dataloader,
    radius=3,
    thickness=-1,
    showvis=False,
    modeltype="CSRT",
    track_writer=None
):
    """tracks and displays pointlist over time
    returns pointlist at seq end"""
    num_pts = startpointlist.shape[1]
    model = modeldict[modeltype]()
    dataloaderiter = iter(dataloader)
    startdata = todevice(next(dataloaderiter))
    assert len(startdata["ims"]) == 1  # make sure no batches
    colors = (np.random.randint(0, 255, 3 * num_pts)).reshape(num_pts, 3)

    pointlist = startpointlist.cpu().numpy()

    #Added 
    global match_on
    global match_list
    global reset_query
    pts_replace = [False] * len(startpointlist)  # initial = false
    match_on = [False] * len(startpointlist)
    match_list = [np.zeros] * len(startpointlist)
    initial_points = np.copy(pointlist)
    firstframe = True
    firstimage = None
    thresh = threshold()
    pts_straight = [0] * len(startpointlist)
    
    
    extractor = ALIKED().eval().to(device)  # extractor(detection model) 
    matcher = LightGlue(features='aliked',depth_confidence=match_threshold.depth_confidence, filter_threshold=match_threshold.filter_threshold).eval().to(device)


    for data in tqdm(dataloaderiter):

        nextdata = todevice(data)
        impair = [
            [*startdata["ims"], *nextdata["ims"]],
            [*startdata["ims_right"], *nextdata["ims_right"]],
        ]
        ims_ori_pair = [*startdata["ims_ori"], *nextdata["ims_ori"]]
        
        if firstframe and showvis:
            color = [0, 0, 255]
            startframe = drawpoints(
                convert_to_opencv(ims_ori_pair[0]), pointlist, color
            )
        if firstframe:
            firstimage = ims_ori_pair[0].clone()
            firstframe = False
        
        image_refined_0 = (firstimage.squeeze(0).permute(2, 0, 1) / 255.0).to('cuda')
        image_refined_1 = (ims_ori_pair[1].squeeze(0).permute(2, 0, 1) / 255.0).to('cuda')
        m_kpts0, m_kpts1 = match_point(image_refined_0, image_refined_1,extractor, matcher)
        point_list_L = m_kpts0.cpu().numpy()
        point_list_R = m_kpts1.cpu().numpy()






        #Optical flow tracking  
        pointlist_optical, reset_query = model.trackpoints2D(pointlist, ims_ori_pair,reset_query)
        startdata = nextdata
        for idx in range(len(startpointlist)):
            dist_optical = np.linalg.norm(pointlist_optical[idx] - pointlist[idx])
            if dist_optical < thresh.op:
                pointlist[idx] = np.copy(pointlist_optical[idx])
                pts_replace[idx] = True
            else:
                pts_replace[idx] = False
        


        for idx, input_point in enumerate(initial_points):
            input_point_distance = np.linalg.norm(point_list_L - input_point, axis=1)
            dist = np.where(input_point_distance < thresh.radius)
            point_list_L_filtered = point_list_L[dist]
            point_list_R_filtered = point_list_R[dist]



        
            if len(point_list_L_filtered) > thresh.num_points:
                M, _ = cv2.estimateAffine2D(np.array(point_list_L_filtered), np.array(point_list_R_filtered), method=cv2.RANSAC)
                transformed_input_point = np.dot(M, np.hstack((initial_points[idx], 1)))[:2]
                dist_gap = np.linalg.norm(transformed_input_point - pointlist[idx])
                point_follow = False
                for point in point_list_R_filtered:
                    if np.linalg.norm(point - pointlist[idx]) < thresh.follow:
                        point_follow = True
                        break


                if point_follow: # if point is close to features, do not update
                    pts_straight[idx] = 0
                    match_on[idx] = False
                elif pts_replace == False: # if optical flow failed to update, use point matching
                    pointlist[idx] = np.copy(transformed_input_point)
                    model.firstframe = True
                    match_on[idx] = True
                    match_list[idx] = point_list_R_filtered.copy()
                    reset_query = True
                elif dist_gap > thresh.distance_gap: #if estimated optical flow is too far from point matching, use point matching
                    pts_straight[idx] += 1
                    if pts_straight[idx] >= thresh.straight:
                        pointlist[idx] = np.copy(transformed_input_point)
                        pts_straight[idx] = 0
                        model.firstframe = True
                        match_on[idx] = True
                        match_list[idx] = point_list_R_filtered.copy()
                        reset_query = True
                else:
                    pts_straight[idx] = 0
                    match_on[idx] = False
            else:
                match_on[idx] = False
        
        
        

        if showvis:
            imend = convert_to_opencv(ims_ori_pair[1])

            color = [0, 255, 0]
            drawpoints(imend, pointlist, color)

            showimage("imagetrack", imend)
            if track_writer:
                track_writer.write(imend)
            cv2.waitKey(1)
    if showvis:
        lastframe = convert_to_opencv(ims_ori_pair[1])
        return pointlist, startframe, lastframe
    else:
        return pointlist




if __name__ == "__main__":
    args = getargs()
    logging.basicConfig(level=logging.INFO)
    modeltype = args.modeltype

    with open("config.json", "r") as f:
        config = json.load(f)
    args.datadir = config["datadir"]
    datasets = STIRLoader.getclips(datadir=args.datadir)
    random.seed(args.seed)
    random.shuffle(datasets)
    errors_avg = defaultdict(int)
    errors_control_avg = 0
    num_data = args.num_data
    num_data_name = num_data
    if num_data_name == -1:
        num_data_name = "all"

    errorlists = {}
    positionlists = {}
    data_used_count = 0
    for ind, dataset in enumerate(datasets[:num_data]):
        try:
            outdir = Path(f'./results{args.modeltype}_{args.seed}/{ind:03d}{modeltype}_tracks.mp4')
            if args.showvis:
                track_writer = vu.VideoWriter(outdir, fps=10, images_export=False)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True
            )
            startseg = np.array(dataset.dataset.getstartseg()).sum(axis=2)
            
            try:
                pointlist_start = np.array(dataset.dataset.getstartcenters())
            except IndexError as e:
                print(f"{e} error on dataset load, continuing")
                continue
            pointlist_start = torch.from_numpy(pointlist_start).to(device)
            if pointlist_start.shape[0] < 1:
                continue
            if not args.ontestingset:
                endseg = dataset.dataset.getendseg()
                endseg = np.array(endseg).sum(axis=2)
                try:
                    pointlistend = np.array(dataset.dataset.getendcenters())
                except IndexError as e:
                    print(f"{e} error on dataset load, continuing")
                    continue
                pointlistend = torch.from_numpy(pointlistend).to(device)
                errors_control = pointlossunidirectional(
                    pointlist_start, pointlistend
                )["averagedistance"]
            

            if args.showvis:
                showimage("seg_start", startseg)
                showimage("seg_end", endseg)
                cv2.waitKey(1)
                end_estimates, startframe, lastframe = trackanddisplay(
                    pointlist_start,
                    dataloader,
                    showvis=args.showvis,
                    modeltype=modeltype,
                    track_writer=track_writer
                )
            else:
                end_estimates = trackanddisplay(
                    pointlist_start,
                    dataloader,
                    showvis=args.showvis,
                    modeltype=modeltype,
                )


            positionlists[str(dataset.dataset.basename)] = end_estimates

            if not args.ontestingset: # Log endpoint error
                errortype = "endpointerror"
                print(f"DATASET_{ind}: {dataset.dataset.basename}")
                errors_control_avg = errors_control_avg + errors_control
                errordict = {}
                errordict[f"{errortype}_control"] = errors_control

                errors = pointlossunidirectional(end_estimates, pointlistend)
                errors_imgavg = errors["averagedistance"]
                errorname = f"{errortype}_{modeltype}"
                errordict[errorname] = errors_imgavg
                print(f"{errorname}_combined: {errors_imgavg}")
                errors_avg[modeltype] = errors_avg[modeltype] + errors_imgavg
                errorlists[str(dataset.dataset.basename)] = errordict
            data_used_count += 1

            if args.showvis:

                imend = lastframe
                color = [0, 255, 0]
                drawpoints(imend, end_estimates, color)

                displacements = errors["displacements"]
                for pt, displacement in zip(end_estimates, displacements):
                    pt = pt.astype(int)
                    displacement = displacement.astype(int)
                    color = [0, 0, 255]
                    if len(displacement) == 1:
                        print(displacement)
                        continue
                    imend = cv2.line(
                        imend, pt, pt + displacement, color, thickness=2
                    )

                showimage("startframe", startframe)
                showimage("lastframe", imend)
                cv2.waitKey(1)
                track_writer.close()
        except AssertionError as e:
            print(f"error on dataset load, continuing")

    if not args.ontestingset:
        print(f"TOTALS:")
        errors_control_avg = errors_control_avg / data_used_count
        errordict = {}
        for model, avg in errors_avg.items():
            errorname = f"mean_{errortype}_{model}"
            error = avg / data_used_count
            errordict[errorname] = error
            print(f"{errorname}_combined: {error}")
        errorlists['total'] = errordict
        with open(f'results/{errortype}{num_data_name}{modeltype}{args.jsonsuffix}.json', 'a') as fp:
            json.dump(errorlists, fp,indent = 4,sort_keys=True)
    with open(f'results/positions_{num_data_name}{modeltype}{args.jsonsuffix}.json', 'a') as fp:
        json.dump(positionlists, fp, cls=NumpyEncoder)
