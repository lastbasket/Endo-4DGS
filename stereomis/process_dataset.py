import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from rectification import StereoRectifier
import torch
from glob import glob
import tqdm



def process_one(dir_name):
    data_dir = dir_name
    video  = glob(os.path.join(data_dir,'*.mp4'))[0]

    calib_file = os.path.join(data_dir, 'StereoCalibration.ini')
    capture = cv2.VideoCapture(video)

    rect = StereoRectifier(calib_file, img_size_new=None)
    calib = rect.get_rectified_calib()
    count = 0
    os.makedirs(os.path.join(data_dir, 'images_1'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images_right_1'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images_2'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images_right_2'), exist_ok=True)
    # os.makedirs(os.path.join(data_dir, 'disparity'), exist_ok=True)


    split_1 = [801, 1000]
    split_2 = [13901, 14100]
    

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if (count>=split_1[0]) and (count <= split_1[1]):
            split = 1
        elif (count>=split_2[0]) and (count <= split_2[1]):
            split = 2
        else:
            count += 1
            continue

        h, w, c = frame.shape
        right = torch.from_numpy(frame[0:h//2, :, :]).permute(2,0,1)
        left = torch.from_numpy(frame[h//2:, :, :]).permute(2,0,1)
        left_calib, right_calib = rect(left, right)
        count_tring = str(count).rjust(6, '0')
        left_path = os.path.join(data_dir, f'images_{split}/{count_tring}.png')
        right_path = os.path.join(data_dir, f'images_right_{split}/{count_tring}.png')
        # dis_path = os.path.join(data_dir, f'disparity/{count_tring}.png')

        left_calib = left_calib.permute(1,2,0).numpy()
        right_calib = right_calib.permute(1,2,0).numpy()

        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        # left_gray = cv2.cvtColor(left_calib, cv2.COLOR_BGR2GRAY)
        # right_gray = cv2.cvtColor(right_calib, cv2.COLOR_BGR2GRAY)
        # disparity = np.clip(stereo.compute(left_gray, right_gray), 0, 255)
        # cv2.imwrite(dis_path, disparity)
        cv2.imwrite(left_path, left_calib)
        cv2.imwrite(right_path, right_calib)
        count += 1
    
folder_list = sorted(glob(os.path.join('..', "P1")))

for f in tqdm.tqdm(folder_list):
    process_one(f)
