import os
import numpy as np
from scipy.spatial.transform import Rotation as R


focal_length = 1039.6275634765625
h = 1024
w = 1280

raw_data_root = '..'

data_dir = f'{raw_data_root}/StereoMIS_0_0_1/P1'
pose_file = 'groundtruth.txt'
start_index = 7801
img_range = 100
f = open(os.path.join(data_dir, pose_file), 'r')
data = f.read()
lines = data.replace(",", " ").replace("\t", " ").split("\n")
components = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
        len(line) > 0 and line[0] != "#"]

trans = np.asarray([l[1:4] for l in components if len(l) > 0], dtype=float)
trans *= 1000.0  # m to mm

# Quaternion
quat = np.asarray([l[4:] for l in components if len(l) > 0], dtype=float)

pose_list = []

for i in range(start_index, start_index+img_range):
    c_quat = quat[i]
    c_trans = trans[i]
    # c2w matrix in opencv format --> RDF (x:right, y:down, z:forward)
    rotation_matrix = R.from_quat(c_quat)
    rotation_matrix = rotation_matrix.as_matrix()
    # the rotation matrix is 3x3
    transformation_matrix = np.ones((4,5))
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = c_trans
    transformation_matrix[0, 4] = h / 2
    transformation_matrix[1, 4] = w / 2
    transformation_matrix[2, 4] = focal_length
    
    # 3x5 -> 15+2=17
    pose_matrix = transformation_matrix[0:3, :].flatten()
    
    # pose_matrix = np.append(pose_matrix, [depth.min(), depth.max()])
    pose_matrix = np.append(pose_matrix, [0, 200])
    
    pose_list.append(pose_matrix)


pose_list = np.array(pose_list) # [n, 17]
np.save(os.path.join(data_dir, 'poses_bounds.npy'), pose_list)