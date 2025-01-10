import torch
import open3d as o3d
import numpy as np


vizualizer = o3d.visualization.Visualizer()
vizualizer.create_window(width=800, height=800)

camera_traj = []

def load_extrinsics(pose_path):
    ''' return: 
    '''
    extrinsics = []
    with open(pose_path, "r") as f:
        lines = f.readlines() # Skip the header
    for i in range(len(lines)):
        line = lines[i]
        pose = list(map(float, line.split(sep=',')))
        pose = torch.Tensor(pose).reshape(4, 4).float().transpose(0, 1)
        extrinsic = np.linalg.inv(pose.detach().cpu().numpy())
        extrinsics.append(extrinsic)
    return extrinsics

def load_pose(file_path):
    poses = np.load(file_path)[:, :15]
    poses = poses.reshape(-1, 3, 5)[:, :3, :4]
    last_line = np.tile(np.array([[0, 0, 0, 1]])[None], (poses.shape[0], 1, 1))
    poses = np.concatenate((poses, last_line), axis=1)

    return poses

width=1280
height=1024
fx = 693.2049560546875
fy = 502.83544921875
cx = 1040.2305908203125
cy = 1040.0146484375

intrin = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])

poses = load_pose('../StereoMIS_0_0_1/P1/poses_bounds.npy')
extrinsics = []

for i, pose in enumerate(poses):
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx = fx, 
        fy = fy,
        cx = cx,
        cy = cy
    )

    camera = o3d.camera.PinholeCameraParameters()
    extrin = np.linalg.inv(pose)
    
    camera.extrinsic = extrin 
    extrinsics.append(extrin)
    camera.intrinsic = intrinsic
    camera_traj.append(camera)

extrinsics = np.stack(extrinsics, axis=0)

o3d_intrinsics = []
o3d_extrinsics = []
count = 0

for i, cam_o3d in enumerate(camera_traj):
    if (i+1)%3 != 0:
        count += 1/len(camera_traj)
        continue
    o3d_intrinsics.append(cam_o3d.intrinsic.intrinsic_matrix)
    o3d_extrinsics.append(cam_o3d.extrinsic)
    xyz = np.asarray(np.linalg.inv(cam_o3d.extrinsic)[:3, 3])
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=675, 
                                                            view_height_px=540, 
                                                            intrinsic=intrin, 
                                                            extrinsic=extrinsics[i])
    
    cameraLines.paint_uniform_color(np.array([0.5, count, count]))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(\
        size=2, origin=[0, 0, 0])
    mesh_frame.transform(np.linalg.inv(cam_o3d.extrinsic))
    vizualizer.add_geometry(mesh_frame)
    
    count += 1/len(camera_traj)
    vizualizer.add_geometry(cameraLines)
    
# The x, y, z axis will be rendered as red, green, and blue arrows respectively.
# FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30, origin=[0, 0, 0])
# vizualizer.add_geometry(FOR1)
vizualizer.run()
vizualizer.destroy_window()