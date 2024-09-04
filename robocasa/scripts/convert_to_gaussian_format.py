'''
Gaussian world model stage 1

e.g., convert to:
datasets/v0.1/single_stage/kitchen_navigate/NavigateKitchen/2024-05-09/r_0_l_0/
    |---input
        |---<image 0>
        |---<image 1>
        |---...

Usage: python robocasa/scripts/convert_to_gaussian_format.py


If the camera poses are known and you want to reconstruct a sparse or dense model of the scene, you must first manually construct a sparse model by creating a cameras.txt, points3D.txt, and images.txt under a new folder:

+── path/to/manually/created/sparse/model
│   +── cameras.txt
│   +── images.txt
│   +── points3D.txt

The points3D.txt file should be empty while every other line in the images.txt should also be empty, since the sparse features are computed, as described below. You can refer to this article for more information about the structure of a sparse model.

Example of images.txt:

1 0.695104 0.718385 -0.024566 0.012285 -0.046895 0.005253 -0.199664 1 image0001.png
# Make sure every other line is left empty
2 0.696445 0.717090 -0.023185 0.014441 -0.041213 0.001928 -0.134851 2 image0002.png

3 0.697457 0.715925 -0.025383 0.018967 -0.054056 0.008579 -0.378221 1 image0003.png

4 0.698777 0.714625 -0.023996 0.021129 -0.048184 0.004529 -0.313427 2 image0004.png

Each image above must have the same image_id (first column) as in the database (next step). This database can be inspected either in the GUI (under Database management > Processing), or, one can create a reconstruction with colmap and later export it as text in order to see the images.txt file it creates.
'''

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import random
import time
from termcolor import colored, cprint

from robocasa.utils.dataset_registry import get_ds_path
from tqdm import trange
import shutil
import open3d as o3d

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    # mask = (depth <= 0.599/0.8)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

def convert_to_ply_with_colmap(args):
    if args.dataset is None:
        dataset = get_ds_path(args.task, ds_type="human_im")
    else:
        dataset = args.dataset
    cprint("Loading dataset: {}".format(dataset), "green")

    f = h5py.File(dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    elif "data" in f.keys():
        demos = list(f["data"].keys())
    else:
        demos = None

    save_root = None
    # for ind in trange(len(demos)):
    for ind in range(len(demos)):
        ep = demos[ind]
        # print(colored("Playing back episode: {}".format(ep), "yellow"))

        # <KeysViewHDF5 ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']>
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        # initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        # get room layout id
        initial_state_dict = json.loads(initial_state["ep_meta"].strip())
        rid = initial_state_dict['layout_id']   # total: 10
        sid = initial_state_dict['style_id']   # total: 12

        # for debugging (demos do not fully cover all room and layout)
        if rid == 0 and sid == 0:
            pass
        else:
            continue

        # set save folder
        # save_root = os.path.join(os.path.dirname(dataset), "colmap", "r_{}_l_{}".format(rid, sid), "input")
        save_root = os.path.join(os.path.dirname(dataset), "colmap", "r_{}_l_{}".format(rid, sid))
        # if exist, remove it
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root, exist_ok=True)

        # write image
        # debug: f["data/demo_0/obs/"].keys()

        # camera_name = "robot0_agentview_left_image"
        # camera_name = "robot0_eye_in_hand_image"
        camera_name = args.camera_name
        imgs = f["data/{}/obs/{}".format(ep, camera_name)]
        depths = f["data/{}/obs/{}".format(ep, camera_name.replace("image", "depth"))]

        Ks = f["data/{}/info/{}_K".format(ep, camera_name)]   # intrinsic
        Rs = f["data/{}/info/{}_R".format(ep, camera_name)]   # extrinsic
        znears = f["data/{}/info/{}_znear".format(ep, camera_name)]
        zfars = f["data/{}/info/{}_zfar".format(ep, camera_name)]

        # save_folder = os.path.join(save_root, "{}".format(camera_name), "{}".format(ep))
        save_folder = os.path.join(save_root, "input")
        os.makedirs(save_folder, exist_ok=True)
        cprint(f"Writing {imgs.shape[0]} images to {save_folder}", "yellow")

        depth_folder = save_folder.replace("input", "depth_debug")
        os.makedirs(depth_folder, exist_ok=True)
        cprint(f"Writing depth images to {depth_folder}", "yellow")

        sparse_folder = os.path.join(save_root, "sparse/0")
        os.makedirs(sparse_folder, exist_ok=True)
        images_file = os.path.join(sparse_folder, "images.txt")

        # mask
        if args.mask_path is not None:
            gripper_mask = imageio.imread(args.mask_path)
            # gripper_mask = np.flip(gripper_mask, axis=0)
            gripper_mask = gripper_mask.astype(bool)
        
        # print(gripper_mask)

        with open(images_file, 'a') as f2:
            f2.write("# Image list with two lines of data per image:\n")
            f2.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f2.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f2.write("# Number of images: {}, mean observations per image: {}\n".format(imgs.shape[0], imgs.shape[0]))

        # NOTE: visualize and save the camera poses and its orientations defined by rotation matrix
        # from camera_utils.camera_pose_visualizer import CameraPoseVisualizer
        # import matplotlib.pyplot as plt
        # from scipy.spatial.transform import Rotation as R
        # visualizer = CameraPoseVisualizer([-5, 5], [-5, 5], [0, 5])

        aggr_pcd = o3d.geometry.PointCloud()
        # for i in range(0, imgs.shape[0], args.skip_interval):
        for i in trange(0, imgs.shape[0], args.skip_interval):
            img = imgs[i]
            depth = np.squeeze(depths[i], axis=-1)

            # NOTE: IMPORTANT: flip image
            depth = np.flip(depth, axis=0)

            znear = znears[i]
            zfar = zfars[i]

            # Make sure that depth values are normalized
            assert np.all(depth >= 0.0) and np.all(depth <= 1.0)
            depth = znear / (1.0 - depth * (1.0 - znear / zfar))

            # Write image
            img_path = os.path.join(save_folder, "{}.png".format(i))
            imageio.imwrite(img_path, img)

            # Write depth image for debugging
            depth_path = os.path.join(depth_folder, "{}_depth.png".format(i))

            # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            # ax[0].imshow(img)
            # ax[0].set_title('Image')
            # ax[0].axis('off')  
            # depth_map_display = ax[1].imshow(depth, cmap='viridis')
            # ax[1].set_title('Depth Map')
            # ax[1].axis('off')  
            # fig.colorbar(depth_map_display, ax=ax[1], fraction=0.046, pad=0.04)
            # plt.savefig(depth_path)

            # write extrinsics to txt
            qw, qx, qy, qz, tx, ty, tz = parse_camera_extrinsic(Rs[i])
            with open(images_file, 'a') as f2:
                f2.write("{} {} {} {} {} {} {} {} {} {}\n".format(i+1, qw, qx, qy, qz, tx, ty, tz, 1, "{}.png".format(i)))
                f2.write("\n")

            # visualizer.extrinsic2pyramid(Rs[i], 'c', focal_len_scaled=1, aspect_ratio=0.3)

            # Compute point cloud
            color = img / 255.0

            K = Ks[i]
            R = Rs[i]
            cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy

            # mask = (depth > 0) & (depth < 1.5)
            if args.mask_path is not None:
                mask = (depth > 0) & (depth < 3.0) & gripper_mask
            else:
                mask = (depth > 0) & (depth < 3.0)
            pcd = depth2fgpcd(depth, mask, cam_param)
            
            # pose = np.linalg.inv(R)   # NOTE: wrong
            pose = R
            
            trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
            trans_pcd = trans_pcd[:3, :].T
            
            pcd_o3d = np2o3d(trans_pcd, color[mask])
            # downsample = False
            downsample = True
            if downsample:
                pcd_o3d = pcd_o3d.uniform_down_sample(every_k_points=5)    # 512*512*(ratio=0.05)=13107

            aggr_pcd += pcd_o3d
        
        if downsample:
            radius = 0.02
            aggr_pcd = aggr_pcd.voxel_down_sample(radius)

        # remove outlier
        aggr_pcd, ind = aggr_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        aggr_pcd, ind = aggr_pcd.remove_radius_outlier(nb_points=20, radius=0.1)

        # save pcd for visualization
        pcd_file = os.path.join(save_root, "sparse_pc.ply")
        o3d.io.write_point_cloud(pcd_file, aggr_pcd)
        # how much point we use
        point_num = len(np.asarray(aggr_pcd.points))
        cprint("Saving point cloud of {} points to: {}".format(point_num, pcd_file), "green")

        # show camera pose trajectory
        # visualizer.show()
        # plt.savefig(os.path.join(save_root, "camera_poses.png"))
        # cprint("Camera poses saved to: {}".format(os.path.join(save_root, "camera_poses.png")), "green")

    # Write camera parameters
    camera_model = 'SIMPLE_PINHOLE'
    width = img.shape[1]
    height = img.shape[0]
    fx, fy, cx, cy = parse_camera_intrinsic(Ks[0])
    
    cameras_file = os.path.join(sparse_folder, "cameras.txt")
    os.makedirs(os.path.dirname(cameras_file), exist_ok=True)
    with open(cameras_file, 'a') as f1:
        f1.write("# Camera list with one line of data per camera:\n")
        f1.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f1.write("# Number of cameras: 1\n")
        f1.write("{} {} {} {} {}\n".format(1, camera_model, width, height, ' '.join(map(str, [fx, cx, cy]))))

    # Write points3D.txt
    points3D_file = os.path.join(sparse_folder, "points3D.txt")
    # open(points3D_file, 'w').close()
    with open(points3D_file, 'a') as f3:
        f3.write("# 3D point list with one line of data per point:\n")
        f3.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f3.write("# Number of points: {}\n".format(point_num))

    for i in range(point_num):
        pcd = np.asarray(aggr_pcd.points)
        color = np.asarray(aggr_pcd.colors)
        with open(points3D_file, 'a') as f3:
            f3.write("{} {} {} {} {} {} {} {} {}\n".format(i+1, pcd[i, 0], pcd[i, 1], pcd[i, 2], img[i, 0], img[i, 1], img[i, 2], 0, ""))
            f3.write("\n")

    f.close()
    cprint("Saving to: {}".format(save_root), "green")

def parse_camera_intrinsic(K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy

def parse_camera_extrinsic(Rt):
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    # qx, qy, qz, qw = mat2quat(R)
    qw, qx, qy, qz = mat2quat(R)
    # qx1, qy1, qz1, qw1 = mat2quat_robosuite(R)
    # assert (qw, qx, qy, qz) == (qw1, qx1, qy1, qz1)
    tx, ty, tz = t
    return qw, qx, qy, qz, tx, ty, tz   # NOTE: we change the rotation order here

def mat2quat_robosuite(rmat):
    """
    ref: robosuite
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])   # x, y, z, w
    return q1[inds]

def mat2quat(rmat):
    """
    ref: 
    JPL: x, y, z, w
    ROS: w, x, y, z
    We return ROS format here
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    K = np.zeros((4, 4))
    
    K[0, 0] = (M[0, 0] - M[1, 1] - M[2, 2]) / 3.0
    K[1, 1] = (M[1, 1] - M[0, 0] - M[2, 2]) / 3.0
    K[2, 2] = (M[2, 2] - M[0, 0] - M[1, 1]) / 3.0
    K[3, 3] = (M[0, 0] + M[1, 1] + M[2, 2]) / 3.0
    
    K[0, 1] = K[1, 0] = (M[0, 1] + M[1, 0]) / 3.0
    K[0, 2] = K[2, 0] = (M[0, 2] + M[2, 0]) / 3.0
    K[0, 3] = K[3, 0] = (M[1, 2] - M[2, 1]) / 3.0
    K[1, 2] = K[2, 1] = (M[1, 2] + M[2, 1]) / 3.0
    K[1, 3] = K[3, 1] = (M[2, 0] - M[0, 2]) / 3.0
    K[2, 3] = K[3, 2] = (M[0, 1] - M[1, 0]) / 3.0
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    
    # Extract the largest eigenvector
    q = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Return in ROS format (w, x, y, z)
    return (q[3], q[0], q[1], q[2])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )
    parser.add_argument(
        "--skip_interval",
        type=int,
        default=1,
        help="sample step in an interval",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--camera_name",
        type=str,
        default="robot0_eye_in_hand_image",
        # default=["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action='store_true',
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        help="log additional information",
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="(optional) mask out the gripper",
    )

    parser.add_argument("--task", type=str, default="NavigateKitchen", help="task (choose among 100+ tasks)")

    args = parser.parse_args()
    convert_to_ply_with_colmap(args)
