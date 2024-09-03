'''
load numpy and visualize depth image

Usage:
    python robocasa/scripts/load_depth_numpy.py
'''

import numpy as np
import matplotlib.pyplot as plt

depth_path = "/mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/colmap/r_0_l_0/mono_depth/demo_demo_1_cam_robot0_eye_in_hand_image_id_235_aligned.npy"
depth = np.load(depth_path)
print(f"max: {np.max(depth)}, min: {np.min(depth)}")

# normalize depth
depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

# visualize depth
plt.imshow(depth)
plt.show()
