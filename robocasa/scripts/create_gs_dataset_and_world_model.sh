# Description: Script to create the Gaussian Splatting dataset from the RoboCASA demonstration data
date
start_time=$(date +%s)
start_path=$(pwd)

# 1. Convert the raw robosuite dataset to robomimic format
export ROBOCASA_DATA_PATH="/mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/colmap/r_0_l_0/"

cd /mnt/disk_1/guanxing/robomimic

echo "Converting robosuite dataset to robomimic format..."

python robomimic/scripts/conversion/convert_robosuite.py --dataset /mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/demo.hdf5

echo "Done converting robosuite dataset to robomimic format."
echo "Extracting image observations from robomimic dataset..."

# 2. Extract image observations from robomimic dataset (demo_im128.hdf5)
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robomimic/scripts/dataset_states_to_obs.py --dataset /mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/demo.hdf5 --num_procs 1 --camera_height 512 --camera_width 512 --skip_interval 5

cd /mnt/disk_1/guanxing/robocasa

echo "Done extracting image observations from robomimic dataset."




echo "Converting h5py demo data to Gaussian Splatting format..."

# 3. Convert h5py demo data to Gaussian Splatting format (default: navigate data)
python robocasa/scripts/convert_to_gaussian_format.py --dataset /mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/demo_im512.hdf5

echo "Done converting h5py demo data to Gaussian Splatting format."
echo "COLMAP..."

# 4. Colmap
python robocasa/scripts/convert.py -s robocasa/models/assets/demonstrations_private/take_a_walk/colmap/r_0_l_0/ --camera_mask_path /mnt/disk_1/guanxing/segment-anything-2/notebooks/mask_rev.png --camera SIMPLE_PINHOLE

echo "Done COLMAP."
echo "Creating per-image masks..."

# 5. Create per-image masks
cd $ROBOCASA_DATA_PATH; mkdir masks
for img in $ROBOCASA_DATA_PATH/input/*; do
  cp "/mnt/disk_1/guanxing/segment-anything-2/notebooks/mask_rev.png" "$ROBOCASA_DATA_PATH/masks/$(basename "$img")"
done

echo "Done creating per-image masks."
echo "Creating normals from pretrain..."

cd /mnt/disk_1/guanxing/dn-splatter

# 6. Normal priors
python dn_splatter/scripts/normals_from_pretrain.py --data-dir $ROBOCASA_DATA_PATH --model-type dsine

echo "Done creating normals from pretrain."
echo "Generating scale aligned mono-depth estimates..."

# 7. Generate scale aligned mono-depth estimates
python dn_splatter/scripts/align_depth.py --data $ROBOCASA_DATA_PATH \
       --no-skip-colmap-to-depths \
       --no-skip-mono-depth-creation

echo "Done generating scale aligned mono-depth estimates."

cd $start_path

end_time=$(date +%s)
echo "Total time: $((end_time-start_time)) seconds"

# eta: Total time: 1352 seconds