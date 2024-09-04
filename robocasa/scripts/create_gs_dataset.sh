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

# 2. Extract image observations from robomimic dataset (demo_im512.hdf5)
echo "Extracting image observations from robomimic dataset..."

OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python robomimic/scripts/dataset_states_to_obs.py --dataset /mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/demo.hdf5 --num_procs 1 --camera_height 512 --camera_width 512 --skip_interval 5

cd /mnt/disk_1/guanxing/robocasa

echo "Done extracting image observations from robomimic dataset."

# 3. Convert h5py demo data to Gaussian Splatting format (default: navigate data)
echo "Converting h5py demo data to Gaussian Splatting format..."

python robocasa/scripts/convert_to_gaussian_format.py --dataset /mnt/disk_1/guanxing/robocasa/robocasa/models/assets/demonstrations_private/take_a_walk/demo_im512.hdf5 --skip_interval 1 --mask_path /mnt/disk_1/guanxing/segment-anything-2/notebooks/r_0_l_0_mask_rev.png

echo "Done converting h5py demo data to Gaussian Splatting format."

# 4. Colmap (do not contain transfroms.json)
echo "COLMAP..."

# python robocasa/scripts/convert.py -s robocasa/models/assets/demonstrations_private/take_a_walk/colmap/r_0_l_0/ \
#   --camera_mask_path /mnt/disk_1/guanxing/segment-anything-2/notebooks/r_0_l_0_mask_rev.png \
#   --camera SIMPLE_PINHOLE \
#   --input_path sparse/0

# export ROBOCASA_DATA_PATH=robocasa/models/assets/demonstrations_private/take_a_walk/colmap/r_0_l_0
# generate db

cp -r $ROBOCASA_DATA_PATH/input $ROBOCASA_DATA_PATH/images 

colmap feature_extractor \
        --database_path $ROBOCASA_DATA_PATH/database.db \
        --image_path $ROBOCASA_DATA_PATH/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_PINHOLE \
        --SiftExtraction.use_gpu 1 \
        --ImageReader.camera_mask_path /mnt/disk_1/guanxing/segment-anything-2/notebooks/r_0_l_0_mask_rev.png

colmap exhaustive_matcher \
        --database_path $ROBOCASA_DATA_PATH/database.db \
        --SiftMatching.use_gpu 1

colmap point_triangulator \
    --database_path $ROBOCASA_DATA_PATH/database.db \
    --image_path $ROBOCASA_DATA_PATH/images \
    --input_path $ROBOCASA_DATA_PATH/sparse/0 \
    --output_path $ROBOCASA_DATA_PATH/sparse/0

# dense reconstruction (cored dumped)
# colmap image_undistorter \
#     --image_path $ROBOCASA_DATA_PATH/images \
#     --input_path $ROBOCASA_DATA_PATH/sparse/0 \
#     --output_path $ROBOCASA_DATA_PATH/
  
echo "Done COLMAP."

# [OR] Move the input/ folder to images/ folder
# cp -r $ROBOCASA_DATA_PATH/input $ROBOCASA_DATA_PATH/images

# 5. Create per-image masks
echo "Creating per-image masks..."

cd $ROBOCASA_DATA_PATH; mkdir masks
for img in $ROBOCASA_DATA_PATH/input/*; do
  cp "/mnt/disk_1/guanxing/segment-anything-2/notebooks/r_0_l_0_mask_rev.png" "$ROBOCASA_DATA_PATH/masks/$(basename "$img")"
done

echo "Done creating per-image masks."

# 6. Generate scale aligned mono-depth estimates (need camera info and COLMAP points)
echo "Generating scale aligned mono-depth estimates..."
cd /mnt/disk_1/guanxing/dn-splatter

python dn_splatter/scripts/align_depth.py --data $ROBOCASA_DATA_PATH \
       --no-skip-colmap-to-depths \
       --no-skip-mono-depth-creation

echo "Done generating scale aligned mono-depth estimates."

# 7. Normal priors (do not need camera info or COLMAP points)
echo "Creating normals from pretrain..."

cd /mnt/disk_1/guanxing/dn-splatter
python dn_splatter/scripts/normals_from_pretrain.py --data-dir $ROBOCASA_DATA_PATH --model-type dsine

echo "Done creating normals from pretrain."

# 8. Gaussian Splatting
echo "Gaussian Splatting..."

# This will generate 'sparse_pc.ply' for debugging

# CUDA_VISIBLE_DEVICES=2 ns-train dn-splatter --data $ROBOCASA_DATA_PATH \
#                  --pipeline.model.use-depth-loss True \
#                  --pipeline.model.sensor-depth-lambda 0.2 \
#                  --pipeline.model.use-depth-smooth-loss True \
#                  --pipeline.model.use-normal-loss True \
#                  --pipeline.model.normal-supervision mono \
#                  --pipeline.model.background_color white \
#                  coolermap --normal-format opencv --normals-from pretrained --load_normals True --masks_path masks

# no normal or depth loss
CUDA_VISIBLE_DEVICES=3 ns-train dn-splatter --data $ROBOCASA_DATA_PATH \
                 --pipeline.model.use-depth-loss False \
                 --pipeline.model.sensor-depth-lambda 0.0 \
                 --pipeline.model.use-depth-smooth-loss False \
                 --pipeline.model.use-normal-loss False \
                 --pipeline.model.background_color white \
                 coolermap --masks_path masks
# if no point cloud (generated by COLMAP), use  --load_pcd_normals False --load_3D_points False

echo "Done Gaussian Splatting."

cd $start_path

end_time=$(date +%s)
echo "Total time: $((end_time-start_time)) seconds"

# eta: Total time: 1352 seconds