set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0

# python train.py --dataroot /home/ntuser/datasets/inpainting_datasets/slam2/combined/ \
#          --name slam_resize_conv_L1_mask --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned \
#          --no_lsgan --norm batch --pool_size 0 --display_server http://10.13.33.74 --niter=600 --niter_decay=200 \
#          --input_nc=3 --output_nc=3 --use_mask_for_L1
#         #   --continue_train

python train.py --dataroot /home/ntuser/datasets/inpainting_datasets/only_slam/combined/ \
         --name slam_only_resize_conv_L1_mask --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned \
         --no_lsgan --norm batch --pool_size 0 --display_server http://10.13.33.74 --niter=800 --niter_decay=200 \
         --input_nc=3 --output_nc=3 --use_mask_for_L1
        #   --continue_train