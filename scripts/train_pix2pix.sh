set -ex
# python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0

python train.py --dataroot /home/ntuser/datasets/inpainting_datasets/slam2/combined/ \
         --name inpainting_resize_conv --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode aligned \
         --no_lsgan --norm batch --pool_size 0 --display_server http://10.13.33.74 --niter=600 --niter_decay=200 \
         --input_nc=3 --output_nc=3
        #   --continue_train
        # --use_mask_for_L1
