set -ex
# python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch

python test.py --dataroot /home/ntuser/datasets/inpainting_datasets/slam2/combined/ \
         --name inpainting_dataset_4c2 --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
