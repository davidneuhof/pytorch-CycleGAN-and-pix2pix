import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

def resize_with_alpha(im, load_size, all_ones=False):
    import numpy as np
    im_RGB = im.convert('RGB')
    im_alpha = im.split()[-1]
    alpha = np.array(im_alpha) < 127
    if all_ones:
        alpha[...] = True
    alpha = (alpha*255).astype(np.uint8)
    im_alpha = Image.fromarray(alpha)
    im_RGB = im_RGB.resize((load_size, load_size), Image.BICUBIC)
    im_alpha = im_alpha.resize((load_size, load_size), Image.NEAREST)
    im_RGB.putalpha(im_alpha)
    return im_RGB


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        if self.opt.output_nc == 4 or self.opt.input_nc == 4:
            AB = Image.open(AB_path).convert('RGBA')
        else:
            AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        if len(AB.getbands()) == 4:  # alpha channel
            A = AB.crop((0, 0, w2, h))
            A = resize_with_alpha(A, self.opt.loadSize, all_ones=True)
            B = AB.crop((w2, 0, w, h))
            B = resize_with_alpha(B, self.opt.loadSize)
        else:
            A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        if self.opt.use_mask_for_L1:
            import numpy as np
            im_mask = Image.open(AB_path.replace('combined', 'mask')).convert('L')
            mask = np.array(im_mask) >= 127
            mask = (mask*255).astype(np.uint8)
            im_mask = Image.fromarray(mask)
            im_mask = im_mask.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
            M = transforms.ToTensor()(im_mask)
            M = M[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]


        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            if self.opt.use_mask_for_L1:
                M = M.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if self.opt.use_mask_for_L1:
            return {'A': A, 'B': B, 'M': M,
                    'A_paths': AB_path, 'B_paths': AB_path}        
        else:
            return {'A': A, 'B': B,
                    'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
