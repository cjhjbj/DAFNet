import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from PIL import ImageFile

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)
        self.pair_style = opt.pair_style

    def __getitem__(self, index):
        if self.opt.isTrain:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
            A_path = self.A_paths[index_A]
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform_A(A_img)
            B_path = self.B_paths[index_B]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)
            name_A = os.path.basename(A_path)
            name_B = os.path.basename(B_path)
            name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
            return {'c': A, 's': B, 'name': name}

        if self.pair_style == 0:
            index_A = index // self.B_size
            index_B = index % self.B_size
            A_path = self.A_paths[index_A]
            A_img = Image.open(A_path).convert('RGB')
            A = self.transform_A(A_img)
            B_path = self.B_paths[index_B]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)
            name_A = os.path.basename(A_path)
            name_B = os.path.basename(B_path)
            name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
            return {'c': A, 's': B, 'name': name}

        else:
            style_pairs = max(1, self.B_size - 1)
            index_A = index // style_pairs
            pair_idx = index % style_pairs
            B1_path = self.B_paths[pair_idx % self.B_size]
            B2_path = self.B_paths[(pair_idx + 1) % self.B_size]
            A_path = self.A_paths[index_A % self.A_size]

            A = self.transform_A(Image.open(A_path).convert('RGB'))
            s1 = self.transform_B(Image.open(B1_path).convert('RGB'))
            s2 = self.transform_B(Image.open(B2_path).convert('RGB'))


            name_A = os.path.basename(A_path)
            name_B1 = os.path.basename(B1_path)
            name_B2 = os.path.basename(B2_path)


            name_A_noext = name_A[:name_A.rfind('.')]
            name_B1_noext = name_B1[:name_B1.rfind('.')]
            name_B2_noext = name_B2[:name_B2.rfind('.')]
            ext = name_A[name_A.rfind('.'):]


            final_name = f"{name_B1_noext}_{name_B2_noext}_{name_A_noext}{ext}"


            return {'c': A, 's1': s1, 's2': s2,'name': final_name}

    def __len__(self):
        if self.opt.isTrain:
            return self.A_size
        if self.pair_style == 1:
            return min(self.A_size * max(1, self.B_size - 1), self.opt.num_test)
        else:
            return min(self.A_size * self.B_size, self.opt.num_test)