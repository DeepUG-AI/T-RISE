import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class isic_loader(Dataset):
    def __init__(self, path_Data, train=True, Test=False):
        super().__init__()

        if train:
            self.split_name = 'train'
            self.data_dir = os.path.join(path_Data, 'train')
        elif Test:
            self.split_name = 'test'
            self.data_dir = os.path.join(path_Data, 'test')
        else:
            self.split_name = 'val'
            self.data_dir = os.path.join(path_Data, 'val')

        self.img_path = os.path.join(self.data_dir, 'imgs')
        self.mask_path = os.path.join(self.data_dir, 'mask')

        print(f'\n[INFO] Loading {self.split_name} dataset')
        print(f'[INFO] Image path: {self.img_path}')
        print(f'[INFO] Mask path: {self.mask_path}')

        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f'Image path does not exist: {self.img_path}')
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Mask path does not exist: {self.mask_path}')

        self.file_list = [f for f in os.listdir(self.img_path) if f.lower().endswith('.png')]
        self.file_list.sort()

        self.mask_list = [f for f in os.listdir(self.mask_path) if f.lower().endswith('.png')]
        self.mask_list.sort()

        print(f'[INFO] Number of images in {self.split_name}: {len(self.file_list)}')
        print(f'[INFO] Number of masks in {self.split_name}: {len(self.mask_list)}')

        expected_masks = set([f.replace('.png', '_mask.png') for f in self.file_list])
        actual_masks = set(self.mask_list)

        missing_masks = expected_masks - actual_masks
        extra_masks = actual_masks - expected_masks

        if len(missing_masks) == 0 and len(extra_masks) == 0:
            print(f'[INFO] {self.split_name} data matched correctly')
        else:
            print(f'[WARNING] {self.split_name} data are not fully matched')
            if len(missing_masks) > 0:
                print(f'[WARNING] Number of missing masks: {len(missing_masks)}')
                print(f'[WARNING] Examples of missing masks: {list(missing_masks)[:5]}')
            if len(extra_masks) > 0:
                print(f'[WARNING] Number of extra masks: {len(extra_masks)}')
                print(f'[WARNING] Examples of extra masks: {list(extra_masks)[:5]}')


        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        print(f'[INFO] Whether img_transform exists: {hasattr(self, "img_transform")}')
        print(f'[INFO] Whether mask_transform exists: {hasattr(self, "mask_transform")}')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        if not hasattr(self, 'img_transform'):
            raise AttributeError('self.img_transform does not exist, please check whether __init__ in loader.py was executed correctly')
        if not hasattr(self, 'mask_transform'):
            raise AttributeError('self.mask_transform does not exist, please check whether __init__ in loader.py was executed correctly')

        img_name = self.file_list[index]
        mask_name = img_name.replace('.png', '_mask.png')

        img_file = os.path.join(self.img_path, img_name)
        mask_file = os.path.join(self.mask_path, mask_name)

        if not os.path.exists(img_file):
            raise FileNotFoundError(f'Image does not exist: {img_file}')
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f'Mask does not exist: {mask_file}')

        image = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file).convert('L')

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask
