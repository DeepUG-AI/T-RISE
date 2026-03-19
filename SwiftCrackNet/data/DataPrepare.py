import os
import glob
import numpy as np
from PIL import Image

# Parameters to resize images and masks to 512 x 512
height = 512
width = 512
channels = 3  # Number of image channels

def load_images_and_masks(image_dir, mask_dir, num_images):
    """
    Load images and masks from specified directories, resize them, and return as numpy arrays.
    """
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    Data = np.zeros([num_images, height, width, channels])
    Label = np.zeros([num_images, height, width])

    for idx, image_path in enumerate(image_files):
        if idx >= num_images:
            break

        print(f"Processing image: {image_path}")
        img = Image.open(image_path)
        img = img.resize((width, height), Image.BILINEAR).convert('RGB')
        img = np.array(img, dtype=np.float32)
        Data[idx] = img

        base_name = os.path.basename(image_path)
        mask_name = os.path.splitext(base_name)[0] + '_mask.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        print(f"Processing mask: {mask_path}")
        try:
            mask = Image.open(mask_path)
            mask = mask.resize((width, height), Image.BILINEAR).convert('L')
            mask = np.array(mask, dtype=np.float32)
            Label[idx] = mask
        except FileNotFoundError:
            print(f"File not found: {mask_path}")
            continue  # Skip this file if the mask is not found

    return Data, Label

# Directories for training, validation, and test sets
train_image_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\train\imgs"
train_mask_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\train\mask"
val_image_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\val\imgs"
val_mask_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\val\mask"
test_image_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\test\imgs"
test_mask_dir = r"H:\SwiftCrackNet\SwiftCrackNet-main\newdata2\test\mask"

# Number of images in each set
train_number = len(glob.glob(os.path.join(train_image_dir, "*.png")))
val_number = len(glob.glob(os.path.join(val_image_dir, "*.png")))
test_number = len(glob.glob(os.path.join(test_image_dir, "*.png")))

# Load datasets
print("Loading training set...")
Train_img, Train_mask = load_images_and_masks(train_image_dir, train_mask_dir, train_number)

print("Loading validation set...")
Validation_img, Validation_mask = load_images_and_masks(val_image_dir, val_mask_dir, val_number)

print("Loading test set...")
Test_img, Test_mask = load_images_and_masks(test_image_dir, test_mask_dir, test_number)

# Save datasets
np.save('data_train.npy', Train_img)
np.save('data_val.npy', Validation_img)
np.save('data_test.npy', Test_img)

np.save('mask_train.npy', Train_mask)
np.save('mask_val.npy', Validation_mask)
np.save('mask_test.npy', Test_mask)

print("Datasets saved successfully.")
