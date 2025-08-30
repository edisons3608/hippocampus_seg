import nibabel as nib
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
import scipy.io as io
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from scipy.ndimage import label, find_objects
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes
from scipy.ndimage import binary_erosion



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestingDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return image, mask
def dice_score(pred, target):
    # round predictions to a binary image using the 0.5 threshold
    pred = pred.float()
    # prevent divide by 0 error
    eps=1e-6

    # compute intersection and union
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    # compute Dice score
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.item()

def multiclass_dice_score(pred, target, num_classes):
    pred = pred.view(-1)
    target = target.view(-1)
    dice_scores = []
    eps = 1e-6
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    return dice_scores


def get_surface_voxels(mask):
    """Identifies surface voxels using erosion"""
    # Convert to boolean array first
    bool_mask = mask.astype(bool)
    eroded = binary_erosion(bool_mask, structure=np.ones((3, 3, 3)))
    surface_mask = bool_mask & ~eroded
    return np.argwhere(surface_mask)

# Calculate Hausdorff Distance and Average Symmetric Surface Distance with 3D scaling in physical space 

# Example: spacing = (2.5, 2.5, 3.0) for RL/AP/IS axes

def hausdorff_distance(pred_mask, true_mask, spacing):
    """Returns tuple: (Hausdorff in voxels, Hausdorff in mm)"""
    pred_surface = get_surface_voxels(pred_mask.cpu().numpy())
    true_surface = get_surface_voxels(true_mask.cpu().numpy())

    if len(pred_surface) == 0 or len(true_surface) == 0:
        return (float('inf'), float('inf'))  # Handle empty surfaces

    # Compute in voxel space (no scaling)
    d_pred_to_true_vox = cdist(pred_surface, true_surface, 'euclidean').min(axis=1)
    d_true_to_pred_vox = cdist(true_surface, pred_surface, 'euclidean').min(axis=1)
    hd_vox = max(np.max(d_pred_to_true_vox), np.max(d_true_to_pred_vox))

    # Compute in physical space (scale by spacing)
    pred_scaled = pred_surface * np.array(spacing)
    true_scaled = true_surface * np.array(spacing)
    d_pred_to_true_mm = cdist(pred_scaled, true_scaled, 'euclidean').min(axis=1)
    d_true_to_pred_mm = cdist(true_scaled, pred_scaled, 'euclidean').min(axis=1)
    hd_mm = max(np.max(d_pred_to_true_mm), np.max(d_true_to_pred_mm))

    return (hd_vox, hd_mm)

def assd(pred_mask, true_mask, spacing):
    """Returns tuple: (ASSD in voxels, ASSD in mm)"""
    pred_surface = get_surface_voxels(pred_mask.cpu().numpy())
    true_surface = get_surface_voxels(true_mask.cpu().numpy())

    if len(pred_surface) == 0 or len(true_surface) == 0:
        return (float('inf'), float('inf'))

    # Compute in voxel space
    d_pred_to_true_vox = cdist(pred_surface, true_surface, 'euclidean').min(axis=1)
    d_true_to_pred_vox = cdist(true_surface, pred_surface, 'euclidean').min(axis=1)
    assd_vox = (np.sum(d_pred_to_true_vox) + np.sum(d_true_to_pred_vox)) / (len(pred_surface) + len(true_surface))

    # Compute in physical space
    pred_scaled = pred_surface * np.array(spacing)
    true_scaled = true_surface * np.array(spacing)
    d_pred_to_true_mm = cdist(pred_scaled, true_scaled, 'euclidean').min(axis=1)
    d_true_to_pred_mm = cdist(true_scaled, pred_scaled, 'euclidean').min(axis=1)
    assd_mm = (np.sum(d_pred_to_true_mm) + np.sum(d_true_to_pred_mm)) / (len(pred_surface) + len(true_surface))

    return (assd_vox, assd_mm)

def keep_largest_island(mask):
    # Label connected components in the binary mask
    labeled_mask, num_features = label(mask.cpu().numpy().astype(np.uint8))

    # If no features are found, return the original mask
    if num_features == 0:
        return mask

    # Calculate the size of each component
    component_sizes = np.bincount(labeled_mask.ravel())
    largest_component = np.argmax(component_sizes[1:]) + 1  # +1 because labels start from 1

    # Create a new mask for the largest component
    largest_mask = np.zeros_like(mask)
    largest_mask[labeled_mask == largest_component] = 1

    return largest_mask

testing_data_directory_img = 'hippo/test_img'
testing_data_directory_mask = 'hippo/test_mask'

testing_images = []
testing_masks = []




        
for filename in sorted(os.listdir(testing_data_directory_img)):
    # if filename ends with CTI, convert nifti to torch tensor and append to images list
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(testing_data_directory_img, filename)
        img = nib.load(image_path).get_fdata()
                
        #img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        testing_images.append(img_tensor)
for filename in sorted(os.listdir(testing_data_directory_mask)):
    # if filename ends with CTM, convert nifti to torch tensor and append to masks list
    if filename.endswith('.nii.gz'):
        mask_path = os.path.join(testing_data_directory_mask, filename)
        mask = nib.load(mask_path).get_fdata()
        #print(f"Unique values in validation mask {filename}: {np.unique(mask)}")
        #print(f"Shape of validation mask {filename}: {mask.shape}")

        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        testing_masks.append(mask_tensor)



testingDataset = TestingDataset(testing_images, testing_masks)
testloader = DataLoader(testingDataset, shuffle=False)  # Important: shuffle=False to match filenames

print(f"Testing Images: {len(testing_images)}")


print(f"Testing Dataset: {len(testingDataset)}")
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.down1 = DoubleConv3D(in_channels, 16)   
        self.down2 = DoubleConv3D(16, 32)            
        self.down3 = DoubleConv3D(32, 64)
        self.down4 = DoubleConv3D(64, 128)
        self.maxpool = nn.MaxPool3d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up1 = DoubleConv3D(128 + 64, 64)      
        self.up2 = DoubleConv3D(64 + 32, 32)  
        self.up3 = DoubleConv3D(32 + 16, 16)
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        
        # sigmoid is required since we are using BCE loss, which only expects values 0-1. 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(f"input: {x.shape}")
        x1 = self.down1(x)
        #print(f"double conv: {x1.shape}")
        x2 = self.down2(self.maxpool(x1))
        #print(f"maxpool 1 + double conv: {x2.shape}")
        x3 = self.down3(self.maxpool(x2))
        #print(f"maxpool 2 + double conv: {x3.shape}")
        x4 = self.down4(self.maxpool(x3))
        #print(f"maxpool 3 + double conv: {x4.shape}")

        # Upsample and concatenate with skip connection
        x = self.upsample(x4)
        #print(f"upsampling: {x.shape}")
        x_padded = pad_tensor(x, x3)  # Pad x to match the size of x3
        #print(f"upsampling padded: {x_padded.shape}")
        x = torch.cat([x_padded, x3], dim=1)
        #print(f"concatenated: {x.shape}")
        x = self.up1(x)
        #print(f"upsampling conv 1: {x.shape}")
        
        x = self.upsample(x)
        #print(f"upsampling: {x.shape}")
        x_padded = pad_tensor(x, x2)  # Pad x to match the size of x2
        #print(f"upsampling padded: {x_padded.shape}")
        x = torch.cat([x_padded, x2], dim=1)
        #print(f"concatenated: {x.shape}")
        x = self.up2(x)
        #print(f"upsampling conv 2: {x.shape}")

        x = self.upsample(x)
        #print(f"upsampling: {x.shape}")
        x_padded = pad_tensor(x, x1)  # Pad x to match the size of x1
        #print(f"upsampling padded: {x_padded.shape}")
        x = torch.cat([x_padded, x1], dim=1)
        #print(f"concatenated: {x.shape}")
        x = self.up3(x)
        #print(f"upsampling conv 3: {x.shape}")

        # Final convolution to map to output channels
        x = self.final_conv(x)
        #print(f"final: {x.shape}")
        #x = self.sigmoid(x)
        return x


def pad_tensor(tensor, target_tensor):
    """Pad `tensor` to match the dimensions of `target_tensor`."""
    diff_depth = target_tensor.size(2) - tensor.size(2)
    diff_height = target_tensor.size(3) - tensor.size(3)
    diff_width = target_tensor.size(4) - tensor.size(4)

    # Apply padding
    return F.pad(tensor, (diff_width // 2, diff_width - diff_width // 2,
                          diff_height // 2, diff_height - diff_height // 2,
                          diff_depth // 2, diff_depth - diff_depth // 2))


class DoubleConv3D(nn.Module):
    """convolution => [BN] => ReLU => convolution => [BN] => ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
loaded_model = torch.load('classic_08_30_2025_13_25_37.pth', map_location=torch.device('cpu'),weights_only=False)

dice_scores = []
num_classes = 3  # adjust if needed
d1 = []
d2 = []
d3 = []

for image_batch, mask_batch in testloader:
    image_batch = image_batch.unsqueeze(1).to(device)         # [B, 1, D, H, W]
    mask_batch = mask_batch.to(device).long()                      # [B, D, H, W]
    print(f"Image Batch: {image_batch.shape}")
    print(f"Mask Batch: {mask_batch.shape}")
    outputs = loaded_model(image_batch)                       # [B, C, D, H, W]
    print(f"Outputs: {outputs.shape}")
    pred_mask = torch.argmax(outputs, dim=1)                  # [B, D, H, W]
    true_mask = mask_batch
    
    #spacing = [float(s.item()) for s in voxel_size[0][0]]
    
    dice_per_class = []
    #hd_per_class = []
    #assd_per_class = []

    for cls in range(0, num_classes):
        pred_cls = (pred_mask[0] == cls)
        target_cls = (true_mask[0] == cls)

        dice_cls = dice_score(pred_cls, target_cls)
        dice_per_class.append(dice_cls)

        if cls != 0:
            #hd_vox, hd_mm = hausdorff_distance(pred_cls, target_cls, spacing)
            #assd_vox, assd_mm = assd(pred_cls, target_cls, spacing)
            #hd_per_class.append(hd_mm)
            #assd_per_class.append(assd_mm)
            print(f"[{cls}] Dice: {dice_cls:.4f}")
            pass
        else:
            #hd_per_class.append(np.nan)
            #assd_per_class.append(np.nan)
            print(f"[{cls}] Dice: {dice_cls:.4f} (background)")

    print(f"Dice Scores: {dice_per_class}")
    #dice_scores.append(dice_per_class)
    d1.append(dice_per_class[0])
    d2.append(dice_per_class[1])
    d3.append(dice_per_class[2])
    #hausdorff_distances.append(hd_per_class)
    #assd_scores.append(assd_per_class)
# This code is meant to print the summaries to be copied into Excel. Currently not functional. 

ds_percentiles = np.percentile(d1, [25, 50, 75])
print(f"{round(ds_percentiles[0], 3)}")
print(f"{round(ds_percentiles[1], 3)}")
print(f"{round(ds_percentiles[2], 3)}")

ds_percentiles = np.percentile(d2, [25, 50, 75])
print(f"{round(ds_percentiles[0], 3)}")
print(f"{round(ds_percentiles[1], 3)}")
print(f"{round(ds_percentiles[2], 3)}")

ds_percentiles = np.percentile(d3, [25, 50, 75])
print(f"{round(ds_percentiles[0], 3)}")
print(f"{round(ds_percentiles[1], 3)}")
print(f"{round(ds_percentiles[2], 3)}")