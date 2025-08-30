import nibabel as nib
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from scipy.ndimage import label, find_objects
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes
from datetime import datetime



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# specify the directory containing nifti files
#training_data_directory = '/home/lij6263/AutoSegProject/training_data'
#validation_data_directory = '/home/lij6263/AutoSegProject/validation_data'
training_data_directory_img = 'hippo/train_img'
training_data_directory_mask = 'hippo/train_mask'

validation_data_directory_img = 'hippo/val_img'
validation_data_directory_mask = 'hippo/val_mask'
# list to store loaded nifti images and masks
training_images = []
training_masks = []
validation_images = []
validation_masks = []

# loop through all files in the training data directory
for filename in sorted(os.listdir(training_data_directory_img)):
    # if filename ends with CTI, convert nifti to torch tensor and append to images list
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(training_data_directory_img, filename)
        img = nib.load(image_path).get_fdata()


        img_tensor = torch.tensor(img, dtype=torch.float32)
        training_images.append(img_tensor)
    # if filename ends with CTM, convert nifti to torch tensor and append to masks list
for filename in sorted(os.listdir(training_data_directory_mask)):
    if filename.endswith('.nii.gz'):
        mask_path = os.path.join(training_data_directory_mask, filename)
        mask = nib.load(mask_path).get_fdata()


        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        training_masks.append(mask_tensor)


# loop through all files in the validation data directory
for filename in sorted(os.listdir(validation_data_directory_img)):
    # if filename ends with CTI, convert nifti to torch tensor and append to images list
    if filename.endswith('.nii.gz'):
        image_path = os.path.join(validation_data_directory_img, filename)
        img = nib.load(image_path).get_fdata()
                

        img_tensor = torch.tensor(img, dtype=torch.float32)
        validation_images.append(img_tensor)
for filename in sorted(os.listdir(validation_data_directory_mask)):
    # if filename ends with CTM, convert nifti to torch tensor and append to masks list
    if filename.endswith('.nii.gz'):
        mask_path = os.path.join(validation_data_directory_mask, filename)
        mask = nib.load(mask_path).get_fdata()
        #print(f"Unique values in validation mask {filename}: {np.unique(mask)}")
        #print(f"Shape of validation mask {filename}: {mask.shape}")

        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        validation_masks.append(mask_tensor)

print(f"Training Images: {len(training_images)}")
print(f"Training Masks: {len(training_masks)}")
print(f"Validation Images: {len(validation_images)}")
print(f"Validation Masks: {len(validation_masks)}")


class TrainingDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return image, mask
    
# Create dataset
trainingDataset = TrainingDataset(training_images, training_masks)
validationDataset = TrainingDataset(validation_images, validation_masks)



trainloader = DataLoader(trainingDataset,shuffle=True)
validationloader = DataLoader(validationDataset,shuffle=True)

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
        # Remove sigmoid for CrossEntropyLoss - it expects raw logits
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
    
class LearnableBCEDiceLoss(nn.Module):
    def __init__(self):
        super(LearnableBCEDiceLoss, self).__init__()
        # Initialize the learnable weight parameter (starting with equal weighting)
        self.bce_weight = nn.Parameter(torch.tensor(0.5))  # Initial value of 0.5

        # BCE loss definition
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        # Clamp the learnable weight to ensure it stays within [0, 1]
        bce_weight = torch.sigmoid(self.bce_weight)
        
        # Binary Cross-Entropy Loss
        bce = self.bce_loss(pred, target)
        
        # Dice Loss
        dice = dice_loss(pred, target)
        
        # Combined loss with learnable weight
        return bce_weight * bce + (1 - bce_weight) * dice

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# define model
model = UNet3D(in_channels=1, out_channels=3)
#device = torch.device('cuda')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
# define loss function and optimizer


#criterion = LearnableBCEDiceLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)

# define variables for early stopping
best_loss = float('inf')
patience = 10  # number of epochs to wait before stopping if no improvement
trigger_times = 0

begin = datetime.now()
fulltime = begin.strftime("%m_%d_%Y_%H_%M_%S")
print("Starting at "+fulltime,flush=True)

# Proceed with the training loop as before
for epoch in range(100):
    training_loss = 0.0
    validation_loss = 0.0

    # Training dataset
    for image_batch, mask_batch in trainloader:
        image_batch = image_batch.unsqueeze(1).to(device)
        # For CrossEntropyLoss, target should be class indices (0, 1, 2) as Long tensor
        # and should NOT have a channel dimension
        mask_batch = mask_batch.to(device).long()

        # Forward pass
        outputs = model(image_batch)
        loss = criterion(outputs, mask_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    # validation dataset
    for image_batch, mask_batch in validationloader:
        image_batch = image_batch.unsqueeze(1).to(device)
        # For CrossEntropyLoss, target should be class indices (0, 1, 2) as Long tensor
        # and should NOT have a channel dimension
        mask_batch = mask_batch.to(device).long()

        # Forward pass
        outputs = model(image_batch)
        loss = criterion(outputs, mask_batch)
        validation_loss += loss.item()

    # Access the learned weight (optional for monitoring)
    #current_weight = torch.sigmoid(criterion.bce_weight).item()
    current = datetime.now()
    elapsed = current - begin
    print(f"Epoch [{epoch+1}], Training loss: {training_loss:.4f}, validation loss: {validation_loss:.4f}",flush=True)
    print("Elapsed time: " + str(elapsed),flush=True)
    #print(f"Current BCE weight: {current_weight:.4f}",flush=True)
    print("Best loss: " + str(best_loss),flush=True)

    # Early stopping and saving logic remains the same
    if validation_loss < best_loss:
        print("Validation loss decreased, saving model...",flush=True)
        print("New validation loss " + str(validation_loss)+" was lower than "+str(best_loss),flush=True)

        best_loss = validation_loss
        trigger_times = 0
        # save the best model if new lowest loss is reached
        torch.save(model, 'classic_'+fulltime+".pth")
    else:
        trigger_times += 1
        print(f"Trigger Times: {trigger_times}")

        if trigger_times >= patience:
            print("Early stopping triggered")
            break
