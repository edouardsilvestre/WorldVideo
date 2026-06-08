import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MovingMNISTDataset(Dataset):
    """
    Moving MNIST PyTorch Dataset.
    
    Loads a .npy file of shape (20, 10000, 64, 64) containing 10,000 sequences
    of 20 frames of 64x64 resolution (grayscale).
    
    Splits the 10,000 sequences into train and validation sets (80% train, 20% val).
    Returns (input_frames, target_frames), where each is a tensor of shape
    (10, 1, 64, 64) normalized in the range [0.0, 1.0].
    """
    def __init__(self, filepath, split="train", transform=None):
        """
        Args:
            filepath (str): Path to the mnist_test_seq.npy file.
            split (str): 'train' for the first 80%, 'val' for the remaining 20%,
                         or 'all' for the full dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
            
        self.filepath = filepath
        self.split = split.lower()
        self.transform = transform
        
        # Load Moving MNIST dataset: shape is (20, 10000, 64, 64)
        raw_data = np.load(self.filepath)
        
        # Transpose to (10000, 20, 64, 64) -> (num_sequences, seq_len, height, width)
        self.data = np.transpose(raw_data, (1, 0, 2, 3))
        
        # Split into train/validation/all
        num_samples = len(self.data)
        split_idx = int(num_samples * 0.8)  # 80% train
        
        if self.split == "train":
            self.data = self.data[:split_idx]
        elif self.split == "val":
            self.data = self.data[split_idx:]
        elif self.split == "all":
            pass
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'all'.")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Retrieve sequence of shape (20, 64, 64)
        seq = self.data[idx]
        
        # Convert to float32 and normalize to [0.0, 1.0]
        seq = seq.astype(np.float32) / 255.0
        
        # Convert to PyTorch Tensor
        seq_tensor = torch.from_numpy(seq)
        
        # Add channel dimension: (20, 64, 64) -> (20, 1, 64, 64)
        seq_tensor = seq_tensor.unsqueeze(1)
        
        # Split into input_frames (first 10) and target_frames (next 10)
        input_frames = seq_tensor[:10]
        target_frames = seq_tensor[10:]
        
        if self.transform:
            input_frames = self.transform(input_frames)
            target_frames = self.transform(target_frames)
            
        return input_frames, target_frames
