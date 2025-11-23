import torch
from torch.utils.data import Dataset
import numpy as np

class BioArcDataset(Dataset):
    def __init__(self, num_samples=1000, mode='train'):
        self.num_samples = num_samples
        self.mode = mode
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # --- Synthetic Data Generation (Replace with actual BioArc JSON parsing) ---
        
        # 1. Image (OCT slice): 1 channel, 224x224
        img = torch.randn(1, 224, 224)
        
        # 2. Time Series (IOP): Sequence length 10
        iop_series = torch.randn(10, 1)
        
        # 3. Static Data (Demographics): 10 features
        static_data = torch.randn(10)
        
        # 4. Target (Glaucoma): 0 or 1
        target = torch.randint(0, 2, (1,)).squeeze() # scalar
        
        # --- Ground Truth Concepts (from BioArc Metadata) ---
        # Note: Some might be missing (represented by mask=0)
        
        concepts_true = {
            'c_cdr': torch.rand(1),       # 0-1
            'c_notch': torch.rand(1),     # Binary (float for BCE)
            'c_rnfl': torch.rand(1),      # Binary
            'c_iop': torch.randn(1) + 15, # IOP value ~15
            'c_fam': torch.rand(1)        # Binary
        }
        
        # --- Masks (Simulating Missing Data) ---
        # Randomly drop concepts (simulating real-world sparsity)
        masks = {}
        for k in concepts_true.keys():
            # 80% chance data exists, 20% missing
            masks[k] = torch.bernoulli(torch.tensor(0.8))
            
        return {
            'img': img,
            'iop': iop_series,
            'static': static_data,
            'target': target,
            'concepts': concepts_true,
            'masks': masks
        }
