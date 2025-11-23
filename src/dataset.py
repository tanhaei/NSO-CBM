import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np
import os
from PIL import Image

class BioArcDataset(Dataset):
    def __init__(self, csv_path, img_dir=None, mode='train'):
        """
        Args:
            csv_path (string): Path to Farabi-EHR.csv
            img_dir (string): Path to OCT images folder (optional for now)
        """
        self.img_dir = img_dir
        self.mode = mode
        
        # Load CSV
        print(f"Loading clinical records from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Filter valid records (those with 'data' column)
        self.df = self.df.dropna(subset=['data'])
        print(f"Loaded {len(self.df)} valid patient records.")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse the JSON string in 'data' column
        try:
            record_data = json.loads(row['data'])
        except:
            record_data = {}

        # --- 1. Extract Concepts (The "Glaucoma 5") ---
        
        # C1: IOP (Intraocular Pressure) - Code 1046
        # We take the average of Right and Left if available
        iop_r = record_data.get('1046', {}).get('RightIOPsize', 0)
        iop_l = record_data.get('1103', {}).get('LeftIOPsize', 0) # Note: Left eye code might be 1103 based on data
        # Handle cases where IOP is None or string
        try: iop_val = max(float(iop_r or 0), float(iop_l or 0))
        except: iop_val = 0.0
        
        # C2: CDR (Cup-to-Disc Ratio) - Code 1053 (OD) / 1091 (OS)
        # Key: RightOpticDiscCupDiscRatioCup (Numeric value)
        cdr_r = record_data.get('1053', {}).get('RightOpticDiscCupDiscRatioCup', 0.0)
        cdr_l = record_data.get('1091', {}).get('LeftOpticDiscCupDiscRatioCup', 0.0)
        try: cdr_val = max(float(cdr_r or 0), float(cdr_l or 0))
        except: cdr_val = 0.0

        # C3: Family History - Code 1034
        # Key: FamilialHistoryGlaucoma inside nested list or dict
        fam_hist = 0.0
        hist_data = record_data.get('1034', {})
        if 'cfgc_1034_FamilialHistory' in record_data: # Check parsed root
             # Logic to parse list if exists
             pass
        # Simplified check for keyword in history
        if 'Glaucoma' in str(hist_data):
            fam_hist = 1.0

        # C4: Structural Damage (Notch/RNFL) - Code 1053
        # Checking for keywords like "RimThinning" or "NFLDefect"
        struct_damage = 0.0
        disc_data = str(record_data.get('1053', {}))
        if 'RimThinning' in disc_data or 'NFLDefect' in disc_data:
            struct_damage = 1.0

        # Construct Concepts Dictionary
        concepts_true = {
            'c_iop': torch.tensor([iop_val], dtype=torch.float32),
            'c_cdr': torch.tensor([cdr_val], dtype=torch.float32),
            'c_fam': torch.tensor([fam_hist], dtype=torch.float32),
            'c_struct': torch.tensor([struct_damage], dtype=torch.float32),
            # Add placeholders for others if missing in CSV
            'c_notch': torch.tensor([struct_damage], dtype=torch.float32), 
            'c_rnfl': torch.tensor([0.0], dtype=torch.float32) 
        }

        # --- 2. Create Masks ---
        # If value is 0 (and likely missing), we can set mask to 0 (optional)
        masks = {}
        for k, v in concepts_true.items():
            masks[k] = torch.tensor([1.0]) if v.item() > 0 else torch.tensor([0.0]) 
            # Note: Refine masking logic based on "completeness" column if available

        # --- 3. Generate Target (Diagnosis) ---
        # Heuristic: If Doctor Note contains "Glaucoma" or specific codes
        # In real training, use explicit 'diagnosis' column
        is_glaucoma = 0
        full_text = str(record_data).lower()
        if 'glaucoma' in full_text or 'poag' in full_text:
            is_glaucoma = 1
        elif iop_val > 21 or cdr_val > 0.6: # Clinical Rule Fallback
            is_glaucoma = 1
            
        target = torch.tensor(is_glaucoma, dtype=torch.long)

        # --- 4. Dummy Image/Series (Since CSV doesn't have them) ---
        img = torch.zeros(1, 224, 224) 
        iop_series = torch.zeros(10, 1) 

        return {
            'img': img,
            'iop': iop_series,
            'static': torch.zeros(10), # Placeholder
            'target': target,
            'concepts': concepts_true,
            'masks': masks
        }
