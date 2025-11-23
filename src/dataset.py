import torch
from torch.utils.data import Dataset
import json
import gzip
import numpy as np
from PIL import Image
import os

class BioArcDataset(Dataset):
    def __init__(self, data_path, img_dir, mode='train', transform=None):
        """
        Args:
            data_path (string): مسیر فایل فشرده (مثلاً 'data/bioarc_data.json.gz')
            img_dir (string): مسیر پوشه تصاویر OCT
            mode (string): 'train' یا 'test'
        """
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        print(f"Loading data from {data_path}...")
        
        # خواندن فایل GZ به صورت مستقیم
        # فرض بر این است که فایل حاوی لیستی از رکوردهاست
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            self.records = json.load(f)
            
        print(f"Loaded {len(self.records)} records.")

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # --- 1. Load Metadata/Static Data ---
        # نگاشت فیلدهای JSON به بردار ویژگی
        # این بخش باید بر اساس ساختار دقیق دیتای شما تنظیم شود
        static_features = []
        # مثال: استخراج سن و جنسیت
        age = record.get('1196', {}).get('Age', 0)
        gender = 1 if record.get('1196', {}).get('Sex') == 'Male' else 0
        static_features.extend([float(age), float(gender)])
        # ... سایر ویژگی‌های استاتیک را اینجا اضافه کنید ...
        
        static_data = torch.tensor(static_features, dtype=torch.float32)
        
        # --- 2. Load Time Series (IOP) ---
        # فرض: IOPها در یک لیست ذخیره شده‌اند
        iop_values = []
        # استخراج از ساختار پیچیده JSON شما (نیاز به لاجیک پارس کردن دارد)
        # اینجا یک نمونه ساده می‌گذارم:
        if '1046' in record and 'RightIOPsize' in record['1046']:
             val = record['1046']['RightIOPsize']
             iop_values.append(float(val) if val else 0.0)
        
        # پدینگ یا برش سری زمانی به طول ثابت (مثلاً ۱۰)
        seq_len = 10
        if len(iop_values) < seq_len:
            iop_values += [0.0] * (seq_len - len(iop_values))
        else:
            iop_values = iop_values[:seq_len]
            
        iop_series = torch.tensor(iop_values, dtype=torch.float32).unsqueeze(-1) # (Seq, 1)
        
        # --- 3. Load Image (OCT) ---
        # فرض: اسم فایل عکس در متادیتا هست
        img_name = record.get('img_filename', 'placeholder.png')
        img_path = os.path.join(self.img_dir, img_name)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('L') # L = Grayscale for OCT
            image = np.array(image)
        else:
            image = np.zeros((224, 224), dtype=np.uint8) # تصویر خالی اگر نبود
            
        # تبدیل به تنسور
        img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # --- 4. Ground Truth Concepts & Targets ---
        # استخراج مفاهیم از JSON برای آموزش مدل CBM
        
        # مثال استخراج
        cdr_val = float(record.get('1053', {}).get('RightOpticDiscCupDiscRatio', 0.5))
        has_glaucoma = 1 if "Glaucoma" in str(record.get('Diagnosis', '')) else 0
        
        concepts_true = {
            'c_cdr': torch.tensor([cdr_val], dtype=torch.float32),
            'c_iop': torch.tensor([np.mean(iop_values)], dtype=torch.float32),
            'c_notch': torch.tensor([0.0], dtype=torch.float32), # باید از JSON پر شود
            'c_rnfl': torch.tensor([0.0], dtype=torch.float32),  # باید از JSON پر شود
            'c_fam': torch.tensor([0.0], dtype=torch.float32)    # باید از JSON پر شود
        }
        
        # تولید ماسک (اگر دیتا در JSON نال بود، ماسک صفر شود)
        masks = {k: torch.tensor([1.0]) for k in concepts_true} 
        
        return {
            'img': img_tensor,
            'iop': iop_series,
            'static': static_data,
            'target': torch.tensor(has_glaucoma, dtype=torch.long),
            'concepts': concepts_true,
            'masks': masks
        }
