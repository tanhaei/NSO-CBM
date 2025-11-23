# Neuro-Symbolic Ophthalmology: A Temporal-Multimodal Concept Bottleneck Framework

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Medical AI](https://img.shields.io/badge/Medical_AI-BioArc-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

This repository contains the official PyTorch implementation for the paper: **"Neuro-Symbolic Ophthalmology: A Temporal-Multimodal Concept Bottleneck Framework for Interpretable Glaucoma Progression Prediction"**.

Our framework (TM-CBM) bridges the gap between deep learning precision and clinical interpretability by introducing a **Concept Bottleneck** layer aligned with ophthalmic ontology.

---

## 🧠 Core Features

- **Multimodal Fusion:** Integrates Structural (OCT), Temporal (IOP Series), and Static (Demographics) data.
- **Concept Bottleneck:** Explicitly predicts clinical biomarkers (e.g., `CDR`, `Notch`, `IOP_Trend`) before diagnosis.
- **Intervenability:** Allows clinicians to manually correct intermediate concepts during inference to rectify model predictions.
- **Masked Joint Loss:** Handles missing clinical data (common in real-world EHRs like BioArc) effectively.

---

## 📂 Repository Structure

```text
root/
├── data/                   # Dataset placeholders (BioArc samples)
├── src/
│   ├── __init__.py
│   ├── model.py            # TM-CBM Architecture (Perceptual Encoders + Reasoner)
│   ├── loss.py             # Masked Joint Loss Function
│   ├── dataset.py          # BioArc Data Loader & Mask Generation
│   └── utils.py            # Metrics, Checkpointing, and Intervention Scoring
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Installation
Clone the repository:

```bash
git clone [https://github.com/your-username/neuro-symbolic-ophthalmology.git](https://github.com/your-username/neuro-symbolic-ophthalmology.git)
cd neuro-symbolic-ophthalmology
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Training the Model
To train the model with the joint optimization objective (Task Loss + Concept Loss), run:

```bash
python train.py
```

*Note: You can adjust hyperparameters like `BATCH_SIZE`, `LR`, and `lambda_c` (concept weight) inside `train.py`.*


### 2. Simulating Clinical Intervention
To evaluate the Intervention Efficacy Score (IES)—i.e., how much the model responds to doctor's corrections—the training script automatically runs an evaluation at the end of the last epoch.

Alternatively, use the utils function:

```bash
from src.utils import calculate_intervention_efficacy
# ... load model ...
ies = calculate_intervention_efficacy(model, test_loader, device)
print(ies)
```


## 📊 Data Format (BioArc)

The model expects data in the following format (handled by `src/dataset.py`):

| Modality | Dimensions | Description |
| :--- | :--- | :--- |
| **OCT Image** | `(1, 224, 224)` | Retinal structural scans |
| **IOP Series** | `(Seq_Len, 1)` | Longitudinal Intraocular Pressure |
| **Metadata** | `(Num_Feats,)` | Demographics & History |

**Concepts Dictionary:**
The dataset must return a dictionary of ground truth concepts (derived from BioArc JSONs):
- `c_cdr` (0-1)
- `c_iop` (mmHg)
- `c_notch` (0/1)
- `c_rnfl` (0/1)
- `c_fam` (0/1)


### 🧬 BioArc Data Structure Overview

```mermaid
---
config:
  look: handDrawn
  theme: base
---
classDiagram
    classDef root fill:#ff7675,stroke:#2d3436,stroke-width:3px,color:white,font-weight:bold;
    classDef structural fill:#74b9ff,stroke:#2d3436,stroke-width:2px,color:white;
    classDef detail fill:#55efc4,stroke:#2d3436,stroke-width:2px,color:#2d3436;
    
    %% کلاس اصلی
    class PatientRecord:::root {
        +int PatientID
        +Demographics Demographics
        +History History
        +Examinations Exams
    }

    %% کلاس‌های میانی
    class Demographics:::structural {
        +string NationalCode
        +int Age
        +string Gender
    }

    class History:::structural {
        +string MedicalHistory
        +string FamilyHistory
    }

    class Examinations:::structural {
        +IOP_Data RightEye
        +IOP_Data LeftEye
        +OCT_Data Structural
    }

    %% کلاس‌های جزئیات
    class IOP_Data:::detail {
        +float IOP_mmHg
        +float CupDiscRatio
        +boolean RimThinning
    }
    
    class OCT_Data:::detail {
        +float RNFL_Thickness
        +string Analysis
    }

    %% روابط
    PatientRecord *-- Demographics
    PatientRecord *-- History
    PatientRecord *-- Examinations
    Examinations *-- IOP_Data
    Examinations *-- OCT_Data
```

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
