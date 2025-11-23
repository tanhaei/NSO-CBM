import torch
import torch.nn as nn

class PerceptualEncoder(nn.Module):
    """
    Stage 1: Maps raw multimodal data to intermediate clinical concepts.
    """
    def __init__(self, num_static_features):
        super(PerceptualEncoder, self).__init__()
        
        # --- Branch 1: Vision (OCT/Fundus Images) ---
        # Input: (Batch, 1, 224, 224) -> Output: Structural Concepts
        # Concepts: CDR (num), Notch (bin), RNFL Defect (bin)
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.vis_head = nn.Linear(128 * 4 * 4, 3) # Outputs raw logits for 3 vision concepts

        # --- Branch 2: Temporal (IOP Series) ---
        # Input: (Batch, Seq_Len, 1) -> Output: Hemodynamic Concepts
        # Concept: Current IOP State (num)
        self.temporal_lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.temp_head = nn.Linear(32, 1)

        # --- Branch 3: Static (Metadata/History) ---
        # Input: (Batch, Num_Features) -> Output: Risk Concepts
        # Concept: Family History (bin)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, img, iop_series, static_data):
        # Vision Forward
        vis_feat = self.vision_backbone(img)
        c_vis = self.vis_head(vis_feat)
        
        # Temporal Forward
        # Take the last hidden state
        _, (h_n, _) = self.temporal_lstm(iop_series)
        c_temp = self.temp_head(h_n[-1])
        
        # Static Forward
        c_stat = self.static_mlp(static_data)
        
        return c_vis, c_temp, c_stat

class ReasoningHead(nn.Module):
    """
    Stage 2: Maps Concepts to Final Diagnosis.
    Kept linear for interpretability (W = Concept Importance).
    """
    def __init__(self, num_concepts, num_classes):
        super(ReasoningHead, self).__init__()
        self.layer = nn.Linear(num_concepts, num_classes)
    
    def forward(self, concepts):
        return self.layer(concepts)

class TemporalMultimodalCBM(nn.Module):
    def __init__(self, num_static_features, num_classes=2):
        super(TemporalMultimodalCBM, self).__init__()
        self.encoder = PerceptualEncoder(num_static_features)
        # Total concepts = 3 (Vision) + 1 (Temporal) + 1 (Static) = 5
        self.reasoner = ReasoningHead(num_concepts=5, num_classes=num_classes)
        
    def forward(self, img, iop_series, static_data):
        # 1. Get Raw Concept Logits
        c_vis, c_temp, c_stat = self.encoder(img, iop_series, static_data)
        
        # 2. Process Concepts for the Bottleneck
        # CDR (0-1 bound)
        c_cdr = torch.sigmoid(c_vis[:, 0:1])
        # Notch & RNFL (Probabilities)
        c_notch = torch.sigmoid(c_vis[:, 1:2])
        c_rnfl = torch.sigmoid(c_vis[:, 2:3])
        # IOP (Raw value, regression)
        c_iop = c_temp 
        # Family History (Probability)
        c_fam = torch.sigmoid(c_stat)
        
        # Concatenate to form the Concept Bottleneck Vector
        bottleneck = torch.cat([c_cdr, c_notch, c_rnfl, c_iop, c_fam], dim=1)
        
        # 3. Final Prediction
        logits = self.reasoner(bottleneck)
        
        # Return logits AND dictionary of raw concepts for loss calculation
        concepts_dict = {
            'c_cdr': c_cdr,
            'c_notch': c_vis[:, 1:2], # Logits for BCEWithLogitsLoss
            'c_rnfl': c_vis[:, 2:3],  # Logits
            'c_iop': c_iop,
            'c_fam': c_stat           # Logits
        }
        
        return logits, concepts_dict
