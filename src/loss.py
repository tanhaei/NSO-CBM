import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedJointLoss(nn.Module):
    def __init__(self, lambda_c=1.0):
        super(MaskedJointLoss, self).__init__()
        self.lambda_c = lambda_c
        self.task_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, concepts_pred, targets, concepts_true, masks):
        """
        predictions: Model output (Y_hat)
        concepts_pred: Dict of predicted concepts
        targets: True labels (Y)
        concepts_true: Dict of ground truth concepts (from BioArc)
        masks: Dict of binary masks (1=present, 0=missing)
        """
        
        # 1. Task Loss (Diagnosis)
        L_task = self.task_loss(predictions, targets)
        
        # 2. Concept Loss (with Masking)
        loss_c = 0
        total_valid_concepts = 0
        
        # Group A: Regression Concepts (MSE) -> IOP, CDR
        for key in ['c_cdr', 'c_iop']:
            # Compute element-wise MSE
            mse = F.mse_loss(concepts_pred[key], concepts_true[key], reduction='none')
            # Apply Mask: Zero out loss for missing data
            masked_mse = mse * masks[key]
            loss_c += masked_mse.sum()
            total_valid_concepts += masks[key].sum()
            
        # Group B: Classification Concepts (BCE) -> Notch, RNFL, Family
        for key in ['c_notch', 'c_rnfl', 'c_fam']:
            # Use BCEWithLogits for numerical stability
            bce = F.binary_cross_entropy_with_logits(
                concepts_pred[key], concepts_true[key], reduction='none'
            )
            masked_bce = bce * masks[key]
            loss_c += masked_bce.sum()
            total_valid_concepts += masks[key].sum()
            
        # Normalize Concept Loss
        # Avoid division by zero
        L_concept = loss_c / (total_valid_concepts + 1e-8)
        
        # 3. Total Loss
        L_total = L_task + (self.lambda_c * L_concept)
        
        return L_total, L_task, L_concept
