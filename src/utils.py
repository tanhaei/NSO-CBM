import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    """Loads model parameters (state_dict) from checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def calculate_metrics(predictions, targets, concepts_pred, concepts_true, masks):
    """
    Calculates comprehensive metrics for both Final Diagnosis and Intermediate Concepts.
    Useful for populating the 'Results' section of your paper.
    """
    metrics = {}
    
    # --- 1. Task Metrics (Glaucoma Diagnosis) ---
    # Convert logits to probabilities and classes
    probs = torch.softmax(predictions, dim=1)[:, 1] # Prob of class 1
    preds_cls = torch.argmax(predictions, dim=1)
    
    y_true = targets.cpu().detach().numpy()
    y_pred = preds_cls.cpu().detach().numpy()
    y_prob = probs.cpu().detach().numpy()
    
    metrics['task_acc'] = accuracy_score(y_true, y_pred)
    metrics['task_f1'] = f1_score(y_true, y_pred, average='macro')
    try:
        metrics['task_auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['task_auc'] = 0.0 # Handle single-class batch edge case

    # --- 2. Concept Metrics (Only on valid/masked data) ---
    
    # Continuous Concepts (IOP, CDR) -> RMSE
    for key in ['c_cdr', 'c_iop']:
        mask = masks[key].bool().cpu().numpy().flatten()
        if mask.sum() > 0:
            pred_val = concepts_pred[key].detach().cpu().numpy().flatten()[mask]
            true_val = concepts_true[key].detach().cpu().numpy().flatten()[mask]
            metrics[f'{key}_rmse'] = np.sqrt(mean_squared_error(true_val, pred_val))
        else:
            metrics[f'{key}_rmse'] = np.nan

    # Binary Concepts (Notch, RNFL, Family) -> Accuracy
    for key in ['c_notch', 'c_rnfl', 'c_fam']:
        mask = masks[key].bool().cpu().numpy().flatten()
        if mask.sum() > 0:
            # Apply sigmoid and threshold
            pred_prob = torch.sigmoid(concepts_pred[key]).detach().cpu().numpy().flatten()[mask]
            pred_bin = (pred_prob > 0.5).astype(int)
            true_bin = concepts_true[key].detach().cpu().numpy().flatten()[mask]
            metrics[f'{key}_acc'] = accuracy_score(true_bin, pred_bin)
        else:
            metrics[f'{key}_acc'] = np.nan
            
    return metrics

def calculate_intervention_efficacy(model, loader, device):
    """
    Simulates Doctor Intervention:
    1. Predicts Diagnosis based on Model's Concepts.
    2. Replaces ONE concept (e.g., IOP) with Ground Truth (simulating correction).
    3. Measures how much the Diagnosis changes (corrects).
    
    Returns: Intervention Efficacy Score (IES) for each concept.
    """
    model.eval()
    improvements = {k: [] for k in ['c_iop', 'c_cdr', 'c_notch', 'c_rnfl', 'c_fam']}
    
    print("Calculating Intervention Scores...")
    with torch.no_grad():
        for batch in loader:
            img = batch['img'].to(device)
            iop = batch['iop'].to(device)
            static = batch['static'].to(device)
            targets = batch['target'].to(device)
            
            # 1. Get Initial Prediction
            # We need to manually call encoder parts to access the bottleneck
            c_vis, c_temp, c_stat = model.encoder(img, iop, static)
            
            # Process raw concepts to bottleneck format
            c_cdr = torch.sigmoid(c_vis[:, 0:1])
            c_notch = torch.sigmoid(c_vis[:, 1:2])
            c_rnfl = torch.sigmoid(c_vis[:, 2:3])
            c_iop = c_temp
            c_fam = torch.sigmoid(c_stat)
            
            # Current Bottleneck
            bottleneck_original = torch.cat([c_cdr, c_notch, c_rnfl, c_iop, c_fam], dim=1)
            logits_orig = model.reasoner(bottleneck_original)
            preds_orig = torch.argmax(logits_orig, dim=1)
            
            # Accuracy BEFORE Intervention
            acc_orig = (preds_orig == targets).float()
            
            # 2. Perform Intervention (One concept at a time)
            # True concepts from batch
            c_true = {k: v.to(device) for k, v in batch['concepts'].items()}
            
            # Construct intervened bottlenecks
            concepts_map = {
                'c_cdr':   [c_true['c_cdr'], c_notch, c_rnfl, c_iop, c_fam],
                'c_notch': [c_cdr, c_true['c_notch'], c_rnfl, c_iop, c_fam], # Note: c_true['c_notch'] is binary, might need float cast
                'c_rnfl':  [c_cdr, c_notch, c_true['c_rnfl'], c_iop, c_fam],
                'c_iop':   [c_cdr, c_notch, c_rnfl, c_true['c_iop'], c_fam],
                'c_fam':   [c_cdr, c_notch, c_rnfl, c_iop, c_true['c_fam']]
            }
            
            for key, concept_list in concepts_map.items():
                # Create new bottleneck with ONE true concept injected
                bottleneck_new = torch.cat(concept_list, dim=1)
                logits_new = model.reasoner(bottleneck_new)
                preds_new = torch.argmax(logits_new, dim=1)
                
                # Accuracy AFTER Intervention
                acc_new = (preds_new == targets).float()
                
                # Improvement: Did fixing the concept fix the diagnosis?
                # 1 = Yes (Fixed), 0 = No change, -1 = Broke it
                delta = acc_new - acc_orig
                improvements[key].extend(delta.cpu().numpy())

    # Calculate average improvement
    ies_scores = {k: np.mean(v) for k, v in improvements.items()}
    return ies_scores
