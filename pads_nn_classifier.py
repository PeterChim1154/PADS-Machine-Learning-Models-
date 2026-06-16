import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

# Ensure deterministic results for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class OptimizedPDClassifier(nn.Module):
    def __init__(self, input_dim):
        super(OptimizedPDClassifier, self).__init__()
        # NEW CONFIGURATION: 3 Hidden Layers (Input -> 32 -> 16 -> 8 -> 1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Prevents overfitting in wider layers
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)     # Final Output Logit
        )
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate_nn(X, y, groups, modality_name):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    b_acc, f1, prec, rec = [], [], [], []
    
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    groups_mat = groups.to_numpy()
    
    for train_idx, test_idx in sgkf.split(X_mat, y_mat, groups=groups_mat):
        X_train, X_test = X_mat[train_idx], X_mat[test_idx]
        y_train, y_test = y_mat[train_idx], y_mat[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert data structures to PyTorch Tensors
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test_scaled)
        
        # Initialize our deeper architecture
        model = OptimizedPDClassifier(input_dim=X_train.shape[1])
        
        # Calculate positive class weight to handle slight class imbalances perfectly
        pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))
        
        # OPTIMIZED HYPERPARAMETERS: lr=0.001 (down from 0.005) for highly precise tuning
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # OPTIMIZED HYPERPARAMETERS: 150 Epochs (up from 60) paired with a tight batch size
        model.train()
        dataset_size = len(X_train_t)
        batch_size = 16
        
        for epoch in range(150):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            raw_outputs = model(X_test_t)
            # Apply sigmoid thresholding (logits >= 0.0 implies probability >= 0.5)
            predictions = (raw_outputs >= 0.0).int().numpy().flatten()
            
        b_acc.append(balanced_accuracy_score(y_test, predictions))
        f1.append(f1_score(y_test, predictions, zero_division=0))
        prec.append(precision_score(y_test, predictions, zero_division=0))
        rec.append(recall_score(y_test, predictions, zero_division=0))
        
    return {
        "Modality": modality_name,
        "Balanced Acc": f"{np.mean(b_acc):.4f} (±{np.std(b_acc):.3f})",
        "F1-Score": f"{np.mean(f1):.4f} (±{np.std(f1):.3f})",
        "Precision": f"{np.mean(prec):.4f} (±{np.std(prec):.3f})",
        "Recall": f"{np.mean(rec):.4f} (±{np.std(rec):.3f})"
    }

def run_comprehensive_nn_analysis(df_input, task_name, positive_class, negative_class):
    task_df = df_input[df_input['condition'].isin([positive_class, negative_class])].copy()
    if task_df.empty:
        return

    groups = task_df['participant_id']
    y = task_df['condition'].apply(lambda x: 1 if x == positive_class else 0)
    
    all_features = [c for c in task_df.columns if c.startswith('acc_') or c.startswith('gyr_')]
    acc_features = [c for c in task_df.columns if c.startswith('acc_')]
    gyr_features = [c for c in task_df.columns if c.startswith('gyr_')]
    
    results = []
    results.append(train_and_evaluate_nn(task_df[all_features], y, groups, "Accel + Gyro (Combined)"))
    results.append(train_and_evaluate_nn(task_df[acc_features], y, groups, "Accel Only (Linear)"))
    results.append(train_and_evaluate_nn(task_df[gyr_features], y, groups, "Gyro Only (Rotation)"))
    
    print(f"\n==============================================================================================")
    print(f" TASK: {task_name} (OPTIMIZED HYPERPARAMETER NEURAL NETWORK)")
    print(f" Samples: {task_df.shape[0]} | Patients: {groups.nunique()} | ({negative_class}=0, {positive_class}=1)")
    print(f"==============================================================================================")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print(f"==============================================================================================\n")

FEATURES_PATH = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'

try:
    df = pd.read_csv(FEATURES_PATH)
    df['condition'] = df['condition'].astype(str).str.upper().str.strip()
    
    run_comprehensive_nn_analysis(df, "Parkinson's (PD) vs. Healthy Control (HC)", 
                                  positive_class="PARKINSON'S", negative_class="HEALTHY")
    
    dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM']
    df_with_dd = df.copy()
    df_with_dd['condition'] = df_with_dd['condition'].apply(lambda x: 'DD' if x in dd_conditions else x)
    
    run_comprehensive_nn_analysis(df_with_dd, "Parkinson's (PD) vs. Differential Diagnosis (DD)", 
                                  positive_class="PARKINSON'S", negative_class="DD")

except FileNotFoundError:
    print(f"Error: Could not find feature file at {FEATURES_PATH}.")