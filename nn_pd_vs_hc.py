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

class PDvsHCNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PDvsHCNeuralNetwork, self).__init__()
        # Architecture: 3 Hidden Layers (Input -> 32 -> 16 -> 8 -> 1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1) # Output Logit
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
        
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test_scaled)
        
        model = PDvsHCNeuralNetwork(input_dim=X_train.shape[1])
        
        pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight]))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
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
                
        model.eval()
        with torch.no_grad():
            raw_outputs = model(X_test_t)
            predictions = (raw_outputs >= 0.0).int().numpy().flatten() #threshold line to force the continuous number into a binary choice 
            
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

FEATURES_PATH = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'

try:
    df = pd.read_csv(FEATURES_PATH)
    df['condition'] = df['condition'].astype(str).str.upper().str.strip()
    
    # Filter STRICTLY for Parkinson's vs Healthy
    task_df = df[df['condition'].isin(["PARKINSON'S", "HEALTHY"])].copy()
    
    groups = task_df['participant_id']
    y = task_df['condition'].apply(lambda x: 1 if x == "PARKINSON'S" else 0)
    
    all_features = [c for c in task_df.columns if c.startswith('acc_') or c.startswith('gyr_')]
    acc_features = [c for c in task_df.columns if c.startswith('acc_')]
    gyr_features = [c for c in task_df.columns if c.startswith('gyr_')]
    
    results = []
    results.append(train_and_evaluate_nn(task_df[all_features], y, groups, "Accel + Gyro (Combined)"))
    results.append(train_and_evaluate_nn(task_df[acc_features], y, groups, "Accel Only (Linear)"))
    results.append(train_and_evaluate_nn(task_df[gyr_features], y, groups, "Gyro Only (Rotation)"))
    
    print(f"\n==============================================================================================")
    print(f" TASK: Parkinson's (PD) vs. Healthy Control (HC) (FEED-FORWARD NEURAL NETWORK)")
    print(f" Samples: {task_df.shape[0]} | Patients: {groups.nunique()} | (HEALTHY=0, PARKINSON'S=1)")
    print(f"==============================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"==============================================================================================\n")

except FileNotFoundError:
    print(f"Error: Could not find feature file at {FEATURES_PATH}.")