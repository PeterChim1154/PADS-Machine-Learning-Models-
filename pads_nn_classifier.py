import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

# Fix random seeds for perfect reproducibility across runs
torch.manual_seed(42)
np.random.seed(42)

# ==========================================================
# 1. DEFINE THE FEED-FORWARD NEURAL NETWORK ARCHITECTURE
# ==========================================================
class PADSFeedForwardNN(nn.Module):
    def __init__(self, input_dim):
        super(PADSFeedForwardNN, self).__init__()
        
        # Simple, robust architecture to prevent overfitting on small tabular samples
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Regularization to prevent memorization
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(16, 1)    # Single output neuron for binary classification (0 or 1)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================================
# 2. TRAINING AND EVALUATION ENGINE
# ==========================================================
def train_and_evaluate_nn(X, y, groups, modality_name, epochs=60, batch_size=16):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    b_acc_list, f1_list, prec_list, rec_list = [], [], [], []
    
    # Standardize data to numpy matrices
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    for train_idx, test_idx in sgkf.split(X_arr, y_arr, groups=groups):
        # Data Splitting & Scaling
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Calculate class weights dynamically to fight class imbalance (just like SVM's class_weight='balanced')
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32)
        
        # Convert data structures to PyTorch Tensors
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize NN Model, Loss, and Optimizer
        model = PADSFeedForwardNN(input_dim=X.shape[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Binary Cross Entropy with built-in Logits & Weighting
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluation Phase (Inference)
        model.eval()
        with torch.no_grad():
            raw_logits = model(test_x_tensor)
            # Apply Sigmoid mathematically: positive logits (>0) mean probability > 0.5 (Class 1)
            y_pred = (raw_logits >= 0.0).cpu().numpy().astype(int).flatten()
            
        # Record evaluation metrics for this fold
        b_acc_list.append(balanced_accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))
        prec_list.append(precision_score(y_test, y_pred, zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, zero_division=0))
        
    return {
        "Modality": modality_name,
        "Balanced Acc": f"{np.mean(b_acc_list):.4f} (±{np.std(b_acc_list):.3f})",
        "F1-Score": f"{np.mean(f1_list):.4f} (±{np.std(f1_list):.3f})",
        "Precision": f"{np.mean(prec_list):.4f} (±{np.std(prec_list):.3f})",
        "Recall": f"{np.mean(rec_list):.4f} (±{np.std(rec_list):.3f})"
    }

def run_nn_task(df_input, task_name, positive_class, negative_class):
    task_df = df_input[df_input['condition'].isin([positive_class, negative_class])].copy()
    if task_df.empty:
        return

    groups = task_df['participant_id']
    y = task_df['condition'].apply(lambda x: 1 if x == positive_class else 0)
    
    # Filter specific modalities
    all_features = [c for c in task_df.columns if c.startswith('acc_') or c.startswith('gyr_')]
    acc_features = [c for c in task_df.columns if c.startswith('acc_')]
    gyr_features = [c for c in task_df.columns if c.startswith('gyr_')]
    
    results = []
    results.append(train_and_evaluate_nn(task_df[all_features], y, groups, "Accel + Gyro (Combined)"))
    results.append(train_and_evaluate_nn(task_df[acc_features], y, groups, "Accel Only (Linear)"))
    results.append(train_and_evaluate_nn(task_df[gyr_features], y, groups, "Gyro Only (Rotation)"))
    
    print(f"\n==============================================================================================")
    print(f" NEURAL NETWORK TASK: {task_name}")
    print(f" Samples: {task_df.shape[0]} | Patients: {groups.nunique()} | ({negative_class}=0, {positive_class}=1)")
    print(f"==============================================================================================")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print(f"==============================================================================================\n")

# ==========================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================
FEATURES_PATH = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'

try:
    df = pd.read_csv(FEATURES_PATH)
    df['condition'] = df['condition'].astype(str).str.upper().str.strip()
    
    # Task A: NN Evaluation
    run_nn_task(df, "Parkinson's (PD) vs. Healthy Control (HC)", 
                positive_class="PARKINSON'S", negative_class="HEALTHY")
    
    # Task B: NN Evaluation
    dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM']
    df_with_dd = df.copy()
    df_with_dd['condition'] = df_with_dd['condition'].apply(lambda x: 'DD' if x in dd_conditions else x)
    
    run_nn_task(df_with_dd, "Parkinson's (PD) vs. Differential Diagnosis (DD)", 
                positive_class="PARKINSON'S", negative_class="DD")

except FileNotFoundError:
    print(f"Error: Could not find feature file at {FEATURES_PATH}.")