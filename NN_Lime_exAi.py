import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lime import lime_tabular  # Required for tabular LIME explainability
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# Ensure deterministic results for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class OptimizedPDClassifier(nn.Module):
    def __init__(self, input_dim):
        super(OptimizedPDClassifier, self).__init__()
        # 3 Hidden Layers (Input -> 32 -> 16 -> 8 -> 1)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),  
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Linear(16, 8),
            nn.ReLU(),
            
            nn.Linear(8, 1)     # Final Output Logit
        )
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate_nn(X, y, groups, modality_name, task_name):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    b_acc, f1, prec, rec = [], [], [], []
    
    X_mat = X.to_numpy()
    y_mat = y.to_numpy()
    groups_mat = groups.to_numpy()
    
    # Track variables from the final fold to feed into LIME
    last_model = None
    last_scaler = None
    
    for train_idx, test_idx in sgkf.split(X_mat, y_mat, groups=groups_mat):
        X_train, X_test = X_mat[train_idx], X_mat[test_idx]
        y_train, y_test = y_mat[train_idx], y_mat[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_t = torch.FloatTensor(X_train_scaled)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_t = torch.FloatTensor(X_test_scaled)
        
        model = OptimizedPDClassifier(input_dim=X_train.shape[1])
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
            predictions = (raw_outputs >= 0.0).int().numpy().flatten()
            
        b_acc.append(balanced_accuracy_score(y_test, predictions))
        f1.append(f1_score(y_test, predictions, zero_division=0))
        prec.append(precision_score(y_test, predictions, zero_division=0))
        rec.append(recall_score(y_test, predictions, zero_division=0))
        
        # Cache resources from final split loop execution
        last_model = model
        last_scaler = scaler

    # --- INTEGRATING LIME EXPLAINABLE AI ENGINE FOR NEURAL NETWORK ---
    # Only run LIME for the combined modality matrix to avoid duplicate graph output clutter
    if modality_name == "Accel + Gyro (Combined)":
        print(f" -> Initializing LIME Tabular Explainer specifically for the NN architecture ({task_name})...")
        
        # Define custom pipeline wrapper to handle scaling and logits -> sigmoid probability conversion
        def nn_predict_proba_pipeline(numpy_array_input):
            last_model.eval()
            with torch.no_grad():
                # Scale perturbations using weights learned on the training fold
                scaled_input = last_scaler.transform(numpy_array_input)
                tensor_input = torch.FloatTensor(scaled_input)
                logits = last_model(tensor_input)
                probabilities = torch.sigmoid(logits).numpy().flatten()
                
            # Construct standard 2D class array layout required by LIME
            proba_2d = np.zeros((numpy_array_input.shape[0], 2))
            proba_2d[:, 0] = 1.0 - probabilities  # Column 0: Baseline Probability
            proba_2d[:, 1] = probabilities        # Column 1: Target PD Probability
            return proba_2d

        # SPECIFICATION UPDATE: Explicit class names identifying NN target mappings
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_mat,
            feature_names=X.columns.tolist(),
            class_names=['Non-PD (NN Baseline)', 'Parkinson\'s (NN Target)'],
            mode='classification',
            random_state=42
        )
        
        # Target evaluation instance index (Sample 0 from our testing vector)
        target_sample_idx = 0
        raw_target_instance = X_mat[test_idx][target_sample_idx]
        
        exp = explainer.explain_instance(
            data_row=raw_target_instance,
            predict_fn=nn_predict_proba_pipeline,
            num_features=10
        )
        
        # Render and Save LIME Visual Explanation
        os.makedirs("figs", exist_ok=True)
        sanitized_task = task_name.replace("'", "").replace(".", "").replace(" ", "_").replace("(", "").replace(")", "").lower()
        fig = exp.as_pyplot_figure()
        
        # SPECIFICATION UPDATE: Clearly denoting the Neural Network model on plot title
        plt.title(f"NN Model LIME Feature Explanations: {task_name}\n(Local 3-Layer Neural Network Weights Evaluation)", fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        
        fig_path = f"figs/nn_lime_{sanitized_task}_patient_breakdown.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f" 🚀 Success! Saved local NN-specific LIME explanation figure to: {fig_path}")

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
    results.append(train_and_evaluate_nn(task_df[all_features], y, groups, "Accel + Gyro (Combined)", task_name))
    results.append(train_and_evaluate_nn(task_df[acc_features], y, groups, "Accel Only (Linear)", task_name))
    results.append(train_and_evaluate_nn(task_df[gyr_features], y, groups, "Gyro Only (Rotation)", task_name))
    
    # Print clean results table
    print(f"\n==============================================================================================")
    print(f" TASK: {task_name} (EXPLAINABLE AI MULTILAYER NEURAL NETWORK)")
    print(f" Samples: {task_df.shape[0]} | Patients: {groups.nunique()} | ({negative_class}=0, {positive_class}=1)")
    print(f"==============================================================================================")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print(f"==============================================================================================\n")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    FEATURES_PATH = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'

    try:
        df = pd.read_csv(FEATURES_PATH)
        df['condition'] = df['condition'].astype(str).str.upper().str.strip()
        
        # Run Experiment 1: PD vs HC
        run_comprehensive_nn_analysis(df, "PD vs. HC", 
                                      positive_class="PARKINSON'S", negative_class="HEALTHY")
        
        # Run Experiment 2: PD vs DD
        dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM']
        df_with_dd = df.copy()
        df_with_dd['condition'] = df_with_dd['condition'].apply(lambda x: 'DD' if x in dd_conditions else x)
        
        run_comprehensive_nn_analysis(df_with_dd, "PD vs. DD", 
                                      positive_class="PARKINSON'S", negative_class="DD")

    except FileNotFoundError:
        print(f"Error: Could not find feature file at {FEATURES_PATH}.")