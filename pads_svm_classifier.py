import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def evaluate_modality(X, y, groups, modality_name):
    """Helper to run 5-Fold StratifiedGroupKFold CV for a specific sensor subset"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    b_acc, f1, prec, rec = [], [], [], []

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        y_pred = svm_model.predict(X_test_scaled)
        
        b_acc.append(balanced_accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, zero_division=0))
        prec.append(precision_score(y_test, y_pred, zero_division=0))
        rec.append(recall_score(y_test, y_pred, zero_division=0))
        
    return {
        "Modality": modality_name,
        "Balanced Acc": f"{np.mean(b_acc):.4f} (±{np.std(b_acc):.3f})",
        "F1-Score": f"{np.mean(f1):.4f} (±{np.std(f1):.3f})",
        "Precision": f"{np.mean(prec):.4f} (±{np.std(prec):.3f})",
        "Recall": f"{np.mean(rec):.4f} (±{np.std(rec):.3f})"
    }

def run_comprehensive_analysis(df_input, task_name, positive_class, negative_class):
    """Filters data for the task and evaluates all three sensor modalities"""
    task_df = df_input[df_input['condition'].isin([positive_class, negative_class])].copy()
    if task_df.empty:
        return

    groups = task_df['participant_id']
    y = task_df['condition'].apply(lambda x: 1 if x == positive_class else 0)
    
    # Isolate feature subsets
    all_features = [c for c in task_df.columns if c.startswith('acc_') or c.startswith('gyr_')]
    acc_features = [c for c in task_df.columns if c.startswith('acc_')]
    gyr_features = [c for c in task_df.columns if c.startswith('gyr_')]
    
    results = []
    results.append(evaluate_modality(task_df[all_features], y, groups, "Accel + Gyro (Combined)"))
    results.append(evaluate_modality(task_df[acc_features], y, groups, "Accel Only (Linear)"))
    results.append(evaluate_modality(task_df[gyr_features], y, groups, "Gyro Only (Rotation)"))
    
    # Print clean results table
    print(f"\n==============================================================================================")
    print(f" TASK: {task_name}")
    print(f" Samples: {task_df.shape[0]} | Patients: {groups.nunique()} | ({negative_class}=0, {positive_class}=1)")
    print(f"==============================================================================================")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print(f"==============================================================================================\n")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
FEATURES_PATH = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'

try:
    df = pd.read_csv(FEATURES_PATH)
    df['condition'] = df['condition'].astype(str).str.upper().str.strip()
    
    # Run Experiment 1: PD vs HC 
    run_comprehensive_analysis(df, "Parkinson's (PD) vs. Healthy Control (HC)", 
                               positive_class="PARKINSON'S", negative_class="HEALTHY")
    
    # Run Experiment 2: PD vs DD
    dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM']
    df_with_dd = df.copy()
    df_with_dd['condition'] = df_with_dd['condition'].apply(lambda x: 'DD' if x in dd_conditions else x)
    
    run_comprehensive_analysis(df_with_dd, "Parkinson's (PD) vs. Differential Diagnosis (DD)", 
                               positive_class="PARKINSON'S", negative_class="DD")

except FileNotFoundError:
    print(f"Error: Could not find feature file at {FEATURES_PATH}.")