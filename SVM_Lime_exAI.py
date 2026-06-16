import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime import lime_tabular  # Required for tabular LIME explainability
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

def evaluate_modality(X, y, groups, modality_name, task_name):
    """Helper to run 5-Fold StratifiedGroupKFold CV for a specific sensor subset"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    b_acc, f1, prec, rec = [], [], [], []
    
    # Track model variables from the final loop split to feed directly into LIME
    last_model = None
    last_scaler = None
    
    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # CONFIG FIXED: Added probability=True so LIME can read predict_proba output
        svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        y_pred = svm_model.predict(X_test_scaled)
        
        b_acc.append(balanced_accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, zero_division=0))
        prec.append(precision_score(y_test, y_pred, zero_division=0))
        rec.append(recall_score(y_test, y_pred, zero_division=0))
        
        # Cache standard scaling weights and trained support vectors
        last_model = svm_model
        last_scaler = scaler

    # --- INTEGRATING LIME EXPLAINABLE AI ENGINE FOR SVM ---
    # Isolate to the combined modality matrix to avoid duplicate graph output clutter
    if modality_name == "Accel + Gyro (Combined)":
        print(f" -> Initializing LIME Tabular Explainer specifically for the SVM architecture ({task_name})...")
        
        # Create pipeline wrapper to cleanly pass LIME's internal raw sample perturbations
        def svm_predict_proba_pipeline(numpy_array_input):
            # Scale the raw data drift first using the cached fold training data bounds
            scaled_input = last_scaler.transform(numpy_array_input)
            return last_model.predict_proba(scaled_input)

        # SPECIFICATION UPDATE: Explicit class names identifying SVM target mappings
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.to_numpy(),
            feature_names=X.columns.tolist(),
            class_names=['Non-PD (SVM Baseline)', 'Parkinson\'s (SVM Target)'],
            mode='classification',
            random_state=42
        )
        
        # Target evaluation instance index (Sample 0 from our testing vector)
        target_sample_idx = 0
        raw_target_instance = X.iloc[test_idx].iloc[target_sample_idx].to_numpy()
        
        exp = explainer.explain_instance(
            data_row=raw_target_instance,
            predict_fn=svm_predict_proba_pipeline,
            num_features=10
        )
        
        # Render and Save LIME Visual Explanation
        os.makedirs("figs", exist_ok=True)
        sanitized_task = task_name.replace("'", "").replace(".", "").replace(" ", "_").replace("(", "").replace(")", "").lower()
        fig = exp.as_pyplot_figure()
        
        # SPECIFICATION UPDATE: Clearly denoting the SVM model on plot title
        plt.title(f"SVM Model LIME Feature Explanations: {task_name}\n(Local Support Vector Machine Decision Boundary Evaluation)", fontsize=11, fontweight='bold', pad=10)
        plt.tight_layout()
        
        fig_path = f"figs/svm_lime_{sanitized_task}_patient_breakdown.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f" 🚀 Success! Saved local SVM-specific LIME explanation figure to: {fig_path}")

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
    results.append(evaluate_modality(task_df[all_features], y, groups, "Accel + Gyro (Combined)", task_name))
    results.append(evaluate_modality(task_df[acc_features], y, groups, "Accel Only (Linear)", task_name))
    results.append(evaluate_modality(task_df[gyr_features], y, groups, "Gyro Only (Rotation)", task_name))
    
    # Print clean results table
    print(f"\n==============================================================================================")
    print(f" TASK: {task_name} (EXPLAINABLE AI SUPPORT VECTOR MACHINE RUN)")
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
        run_comprehensive_analysis(df, "PD vs. HC", 
                                   positive_class="PARKINSON'S", negative_class="HEALTHY")
        
        # Run Experiment 2: PD vs DD
        dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM']
        df_with_dd = df.copy()
        df_with_dd['condition'] = df_with_dd['condition'].apply(lambda x: 'DD' if x in dd_conditions else x)
        
        run_comprehensive_analysis(df_with_dd, "PD vs. DD", 
                                   positive_class="PARKINSON'S", negative_class="DD")

    except FileNotFoundError:
        print(f"Error: Could not find feature file at {FEATURES_PATH}.")