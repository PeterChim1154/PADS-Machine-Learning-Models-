import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from lime import lime_tabular

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Group-Aware 82D Feature Extraction Pipeline ---
def extract_group_aware_features(manifest_path="file_list.csv", sampling_rate=50):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return None, None, None, None

    manifest_df = pd.read_csv(manifest_path)
    manifest_df["id"] = manifest_df["id"].astype(str).str.strip()
    manifest_df["condition"] = manifest_df["condition"].astype(str).str.strip()
    raw_csv_pairs = dict(zip(manifest_df["id"], manifest_df["condition"]))

    features_list = []
    labels_list = []
    patient_groups = [] 
    
    local_files = [f for f in os.listdir('.') if f.endswith('.bin')]

    # Reconstruct feature names
    feature_names = []
    channels = ["AX", "AY", "AZ", "GX", "GY", "GZ"]
    metrics = ["zcr", "mad", "gradient", "ptp", "std_dev", "iqr", "jerk", 
               "brady_energy", "tremor_energy", "postural_energy", 
               "dom_freq", "dom_energy_ratio", "entropy"]
    for ch in channels:
        for m in metrics:
            feature_names.append(f"{ch}_{m}")
    feature_names.extend(["corr_ax_ay", "corr_ay_az", "corr_gx_gy", "corr_gy_gz"])

    for filename in local_files:
        file_digits = ''.join(filter(str.isdigit, filename))
        if not file_digits: continue
        
        matched_id = None
        matched_diagnosis = None
        for csv_id, diagnosis in raw_csv_pairs.items():
            csv_id_digits = ''.join(filter(str.isdigit, csv_id))
            if csv_id_digits and int(file_digits) == int(csv_id_digits):
                matched_id = csv_id
                matched_diagnosis = diagnosis
                break

        if matched_diagnosis is None: continue
        
        try:
            raw_data = np.fromfile(filename, dtype=np.int16)
            reshaped_data = raw_data.reshape(-1, 6)
            df_scaled = pd.DataFrame(reshaped_data.astype(np.float32) / 16384.0, columns=channels)
        
            window_size = 250
            step_size = 500
            patient_windows = []
        
            for start_idx in range(0, len(df_scaled) - window_size, step_size):
                window_data = df_scaled.iloc[start_idx : start_idx + window_size]
                sma = np.mean(np.abs(window_data["AX"]) + np.abs(window_data["AY"]) + np.abs(window_data["AZ"]))
                if sma < 0.15: continue
                
                window_features = []
                for col in df_scaled.columns:
                    signal = window_data[col].values
                    signal_centered = signal - np.mean(signal)
                    zcr = len(np.nonzero(np.diff(signal_centered > 0))[0]) / len(signal)
                    mad = np.mean(np.abs(signal_centered))
                    gradient = np.mean(np.abs(np.diff(signal)))
                    ptp = np.ptp(signal)
                    std_dev = np.std(signal)
                    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
                    jerk = np.mean(np.abs(np.diff(signal, n=2)))
                
                    fft_vals = np.abs(np.fft.rfft(signal_centered))
                    freqs = np.fft.rfftfreq(window_size, d=1/sampling_rate)
                    fft_energy_total = np.sum(fft_vals**2) + 1e-6
                
                    brady_energy = np.sum(fft_vals[(freqs >= 0.5) & (freqs <= 3.0)]**2)
                    tremor_energy = np.sum(fft_vals[(freqs >= 4.0) & (freqs <= 6.0)]**2)
                    postural_energy = np.sum(fft_vals[(freqs >= 6.0) & (freqs <= 9.0)]**2)
                
                    peak_idx = np.argmax(fft_vals[1:]) + 1
                    dom_freq = freqs[peak_idx]
                    dom_energy_ratio = (fft_vals[peak_idx]**2) / fft_energy_total
                    normalized_fft_sq = (fft_vals**2) / fft_energy_total
                    entropy = -np.sum(normalized_fft_sq * np.log(normalized_fft_sq + 1e-6))
                
                    window_features.extend([
                        zcr, mad, gradient, ptp, std_dev, iqr, jerk,
                        brady_energy, tremor_energy, postural_energy,
                        dom_freq, dom_energy_ratio, entropy
                    ])
            
                corr_ax_ay = np.corrcoef(window_data["AX"].values, window_data["AY"].values)[0, 1]
                corr_ay_az = np.corrcoef(window_data["AY"].values, window_data["AZ"].values)[0, 1]
                corr_gx_gy = np.corrcoef(window_data["GX"].values, window_data["GY"].values)[0, 1]
                corr_gy_gz = np.corrcoef(window_data["GY"].values, window_data["GZ"].values)[0, 1]
            
                corr_features = [c if not np.isnan(c) else 0.0 for c in [corr_ax_ay, corr_ay_az, corr_gx_gy, corr_gy_gz]]
                window_features.extend(corr_features)
                patient_windows.append(window_features)
            
            if len(patient_windows) > 0:
                df_patient = pd.DataFrame(patient_windows)
                df_smoothed = df_patient.rolling(window=3, min_periods=1, center=True).mean()
                for smoothed_w in df_smoothed.values:
                    features_list.append(smoothed_w)
                    labels_list.append(matched_diagnosis)
                    patient_groups.append(matched_id)
                
        except Exception as e:
            pass

    if len(features_list) == 0:
        return None, None, None, None

    X = np.array(features_list)
    y_raw = np.array(labels_list)
    groups = np.array(patient_groups)

    y_clean = []
    for label in y_raw:
        label_lower = label.lower().replace(" ", "")
        if "parkinson" in label_lower and "atypical" not in label_lower:
            y_clean.append(1)  # PD
        else:
            y_clean.append(0)  # HC
          
    return X, np.array(y_clean), groups, feature_names


# --- Main Execution Block ---
if __name__ == "__main__":
    X, y, groups, feature_names = extract_group_aware_features()
    
    if X is not None:
        # Group-isolated split boundaries
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(X, y, groups=groups))
        
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        # --- THE CRITICAL FIX: STANDARD SCALING ---
        # This strips out the massive values so all variables play on the same math field.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Balance class distributions
        count_hc = np.sum(y_train == 0)
        count_pd = np.sum(y_train == 1)
        imbalance_ratio = float(count_hc) / float(count_pd) if count_pd > 0 else 1.0

        # Train Standardized XGBoost Classifier
        print("\nTraining Standardized XGBoost Model...")
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            scale_pos_weight=imbalance_ratio, 
            random_state=42, 
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print(f"Model Training Complete. True Unseen Patient Testing Accuracy: {model.score(X_test, y_test)*100:.2f}%")
        
        # Initialize the LIME Tabular Explainer on the scaled data space
        print("\nInitializing LIME Explainer...")
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=['Healthy/Other (0)', 'Parkinson\'s (1)'],
            mode='classification',
            random_state=42
        )
        
        # Target a specific patient window inside the scaled evaluation dataset
        patient_idx = 0
        patient_instance = X_test[patient_idx]
        true_label = y_test[patient_idx]
        isolated_patient_id = groups_test[patient_idx]
        
        pred_probs = model.predict_proba(patient_instance.reshape(1, -1))[0]
        print(f"\nAnalyzing Isolated Patient Window (ID: {isolated_patient_id}):")
        print(f" -> True Diagnosis Label: {true_label}")
        print(f" -> Model Predicted Probabilities: HC={pred_probs[0]:.4f}, PD={pred_probs[1]:.4f}")
        
        # Generate LIME Explanation
        print(" -> Generating local linear boundaries...")
        exp = explainer.explain_instance(
            data_row=patient_instance,
            predict_fn=model.predict_proba,
            num_features=10
        )
        
        # Save visual plot
        os.makedirs("figs", exist_ok=True)
        output_fig_path = "figs/lime_scaled_xgboost.png"
        
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Standardized XGBoost (Patient Profile: {isolated_patient_id})\nTrue Label: {true_label} | Model Pred PD: {pred_probs[1]*100:.1f}%", fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_fig_path, dpi=200)
        plt.close()
        
        print(f"\nExecution Complete! Balanced, standardized LIME plot saved to: '{output_fig_path}'")

