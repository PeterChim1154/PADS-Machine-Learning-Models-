import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

def extract_82d_features(manifest_path="file_list.csv", sampling_rate=50):
    print("Step 1: Parsing manifest and extracting 82-dimensional kinematic features...")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found.")
        return None, None, None

    try:
        manifest_df = pd.read_csv(manifest_path).dropna(subset=["id", "condition"])
        manifest_df["id"] = manifest_df["id"].astype(str).str.strip()
        manifest_df["condition"] = manifest_df["condition"].astype(str).str.strip()
        raw_csv_pairs = dict(zip(manifest_df["id"], manifest_df["condition"]))
        print(f" -> Successfully mapped {len(raw_csv_pairs)} clean records from CSV.")
    except Exception as e:
        print(f"❌ Critical Manifest Failure: Check your CSV formatting. Details: {e}")
        return None, None, None

    features_list = []
    labels_list = []
    local_files = [f for f in os.listdir('.') if f.endswith('.bin')]

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
          
        matched_diagnosis = None
        for csv_id, diagnosis in raw_csv_pairs.items():
            csv_id_digits = ''.join(filter(str.isdigit, csv_id))
            if csv_id_digits and int(file_digits) == int(csv_id_digits):
                matched_diagnosis = diagnosis
                break

        if matched_diagnosis is None: continue
        
        label_lower = str(matched_diagnosis).lower().replace(" ", "").replace(".", "").replace("'", "").replace("’", "")
        if "parkinson" in label_lower and "atypical" not in label_lower:
            binary_label = 1
        elif "healthy" in label_lower or "control" in label_lower:
            binary_label = 0
        else:
            continue
          
        try:
            raw_data = np.fromfile(filename, dtype=np.int16)
            if len(raw_data) == 0: continue
            
            reshaped_data = raw_data.reshape(-1, 6)
            df_scaled = pd.DataFrame(
                reshaped_data.astype(np.float32) / 16384.0,
                columns=channels
            )
          
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
                df_smoothed = df_patient.rolling(window=3, min_periods=1, center=True).mean().fillna(0.0)
                
                for smoothed_w in df_smoothed.values:
                    features_list.append(smoothed_w.astype(np.float32))
                    labels_list.append(binary_label)
                  
        except Exception as e:
            pass

    if len(features_list) == 0:
        print("❌ Extraction Error: No matching or valid dataset segments found.")
        return None, None, None

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y, feature_names


class ParkinsonKinematicNN(nn.Module):
    def __init__(self, input_dim):
        super(ParkinsonKinematicNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)


def run_explainable_nn_pipeline():
    X, y, feature_names = extract_82d_features()
    if X is None or len(X) == 0:
        print("Pipeline stopped.")
        return

    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    print("\nStep 2: Initializing Deep Neural Network Training loop...")
    model = ParkinsonKinematicNN(input_dim=82)
    
    num_healthy = np.sum(y_train == 0)
    num_parkinsons = np.sum(y_train == 1)
    pos_weight_val = torch.tensor([num_healthy / num_parkinsons], dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

    model.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f" -> Epoch {epoch+1}/15 Complete | Avg Batch Loss: {epoch_loss / len(X_train_scaled):.4f}")

    model.eval()
    with torch.no_grad():
        raw_logits = model(X_test_t)
        probabilities = torch.sigmoid(raw_logits).numpy()
        y_pred = (probabilities >= 0.5).astype(int)

    print("\n================ NEURAL NETWORK PERFORMANCE SUMMARY ================")
    print(f"Overall Testing Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy Control (0)", "Parkinson's (1)"]))
    print("====================================================================\n")

    print("Step 3: Calculating SHAP values using DeepExplainer gradients...")
    
    background_indices = np.random.choice(X_train_scaled.shape[0], 100, replace=False)
    background_tensor = torch.tensor(X_train_scaled[background_indices], dtype=torch.float32)
    
    explainer = shap.DeepExplainer(model, background_tensor)
    
    test_sample_size = 30
    X_test_subset_t = X_test_t[:test_sample_size]
    
    shap_values_list = explainer.shap_values(X_test_subset_t)
    if isinstance(shap_values_list, list):
        shap_matrix = shap_values_list[0]
    else:
        shap_matrix = shap_values_list

    # Reshape the summary matrix to strip the nested logit dimension channel out
    if len(shap_matrix.shape) == 3:
        shap_matrix_2d = shap_matrix[:, :, 0]
    elif len(shap_matrix.shape) == 2:
        shap_matrix_2d = shap_matrix
    else:
        shap_matrix_2d = np.squeeze(shap_matrix)

    expected_val_raw = explainer.expected_value
    if hasattr(expected_val_raw, "numpy"):
        expected_val_np = expected_val_raw.numpy()
    else:
        expected_val_np = np.array(expected_val_raw)

    if isinstance(expected_val_np, np.ndarray):
        expected_val_np = expected_val_np.ravel()[0]

    explanation_object = shap.Explanation(
        values=shap_matrix_2d,
        base_values=np.repeat(expected_val_np, test_sample_size),
        data=X_test_scaled[:test_sample_size],
        feature_names=feature_names
    )

    os.makedirs("figs", exist_ok=True)

    print(" -> Rendering Global NN Feature Importance Summary Map...")
    plt.figure(figsize=(12, 8))
    
    # FIX: Pass the cleaned 2D matrix directly along with the input features to refresh the map layout
    shap.summary_plot(shap_matrix_2d, X_test_scaled[:test_sample_size], feature_names=feature_names, show=False)
    
    plt.title("Neural Network: Global Kinematic Feature Attribution (SHAP Summary)", fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("figs/nn_xai_global_importance.png", dpi=300)
    plt.close()

    print(" -> Rendering Local Patient Sample Prediction Breakdown...")
    plt.figure(figsize=(11, 6))
    
    patient_sample_explanation = explanation_object[0]
    shap.plots.waterfall(patient_sample_explanation, show=False)
    
    plt.title("Neural Network: Local Prediction Weight Verification (Patient Sample #0)", fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("figs/nn_xai_local_patient_breakdown.png", dpi=300)
    plt.close()
    
    print("\nExecution complete. Neural Network plots successfully generated and saved to figs/ folder!")

if __name__ == "__main__":
    import sys
    sys.dont_write_bytecode = True
    run_explainable_nn_pipeline()

