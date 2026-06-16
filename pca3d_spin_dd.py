import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
np.random.seed(42)

def extract_82d_features_dd_vs_pd(manifest_path="file_list.csv", sampling_rate=50):
    print("Step 1: Parsing manifest and extracting 82-dimensional kinematic features (DD vs. PD)...")
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
        print(f" Critical Manifest Failure: Check your CSV formatting. Details: {e}")
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
        
        # TARGET SEPARATION: Parkinson's vs Differential Diagnosis (Atypical Parkinsonism)
        if "parkinson" in label_lower and "atypical" not in label_lower:
            binary_label = 1  # Parkinson's Disease (PD)
        elif any(dd_term in label_lower for dd_term in ["atypical", "msa", "psp", "cbd", "dld", "differential"]):
            binary_label = 0  # Differential Diagnosis (DD)
        else:
            continue  # Skip Healthy Controls or unmapped conditions
          
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
        print(" Extraction Error: No matching DD vs. PD segments found. Check condition strings in CSV.")
        return None, None, None

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y, feature_names


def run_3d_pca_spin_dd_vs_pd():
    X, y, feature_names = extract_82d_features_dd_vs_pd()
    if X is None or len(X) == 0:
        print("Pipeline stopped.")
        return

    print(f" -> Found {np.sum(y == 0)} windows for Differential Diagnosis (DD)")
    print(f" -> Found {np.sum(y == 1)} windows for Parkinson's Disease (PD)")

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nStep 2: Fitting 3-Dimensional PCA Decomposition...")
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_ * 100
    print(f" -> PC1 Explains: {explained_variance[0]:.2f}% Variance")
    print(f" -> PC2 Explains: {explained_variance[1]:.2f}% Variance")
    print(f" -> PC3 Explains: {explained_variance[2]:.2f}% Variance")
    print(f" -> Total Cumulative Variance Retained: {np.sum(explained_variance):.2f}%")

    os.makedirs("figs", exist_ok=True)

    print("\nStep 3: Rendering 3D Spin Sequence for DD vs. PD...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot configuration: Purple/Violet for DD, Crimson/Red for PD
    scatter_dd = ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], X_pca[y==0, 2], 
                            c='#9467bd', label='Differential Diagnosis (0)', alpha=0.6, edgecolors='none', s=15)
    scatter_pd = ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], X_pca[y==1, 2], 
                            c='#d62728', label="Parkinson's Disease (1)", alpha=0.6, edgecolors='none', s=15)

    # Labels and Legends
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_zlabel(f"PC3 ({explained_variance[2]:.1f}%)", fontweight='bold', labelpad=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

    fixed_elevation = 20
    
    for azim_angle in range(0, 360, 10):
        ax.view_init(elev=fixed_elevation, azim=azim_angle)
        
        plt.title(f"3D PCA: Differential Diagnosis vs. Parkinson's\n[Rotation Angle Azimuth: {azim_angle}°]", fontsize=12, weight='bold', pad=10)
        
        # Unique prefix so it doesn't overwrite your healthy control images
        filename = f"figs/pca_3d_dd_vs_pd_angle_{azim_angle:03d}.png"
        
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f" -> Exported frame: {filename}")

    plt.close()
    print(f"\nExecution complete! 36 frames saved to 'figs/' folder with prefix 'pca_3d_dd_vs_pd_'.")

if __name__ == "__main__":
    import sys
    sys.dont_write_bytecode = True
    run_3d_pca_spin_dd_vs_pd()

