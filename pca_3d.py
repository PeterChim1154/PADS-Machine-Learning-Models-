import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def extract_82d_features(manifest_path="file_list.csv", sampling_rate=50):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found in the current directory.")
        return None, None

    manifest_df = pd.read_csv(manifest_path)
    manifest_df["id"] = manifest_df["id"].astype(str).str.strip()
    manifest_df["condition"] = manifest_df["condition"].astype(str).str.strip()
    raw_csv_pairs = dict(zip(manifest_df["id"], manifest_df["condition"]))

    features_list = []
    labels_list = []
    local_files = [f for f in os.listdir('.') if f.endswith('.bin')]
    print(f"Found {len(local_files)} binary files. Starting feature extraction...")

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
          
        try:
            raw_data = np.fromfile(filename, dtype=np.int16)
            reshaped_data = raw_data.reshape(-1, 6)
          
            df_scaled = pd.DataFrame(
                reshaped_data.astype(np.float32) / 16384.0,
                columns=["AX", "AY", "AZ", "GX", "GY", "GZ"]
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
                df_smoothed = df_patient.rolling(window=3, min_periods=1, center=True).mean()
                for smoothed_w in df_smoothed.values:
                    features_list.append(smoothed_w)
                    labels_list.append(matched_diagnosis)
                  
        except Exception as e:
            print(f"Skipping file {filename}: {str(e)}")

    if len(features_list) == 0:
        return None, None

    X = np.array(features_list)
    y_raw = np.array(labels_list)

    y_clean = []
    for label in y_raw:
        label_lower = label.lower().replace(" ", "").replace(".", "").replace("'", "").replace("’", "")
        if "parkinson" in label_lower and "atypical" not in label_lower:
            y_clean.append("PD")
        elif "healthy" in label_lower or "control" in label_lower:
            y_clean.append("HC")
        else:
            y_clean.append("DD")
            
    return X, np.array(y_clean)

def plot_3d_track(X_scaled, y, classes_to_keep, title, filename, colors):
    mask = np.isin(y, classes_to_keep)
    X_subset = X_scaled[mask]
    y_subset = y[mask]
    
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_subset)
    evr = pca.explained_variance_ratio_
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for cls in np.unique(y_subset):
        cls_mask = (y_subset == cls)
        ax.scatter(
            X_pca[cls_mask, 0], X_pca[cls_mask, 1], X_pca[cls_mask, 2],
            label=f"Cohort: {cls}", c=colors[cls], alpha=0.5, edgecolors='none', s=25
        )
        
    ax.set_title(f"{title}\nTotal Extracted Windows: {len(X_subset):,} | Total Variance: {evr.sum()*100:.2f}%", fontsize=12, weight='bold', pad=15)
    ax.set_xlabel(f"PC 1 ({evr[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC 2 ({evr[1]*100:.1f}%)", fontsize=10)
    ax.set_zlabel(f"PC 3 ({evr[2]*100:.1f}%)", fontsize=10)
    ax.legend(loc='best', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Exported 3D Graph: {filename}")

if __name__ == "__main__":
    X, y = extract_82d_features()
    if X is not None:
        X_scaled = StandardScaler().fit_transform(X)
        palette = {"PD": "#d95f02", "HC": "#1b9e77", "DD": "#7570b3"}
        
        plot_3d_track(X_scaled, y, ["PD", "HC"], "Track 1: Idiopathic Baseline Latent Space (PD vs. HC)", "pca_3d_track1_pd_vs_hc.png", palette)
        plot_3d_track(X_scaled, y, ["PD", "DD"], "Track 2: Phenotypic Mimicry Latent Space (PD vs. DD)", "pca_3d_track2_pd_vs_dd.png", palette)

