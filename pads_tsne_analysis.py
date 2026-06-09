import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def extract_window_features(signal_segment, sampling_rate=100.0):
    """Extracts raw kinematic metrics from a specific time-series window slice"""
    features = []
    x, y, z = signal_segment[:, 0], signal_segment[:, 1], signal_segment[:, 2]
    
    # Time domain metrics per window
    features.extend([np.mean(x), np.std(x), np.std(y), np.std(z)])
    
    # Magnitude metrics per window
    mag = np.sqrt(x**2 + y**2 + z**2)
    features.extend([np.mean(mag), np.std(mag), np.mean(np.abs(x) + np.abs(y) + np.abs(z))])
    
    # Quick, robust frequency proxy for windows: Zero Crossing Rate
    # Highly efficient for dense window matrices compared to full FFT stacks
    x_zerocross = np.nonzero(np.diff(x > np.mean(x)))[0]
    features.append(len(x_zerocross) / (len(x) / sampling_rate))
    
    # Vector magnitude variance metric
    features.append(np.var(mag))
        
    return features

def generate_task_dashboard(X_full, y_full, pos_class, neg_class, task_title, output_filename):
    """Filters window matrix for a task and renders a side-by-side 2D/3D dashboard"""
    # Isolate relevant rows for this specific comparison task
    mask = np.isin(y_full, [pos_class, neg_class])
    X_task = X_full[mask]
    y_task = y_full[mask]
    
    if len(X_task) == 0:
        print(f"Skipping task [{task_title}]: No windows found matching categories.")
        return
        
    # Scale feature subset specifically for this sub-space
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_task)
    
    print(f"\nComputing manifold geometry for: {task_title} ({len(X_task)} dense segments)...")
    
    # Compute t-SNE coordinate systems
    tsne_2d = TSNE(n_components=2, perplexity=40, random_state=42)
    X_2d = tsne_2d.fit_transform(X_scaled)
    
    tsne_3d = TSNE(n_components=3, perplexity=40, random_state=42)
    X_3d = tsne_3d.fit_transform(X_scaled)
    
    # Setup slide-ready dual dashboard canvas
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f"Task Analysis: {task_title}\nWindow-Level Spatial Clustering Behavior", fontsize=14, fontweight='bold')
    
    colors_map = {neg_class: '#2ca02c', pos_class: '#d62728'} # Clinical Green vs Warning Red
    
    # --- PLOT 1: 2D VISUALIZATION VIEW ---
    ax1 = fig.add_subplot(1, 2, 1)
    for target in [neg_class, pos_class]:
        c_mask = (y_task == target)
        # Using confirmed axes swap configuration to fix orientation alignment
        ax1.scatter(X_2d[c_mask, 1], X_2d[c_mask, 0], 
                    c=colors_map[target], label=target, alpha=0.4, s=12, edgecolors='none')
        
    ax1.set_title("A) 2D Spatial Layout Projection", fontsize=11, fontweight='bold')
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.legend(loc='upper right', markerscale=2)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # --- PLOT 2: 3D INTERACTIVE MANIFOLD VIEW ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for target in [neg_class, pos_class]:
        c_mask = (y_task == target)
        ax2.scatter(X_3d[c_mask, 0], X_3d[c_mask, 1], X_3d[c_mask, 2], 
                    c=colors_map[target], label=target, alpha=0.4, s=12, edgecolors='none')
        
    ax2.set_title("B) 3D Intermediary Clustering Manifold", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.set_zlabel("Dimension 3")
    ax2.legend(loc='upper right', markerscale=2)
    ax2.view_init(elev=20, azim=40)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"[+] Saved complete slide matrix as: '{output_filename}'")
    plt.show()

# =========================================================================
# FILE PARSING EXECUTION ROUTINE
# =========================================================================
DEMOGRAPHICS_PATH = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/file_list.csv" 
RAW_DATA_DIR = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/movement"

try:
    print("Loading patient metadata from file_list.csv...")
    demo_df = pd.read_csv(DEMOGRAPHICS_PATH)
    demo_df['id_str'] = demo_df['id'].astype(str).str.zfill(3)

    # Standardize conditions mapping explicitly to isolate cohorts
    dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM', 'DD']
    condition_lookup = {}
    for _, row in demo_df.iterrows():
        cond = str(row['condition']).upper().strip()
        if cond in dd_conditions:
            condition_lookup[row['id_str']] = 'DD'
        elif 'PARKINSON' in cond:
            condition_lookup[row['id_str']] = 'PD'
        else:
            condition_lookup[row['id_str']] = 'HC'

    all_binary_files = []
    for ext in ['*.bin', '*.dat', '*.raw']:
        all_binary_files.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", ext), recursive=True))

    print(f"Slicing uncompressed segments from {len(all_binary_files)} binary streams...")
    
    X_list, y_list = [], []
    WINDOW_SIZE, STEP_SIZE = 500, 500 # 2-second windows with 50% overlap step size

    for full_file_path in all_binary_files:
        filename = os.path.basename(full_file_path)
        id_match = re.search(r'\d+', filename)
        if not id_match:
            continue
            
        pid = id_match.group(0).zfill(3)
        if pid not in condition_lookup:
            continue
            
        cohort_label = condition_lookup[pid]

        try:
            with open(full_file_path, 'rb') as binary_file:
                raw_data = np.fromfile(binary_file, dtype=np.float32)
                raw_matrix = raw_data.reshape(-1, 6)
                total_samples = len(raw_matrix)
                
                for start_idx in range(0, total_samples - WINDOW_SIZE, STEP_SIZE):
                    window_data = raw_matrix[start_idx:start_idx + WINDOW_SIZE]
                    acc_feat = extract_window_features(window_data[:, 0:3])
                    gyr_feat = extract_window_features(window_data[:, 3:6])
                    
                    X_list.append(acc_feat + gyr_feat)
                    y_list.append(cohort_label)
        except:
            pass

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    print(f"Successfully vectorized {len(X_all)} absolute window segments across all clinical cohorts.")

    # -------------------------------------------------------------------------
    # DASHBOARD 1: PARKINSON'S (PD) VS HEALTHY CONTROLS (HC)
    # -------------------------------------------------------------------------
    generate_task_dashboard(X_all, y_all, pos_class='PD', neg_class='HC', 
                            task_title="Parkinson's Disease (PD) vs. Healthy Controls (HC)",
                            output_filename="dashboard_windowed_pd_vs_hc.png")

    # -------------------------------------------------------------------------
    # DASHBOARD 2: PARKINSON'S (PD) VS DIFFERENTIAL DIAGNOSIS (DD)
    # -------------------------------------------------------------------------
    generate_task_dashboard(X_all, y_all, pos_class='PD', neg_class='DD', 
                            task_title="Parkinson's Disease (PD) vs. Differential Diagnosis Mimics (DD)",
                            output_filename="dashboard_windowed_pd_vs_dd.png")

except FileNotFoundError:
    print(f"Error: Verification failed. Confirm file positioning mapping at: {DEMOGRAPHICS_PATH}")