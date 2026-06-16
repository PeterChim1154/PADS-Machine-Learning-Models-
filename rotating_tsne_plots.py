import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# --- FEATURE EXTRACTION CORE ---
def extract_window_features(signal_segment, sampling_rate=100.0):
    features = []
    x, y, z = signal_segment[:, 0], signal_segment[:, 1], signal_segment[:, 2]
    features.extend([np.mean(x), np.std(x), np.std(y), np.std(z)])
    mag = np.sqrt(x**2 + y**2 + z**2)
    features.extend([np.mean(mag), np.std(mag), np.mean(np.abs(x) + np.abs(y) + np.abs(z))])
    x_zerocross = np.nonzero(np.diff(x > np.mean(x)))[0]
    features.append(len(x_zerocross) / (len(x) / sampling_rate))
    features.append(np.var(mag))
    return features

# --- MASTER DATASET PROCESSING WRAPPER ---
def load_and_process_task(demographics_path, raw_data_dir, target_cohort="DD", window_size=500, step_size=500):
    print(f"\n[1/3] Loading metadata for PD vs {target_cohort} task...")
    demo_df = pd.read_csv(demographics_path)
    demo_df['id_str'] = demo_df['id'].astype(str).str.zfill(3)

    dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM', 'DD']
    condition_lookup = {}
    
    for _, row in demo_df.iterrows():
        cond = str(row['condition']).upper().strip()
        if 'PARKINSON' in cond:
            condition_lookup[row['id_str']] = 'PD'
        elif target_cohort == "DD" and cond in dd_conditions:
            condition_lookup[row['id_str']] = 'DD'
        elif target_cohort == "HC" and ('HEALTHY' in cond or 'CONTROL' in cond):
            condition_lookup[row['id_str']] = 'HC'

    all_binary_files = glob.glob(os.path.join(raw_data_dir, "**", "*.bin"), recursive=True)
    X_list, y_list = [], []

    print(f" -> Parsing sensor binaries (Window: {window_size}, Step: {step_size})...")
    for full_file_path in all_binary_files:
        filename = os.path.basename(full_file_path)
        id_match = re.search(r'\d+', filename)
        if not id_match or id_match.group(0).zfill(3) not in condition_lookup:
            continue
        cohort_label = condition_lookup[id_match.group(0).zfill(3)]
        try:
            with open(full_file_path, 'rb') as f:
                raw_matrix = np.fromfile(f, dtype=np.float32).reshape(-1, 6)
                for i in range(0, len(raw_matrix) - window_size, step_size):
                    window = raw_matrix[i:i + window_size]
                    X_list.append(extract_window_features(window[:, 0:3]) + extract_window_features(window[:, 3:6]))
                    y_list.append(cohort_label)
        except:
            pass

    X_scaled = StandardScaler().fit_transform(np.array(X_list))
    y_arr = np.array(y_list)
    
    print(f" -> Computing 3D t-SNE space for PD vs {target_cohort}...")
    X_3d = TSNE(n_components=3, perplexity=40, random_state=42).fit_transform(X_scaled)
    return X_3d, y_arr

# --- ROTATION & ANIMATION PIPELINE ENGINE ---
def generate_rotating_tsne(X_3d, y_arr, target_cohort="DD"):
    print(f"[2/3] Configuring 3D Plot Layout for PD vs {target_cohort}...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={'projection': '3d'})
    
    # Select color maps based on clinical targets
    if target_cohort == "DD":
        colors = {'DD': '#1f77b4', 'PD': '#d62728'}
        labels_list = ['DD', 'PD']
        title_str = "Interactive 3D t-SNE: Parkinson's Disease (PD) vs. Diagnostic Mimics (DD)\nSequential 5-Second Windows"
    else:
        colors = {'HC': '#2ca02c', 'PD': '#d62728'}
        labels_list = ['HC', 'PD']
        title_str = "Interactive 3D t-SNE: Parkinson's Disease (PD) vs. Healthy Controls (HC)\nSequential 5-Second Windows (0% Overlap)"

    # Plot Scatter Maps (Optimized with solid opaque layout for performance safety)
    for label in labels_list:
        mask = (y_arr == label)
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
            c=colors[label], label=label, 
            s=8, alpha=1.0, edgecolors='none', rasterized=True
        )

    ax.set_title(title_str, fontsize=11, fontweight='bold')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(loc='upper right', markerscale=2)

    initial_elevation = 20
    initial_azimuth = 40
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    plt.tight_layout()

    # Save out high-res static copy
    static_img_name = f"tsne_3d_pd_vs_{target_cohort.lower()}.png"
    plt.savefig(static_img_name, dpi=300)
    print(f" -> Saved static high-res frame to: {static_img_name}")

    # Core left-to-right calculation loop
    def update_rotation(frame):
        current_azim = (initial_azimuth + frame) % 360
        ax.view_init(elev=initial_elevation, azim=current_azim)
        return ax,

    gif_filename = f"rotation_pd_vs_{target_cohort.lower()}.gif"
    print(f"[3/3] Exporting 3D rotating animation loop into '{gif_filename}'...")
    print(" -> (Please sit tight, compiling frames via pillow writer can take a moment...)")
    
    # 90 frames at 4-degree steps creates a perfect full 360-degree rotation loop
    ani_export = FuncAnimation(fig, update_rotation, frames=range(0, 360, 4), blit=False)
    ani_export.save(gif_filename, writer='pillow', fps=10)
    print(f"🚀 Success! Saved rotating animation to: {gif_filename}")

    print(" -> Launching interactive display window (Close or Ctrl+C to jump to next)...")
    ani_live = FuncAnimation(fig, update_rotation, frames=360, interval=100, blit=False, repeat=True)
    
    plt.ioff()
    plt.show()

# --- EXECUTION ENGINE ---
if __name__ == "__main__":
    import sys
    sys.dont_write_bytecode = True
    
    DEMOGRAPHICS_PATH = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/file_list.csv" 
    RAW_DATA_DIR = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/movement"

    # RUN TRACK 1: Parkinson's vs. Diagnostic Mimics (DD)
    X_3d_dd, y_arr_dd = load_and_process_task(DEMOGRAPHICS_PATH, RAW_DATA_DIR, target_cohort="DD", window_size=500, step_size=500)
    generate_rotating_tsne(X_3d_dd, y_arr_dd, target_cohort="DD")

    print("\n" + "="*60 + "\nMoving onto the next diagnostic evaluation cohort pipeline...\n" + "="*60)

    # RUN TRACK 2: Parkinson's vs. Clean Healthy Controls (HC)
    X_3d_hc, y_arr_hc = load_and_process_task(DEMOGRAPHICS_PATH, RAW_DATA_DIR, target_cohort="HC", window_size=500, step_size=500)
    generate_rotating_tsne(X_3d_hc, y_arr_hc, target_cohort="HC")
    
    print("\n🎉 Comprehensive Execution Finished! Look inside your directory for your newly minted GIF files.")