import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # CONFIG FIXED: Added missing animation import
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

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

DEMOGRAPHICS_PATH = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/file_list.csv" 
RAW_DATA_DIR = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/movement"

print("Loading metadata...")
demo_df = pd.read_csv(DEMOGRAPHICS_PATH)
demo_df['id_str'] = demo_df['id'].astype(str).str.zfill(3)

dd_conditions = ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM', 'DD']
condition_lookup = {}
for _, row in demo_df.iterrows():
    cond = str(row['condition']).upper().strip()
    if 'PARKINSON' in cond:
        condition_lookup[row['id_str']] = 'PD'
    elif cond in dd_conditions:
        condition_lookup[row['id_str']] = 'DD'

all_binary_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.bin"), recursive=True)
X_list, y_list = [], []
WINDOW_SIZE, STEP_SIZE = 500, 250 

for full_file_path in all_binary_files:
    filename = os.path.basename(full_file_path)
    id_match = re.search(r'\d+', filename)
    if not id_match or id_match.group(0).zfill(3) not in condition_lookup:
        continue
    cohort_label = condition_lookup[id_match.group(0).zfill(3)]
    try:
        with open(full_file_path, 'rb') as f:
            raw_matrix = np.fromfile(f, dtype=np.float32).reshape(-1, 6)
            for i in range(0, len(raw_matrix) - WINDOW_SIZE, STEP_SIZE):
                window = raw_matrix[i:i + WINDOW_SIZE]
                X_list.append(extract_window_features(window[:, 0:3]) + extract_window_features(window[:, 3:6]))
                y_list.append(cohort_label)
    except:
        pass

X_scaled = StandardScaler().fit_transform(np.array(X_list))
y_arr = np.array(y_list)

print("Computing 3D t-SNE (PD vs DD)...")
X_3d = TSNE(n_components=3, perplexity=40, random_state=42).fit_transform(X_scaled)

# -------------------------------------------------------------------
# VISUALIZATION & ROTATION ANIMATION ENGINE
# -------------------------------------------------------------------
plt.ion() 

fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={'projection': '3d'})
colors = {'DD': '#1f77b4', 'PD': '#d62728'}

for label in ['DD', 'PD']:
    mask = (y_arr == label)
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], c=colors[label], label=label, alpha=0.4, s=12, edgecolors='none')

ax.set_title("Interactive t-SNE: Parkinson's Disease (PD) vs. Differential Diagnosis Mimics (DD)\nSequential 5-Second Windows", fontsize=11, fontweight='bold')
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
ax.legend(loc='upper right', markerscale=2)

# Initial viewing angle setup
initial_elevation = 20
initial_azimuth = 40
ax.view_init(elev=initial_elevation, azim=initial_azimuth)
plt.tight_layout()

# CONFIG FIXED: Save the static base plot to disk cleanly without ax object typos
plt.savefig("tsne_3d_pd_vs_dd.png", dpi=300)

def update_rotation(frame):
    """
    Update loop called continuously by FuncAnimation.
    Adding the frame value shifts the perspective from left to right.
    """
    # CONFIG FIXED: Swapped minus to plus to ensure true left-to-right clockwise spinning
    current_azim = (initial_azimuth + frame) % 360
    ax.view_init(elev=initial_elevation, azim=current_azim)
    return ax,

print("\n[+] Spinning up real-time left-to-right rotation window...")
print(" -> Close the pop-up window or press Ctrl+C in terminal to stop.")

ani = FuncAnimation(
    fig, 
    update_rotation, 
    frames=360, 
    interval=40, 
    blit=False, 
    repeat=True
)

plt.ioff() 
plt.show()