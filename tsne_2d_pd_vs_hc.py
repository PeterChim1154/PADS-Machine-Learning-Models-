import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

condition_lookup = {}
for _, row in demo_df.iterrows():
    cond = str(row['condition']).upper().strip()
    if 'PARKINSON' in cond:
        condition_lookup[row['id_str']] = 'PD'
    elif cond not in ['MULTIPLE SCLEROSIS', 'OTHER MOVEMENT DISORDERS', 'ESSENTIAL TREMOR', 'ATYPICAL PARKINSONISM', 'DD']:
        condition_lookup[row['id_str']] = 'HC'

all_binary_files = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.bin"), recursive=True)
X_list, y_list = [], []
WINDOW_SIZE, STEP_SIZE = 500, 500 # 5-second back-to-back windows

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

print("Computing 2D t-SNE (PD vs HC)...")
X_2d = TSNE(n_components=2, perplexity=40, random_state=42).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 9))
colors = {'HC': '#2ca02c', 'PD': '#d62728'}
for label in ['HC', 'PD']:
    mask = (y_arr == label)
    ax.scatter(X_2d[mask, 1], X_2d[mask, 0], c=colors[label], label=label, alpha=0.4, s=12, edgecolors='none')

ax.set_title("2D t-SNE Layout: Parkinson's Disease (PD) vs. Healthy Controls (HC)\nSequential 5-Second Windows (No Gaps)", fontsize=12, fontweight='bold')
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.legend(loc='upper right', markerscale=2)
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("tsne_2d_pd_vs_hc.png", dpi=300)
print("[+] Saved: 'tsne_2d_pd_vs_hc.png'")
plt.show()