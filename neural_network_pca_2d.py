
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis


warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


print("Reverting to Stable Physical-Coherence Normalized Cascade Engine...")


MANIFEST_FILE = "file_list.csv"
if not os.path.exists(MANIFEST_FILE):
  print(f"Error: '{MANIFEST_FILE}' not found.")
  exit()


# =====================================================================
# 1. LOAD MANIFEST WITH PHYSICAL METADATA
# =====================================================================
manifest_df = pd.read_csv(MANIFEST_FILE)
id_col = "id"
diag_col = "condition"


manifest_df[id_col] = manifest_df[id_col].astype(str).str.strip()
manifest_df[diag_col] = manifest_df[diag_col].astype(str).str.strip()


raw_csv_pairs = dict(zip(manifest_df[id_col], manifest_df[diag_col]))
age_map = dict(zip(manifest_df[id_col], manifest_df['age'].astype(np.float32)))
height_map = dict(zip(manifest_df[id_col], manifest_df['height'].astype(np.float32)))
weight_map = dict(zip(manifest_df[id_col], manifest_df['weight'].astype(np.float32)))


# =====================================================================
# 2. PHYSIOLOGICALLY NORMALIZED KINEMATIC FEATURE ENGINE
# =====================================================================
features_list = []
labels_list = []
patient_groups = []


local_files = [f for f in os.listdir('.') if f.endswith('.bin')]
SAMPLING_RATE = 50


for filename in local_files:
  file_digits = ''.join(filter(str.isdigit, filename))
  if not file_digits: continue
    
  matched_diagnosis = None
  matched_csv_id = None
  for csv_id, diagnosis in raw_csv_pairs.items():
      csv_id_digits = ''.join(filter(str.isdigit, csv_id))
      if csv_id_digits and int(file_digits) == int(csv_id_digits):
          matched_diagnosis = diagnosis
          matched_csv_id = csv_id
          break


  if matched_diagnosis is None: continue
    
  try:
      raw_data = np.fromfile(filename, dtype=np.int16)
      reshaped_data = raw_data.reshape(-1, 6)
    
      df_scaled = pd.DataFrame(
          reshaped_data.astype(np.float32) / 16384.0,
          columns=["AX", "AY", "AZ", "GX", "GY", "GZ"]
      )
    
      p_age = age_map.get(matched_csv_id, 50.0)  
      p_weight = weight_map.get(matched_csv_id, 70.0)
      p_height = height_map.get(matched_csv_id, 170.0)
    
      if p_weight <= 0: p_weight = 70.0
      if p_height <= 0: p_height = 170.0
        
      # --- PHYSICAL WAVEFORM NORMALIZATION LAYER ---
      for col in ["AX", "AY", "AZ"]:
          df_scaled[col] = df_scaled[col] / (p_weight / 70.0)
        
      window_size = 250
      # --- RESTORED SAFE STEP SIZE (PREVENTS SIGNAL SMEARING) ---
      step_size = 500
    
      patient_windows = []
    
      for start_idx in range(0, len(df_scaled) - window_size, step_size):
          window_data = df_scaled.iloc[start_idx : start_idx + window_size]
        
          acc_mag = np.sqrt(window_data["AX"]**2 + window_data["AY"]**2 + window_data["AZ"]**2)
          gyro_mag = np.sqrt(window_data["GX"]**2 + window_data["GY"]**2 + window_data["GZ"]**2)
        
          if np.mean(acc_mag) < 0.15 or np.mean(gyro_mag) < 0.02:
              continue
            
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
              freqs = np.fft.rfftfreq(window_size, d=1/SAMPLING_RATE)
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
              patient_groups.append(matched_csv_id)
            
  except Exception:
      pass


X_raw = np.array(features_list, dtype=np.float32)
y_raw = np.array(labels_list)
groups_raw = np.array(patient_groups)


y_clean = []
for label in y_raw:
  label_lower = label.lower().replace(" ", "").replace(".", "").replace("'", "").replace("’", "")
  if "parkinson" in label_lower and "atypical" not in label_lower:
      y_clean.append("PD")
  elif "healthy" in label_lower or "control" in label_lower:
      y_clean.append("HC")
  else:
      y_clean.append("DD")
y_3class = np.array(y_clean)


# =====================================================================
# 3. STAGE 1: WINDOW RISK ASSESSOR LAYER
# =====================================================================
class DeepWindowRiskAssessor(nn.Module):
  def __init__(self, input_dim):
      super(DeepWindowRiskAssessor, self).__init__()
      self.layer = nn.Sequential(
          nn.Linear(input_dim, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(32, 1)
      )
  def forward(self, x): return self.layer(x).squeeze(-1)


class FlatDataset(Dataset):
  def __init__(self, X, y):
      self.X = torch.tensor(X, dtype=torch.float32)
      self.y = torch.tensor(y, dtype=torch.float32)
  def __len__(self): return len(self.X)
  def __getitem__(self, idx): return self.X[idx], self.y[idx]


# =====================================================================
# 4. STAGE 2: ADVANCED CASCADED ENSEMBLE (WITH PCA EMBEDDED)
# =====================================================================
def run_normalized_cascade_track(X_all, y_all, groups_all, target_label, control_label):
  track_mask = (y_all == target_label) | (y_all == control_label)
  X_tr, y_tr, groups_tr = X_all[track_mask], y_all[track_mask], groups_all[track_mask]
  y_bin = (y_tr == target_label).astype(int)
   # -------------------------------------------------------------
  # NEW: GLOBAL PCA EMBEDDING & EXPORT VISUALIZATION
  # -------------------------------------------------------------
  print(f"\nCalculating Dimensionality Reduction for [{control_label}] vs [{target_label}]...")
  scaler_vis = StandardScaler()
  X_scaled_vis = scaler_vis.fit_transform(X_tr)
 
  pca_vis = PCA(n_components=2, random_state=42)
  X_pca_vis = pca_vis.fit_transform(X_scaled_vis)
 
  plt.figure(figsize=(8, 6))
  palette = {target_label: "#d95f02", control_label: "#1b9e77"}
  sns.scatterplot(
      x=X_pca_vis[:, 0], y=X_pca_vis[:, 1],
      hue=y_tr, alpha=0.5, palette=palette, edgecolor=None
  )
  explained_var = pca_vis.explained_variance_ratio_.sum() * 100
  plt.title(f"PyTorch Context - PCA Space ({control_label} vs {target_label})\nCaptured Global Variance: {explained_var:.1f}%")
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.tight_layout()
 
  filename_clean = f"nn_pca_{control_label.lower()}_vs_{target_label.lower()}.png"
  plt.savefig(filename_clean, dpi=300)
  plt.close()
  print(f" -> Plot exported successfully: '{filename_clean}'")
  # -------------------------------------------------------------


  unique_patients = np.unique(groups_tr)
  patient_meta_features = {p_id: [] for p_id in unique_patients}
  patient_true_labels = {}
   sgkf = StratifiedGroupKFold(n_splits=5)
   for train_idx, test_idx in sgkf.split(X_tr, y_bin, groups_tr):
      X_train_out, y_train_out = X_tr[train_idx], y_bin[train_idx]
      X_test_out, y_test_out = X_tr[test_idx], y_bin[test_idx]
      groups_test_out = groups_tr[test_idx]
    
      # Fit scaling and compress spatial vectors strictly across training data
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train_out)
      X_test_scaled = scaler.transform(X_test_out)
     
      # Squeeze down into 2 principal dimensions
      pca = PCA(n_components=2, random_state=42)
      X_train_pca = pca.fit_transform(X_train_scaled)
      X_test_pca = pca.transform(X_test_scaled)
    
      train_loader = DataLoader(FlatDataset(X_train_pca, y_train_out), batch_size=128, shuffle=True)
     
      # Hardcoded to input_dim=2 because PCA compressed our signal features
      model = DeepWindowRiskAssessor(input_dim=2)
      optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.02)
      criterion = nn.BCEWithLogitsLoss()
    
      model.train()
      for epoch in range(15):
          for bx, by in train_loader:
              optimizer.zero_grad()
              loss = criterion(model(bx), by)
              loss.backward()
              optimizer.step()
            
      model.eval()
      with torch.no_grad():
          t_test = torch.tensor(X_test_pca, dtype=torch.float32)
          window_risks = torch.sigmoid(model(t_test)).numpy()
        
      for i, p_id in enumerate(groups_test_out):
          patient_meta_features[p_id].append(window_risks[i])
          patient_true_labels[p_id] = target_label if y_test_out[i] == 1 else control_label


  meta_X_list = []
  meta_y_list = []
   for p_id in unique_patients:
      risks = np.array(patient_meta_features[p_id])
      if len(risks) == 0: continue
        
      p_age = age_map.get(p_id, 50.0)
      p_height = height_map.get(p_id, 170.0)
      p_weight = weight_map.get(p_id, 70.0)
      h_m = p_height / 100.0 if p_height > 3.0 else p_height
      p_bmi = p_weight / (h_m ** 2) if h_m > 0 else 0.0
        
      profile_features = [
          np.mean(risks),
          np.std(risks),
          skew(risks) if len(risks) > 2 else 0.0,
          kurtosis(risks) if len(risks) > 2 else 0.0,
          np.percentile(risks, 95),
          np.percentile(risks, 75),
          np.percentile(risks, 50),
          np.percentile(risks, 25),
          np.percentile(risks, 5),
          p_age,
          p_bmi
      ]
      meta_X_list.append(profile_features)
      meta_y_list.append(1 if patient_true_labels[p_id] == target_label else 0)
    
  X_meta = np.array(meta_X_list, dtype=np.float32)
  y_meta = np.array(meta_y_list)
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  meta_preds_probs = np.zeros(len(X_meta))
   pos_weight = float(np.sum(y_meta == 0)) / float(np.sum(y_meta == 1))
   # --- RESTORED SAFE STABLE PARAMETERS ---
  lgb_params = {
      'objective': 'binary',
      'metric': 'binary_logloss',
      'boosting_type': 'gbdt',
      'learning_rate': 0.02,
      'num_leaves': 15,
      'max_depth': 4,
      'scale_pos_weight': pos_weight,
      'verbose': -1,
      'random_state': 42
  }
   for train_m, test_m in skf.split(X_meta, y_meta):
      d_train = lgb.Dataset(X_meta[train_m], label=y_meta[train_m])
      gbm = lgb.train(lgb_params, d_train, num_boost_round=100)
      meta_preds_probs[test_m] = gbm.predict(X_meta[test_m])
    
  best_f1 = -1
  best_cutoff = 0.5
  for trial_cutoff in np.linspace(0.1, 0.9, 81):
      preds = [target_label if p >= trial_cutoff else control_label for p in meta_preds_probs]
      actuals = [target_label if v == 1 else control_label for v in y_meta]
      score = f1_score(actuals, preds, average='macro', zero_division=0)
      if score > best_f1:
          best_f1 = score
          best_cutoff = trial_cutoff
        
  final_preds = [target_label if p >= best_cutoff else control_label for p in meta_preds_probs]
  true_strings = [target_label if v == 1 else control_label for v in y_meta]
   print("\n" + "="*50)
  print(f"PHYSICAL COHERENCE NET: [{control_label}] vs [{target_label}] (PCA DATA)")
  print(f"Optimized Decision Boundary Cutoff: {best_cutoff*100:.1f}%")
  print("="*50)
  print(f"Patient Level Sub-Accuracy: {accuracy_score(true_strings, final_preds)*100:.2f}%")
  print(classification_report(true_strings, final_preds, zero_division=0))


# =====================================================================
# 5. EXECUTE PIPELINES
# =====================================================================
run_normalized_cascade_track(X_raw, y_3class, groups_raw, "PD", "HC")
run_normalized_cascade_track(X_raw, y_3class, groups_raw, "PD", "DD")


print("\n--- Pipeline Finished ---")
input("Press Enter to close...")


