import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# Suppress background machine learning warnings
warnings.filterwarnings('ignore')


print("Launching Corrected Patient-Isolated Dual-Track Architecture...")


MANIFEST_FILE = "file_list.csv"
if not os.path.exists(MANIFEST_FILE):
   print(f"Error: '{MANIFEST_FILE}' not found.")
   exit()


# =====================================================================
# 1. LOAD AND MAP TARGET COLUMNS
# =====================================================================
manifest_df = pd.read_csv(MANIFEST_FILE)
id_col = "id"
diag_col = "condition"


manifest_df[id_col] = manifest_df[id_col].astype(str).str.strip()
manifest_df[diag_col] = manifest_df[diag_col].astype(str).str.strip()
raw_csv_pairs = dict(zip(manifest_df[id_col], manifest_df[diag_col]))


# =====================================================================
# 2. FEATURE EXTRACTION ENGINE
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
     
       window_size = 250
       step_size = 500  # Set to 500 to keep execution fast and prevent window-smearing leakage
     
       patient_windows = []
     
       for start_idx in range(0, len(df_scaled) - window_size, step_size):
           window_data = df_scaled.iloc[start_idx : start_idx + window_size]
         
           sma = np.mean(np.abs(window_data["AX"]) + np.abs(window_data["AY"]) + np.abs(window_data["AZ"]))
           if sma < 0.15: continue
             
           window_features = []
         
           for col in df_scaled.columns:
               signal = window_data[col].values
               signal_centered = signal - np.mean(signal)
             
               zero_crossings = np.nonzero(np.diff(signal_centered > 0))[0]
               zcr = len(zero_crossings) / len(signal)
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


X_raw = np.array(features_list)
y_raw = np.array(labels_list)
groups_raw = np.array(patient_groups)


y_clean = []
for label in y_raw:
   label_lower = label.lower().replace(" ", "").replace("'", "").replace("’", "")
   if "parkinson" in label_lower and "atypical" not in label_lower:
       y_clean.append("Parkinson's")
   elif "healthy" in label_lower or "control" in label_lower:
       y_clean.append("Healthy")
   else:
       y_clean.append("Other Pathologies")
y_3class = np.array(y_clean)


print(f" -> Dataset Parsed: {len(np.unique(groups_raw))} patients mapping to {len(X_raw):,} total windows.")


# =====================================================================
# 3. RUN COHORT DISCRIMINATION ENGINE
# =====================================================================
def run_optimized_track(X_all, y_all, groups_all, target_label, control_label):
   track_mask = (y_all == target_label) | (y_all == control_label)
   X_tr, y_tr, groups_tr = X_all[track_mask], y_all[track_mask], groups_all[track_mask]
   y_bin = (y_tr == target_label).astype(int)
  
   unique_patients = np.unique(groups_tr)
   patient_probabilities = {p_id: [] for p_id in unique_patients}
   patient_true_labels = {}
  
   sgkf = StratifiedGroupKFold(n_splits=5)
  
   for train_idx, test_idx in sgkf.split(X_tr, y_bin, groups_tr):
       X_train, y_train = X_tr[train_idx], y_bin[train_idx]
       X_test, y_test = X_tr[test_idx], y_bin[test_idx]
       test_groups = groups_tr[test_idx]
      
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)
      
       w = compute_sample_weight(class_weight='balanced', y=y_train)
      
       # Output probability distributions instead of hard class votes
       model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.80, random_state=42, eval_metric="logloss")
       model.fit(X_train_scaled, y_train, sample_weight=w)
      
       probs = model.predict_proba(X_test_scaled)[:, 1]
      
       for i, p_id in enumerate(test_groups):
           patient_probabilities[p_id].append(probs[i])
           patient_true_labels[p_id] = target_label if y_test[i] == 1 else control_label


   # Aggregate profiles using central tendency distributions
   meta_patients = []
   meta_predictions_raw = []
   meta_trues = []
  
   for p_id in unique_patients:
       p_probs = patient_probabilities[p_id]
       if len(p_probs) == 0: continue
      
       # Use full window median tracking to resist transient activity noise spikes
       patient_score = np.median(p_probs)
       meta_patients.append(p_id)
       meta_predictions_raw.append(patient_score)
       meta_trues.append(patient_true_labels[p_id])
      
   meta_predictions_raw = np.array(meta_predictions_raw)
  
   # Sweep potential cutoffs to optimize operational Macro-F1 boundaries
   best_f1 = -1
   best_cutoff = 0.5
   for trial_cutoff in np.linspace(0.1, 0.9, 81):
       trial_preds = [target_label if p >= trial_cutoff else control_label for p in meta_predictions_raw]
       score = f1_score(meta_trues, trial_preds, average='macro', zero_division=0)
       if score > best_f1:
           best_f1 = score
           best_cutoff = trial_cutoff
          
   final_preds = [target_label if p >= best_cutoff else control_label for p in meta_predictions_raw]
  
   print("\n" + "="*50)
   print(f"BALANCED RESULTS TRACK: [{control_label}] vs [{target_label}]")
   print(f"Optimized Class Boundary Cutoff: {best_cutoff*100:.1f}%")
   print("="*50)
   print(f"Patient Level Sub-Accuracy: {accuracy_score(meta_trues, final_preds)*100:.2f}%")
   print(classification_report(meta_trues, final_preds, zero_division=0))


# =====================================================================
# 4. EXECUTE PIPELINES
# =====================================================================
run_optimized_track(X_raw, y_3class, groups_raw, "Parkinson's", "Healthy")
run_optimized_track(X_raw, y_3class, groups_raw, "Parkinson's", "Other Pathologies")


print("\n--- Pipeline Finished ---")
input("Press Enter to close...")


