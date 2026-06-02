import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

def extract_sensor_features(signal_3d, sampling_rate=100.0):
    """ Extracts statistical and frequency features from a 3D sensor component (N, 3) """
    features = {}
    x, y, z = signal_3d[:, 0], signal_3d[:, 1], signal_3d[:, 2]
    
    # Time-Domain metrics
    features['mean_x'] = np.mean(x)
    features['std_x'] = np.std(x)
    features['std_y'] = np.std(y)
    features['std_z'] = np.std(z)
    
    # Overall movement magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    features['mag_mean'] = np.mean(mag)
    features['mag_std'] = np.std(mag)
    features['sma'] = np.mean(np.abs(x) + np.abs(y) + np.abs(z))
    
    # Frequency-Domain metrics (FFT) for isolating physical tremor
    n_samples = len(mag)
    if n_samples > 1:
        fft_vals = np.abs(rfft(mag))
        fft_freqs = rfftfreq(n_samples, d=1.0/sampling_rate)
        dom_idx = np.argmax(fft_vals[1:]) + 1
        features['dominant_freq'] = fft_freqs[dom_idx]
        
        # Parkinsonian muscle tremor band (3 to 7 Hz) energy
        tremor_mask = (fft_freqs >= 3.0) & (fft_freqs <= 7.0)
        features['tremor_energy'] = np.sum(fft_vals[tremor_mask]) if np.any(tremor_mask) else 0.0
    else:
        features['dominant_freq'] = 0.0
        features['tremor_energy'] = 0.0
        
    return features

# Paths adjusted exactly to your directories
DEMOGRAPHICS_PATH = "/Users/peta_bread/Downloads/PADS_Data/preprocessed/file_list.csv" 
RAW_DATA_DIR = "/Users/peta_bread/Downloads/PADS_Data"

# Load demographics and build a lookup for the actual text condition ('Healthy', 'PD', etc.)
demo_df = pd.read_csv(DEMOGRAPHICS_PATH)
demo_df['id_str'] = demo_df['id'].astype(str).str.zfill(3)

# Build lookups for both the text condition string and the patient ID
condition_lookup = dict(zip(demo_df['id_str'], demo_df['condition']))

print(f"Loaded diagnostics mapping for {len(condition_lookup)} participants.")

# Find raw binary extensions (.bin, .dat, .raw) inside PADS_Data
all_binary_files = []
for ext in ['*.bin', '*.dat', '*.raw']:
    all_binary_files.extend(glob.glob(os.path.join(RAW_DATA_DIR, "**", ext), recursive=True))

print(f"Found {len(all_binary_files)} raw binary sensor streams to process...")

dataset_rows = []

for full_file_path in all_binary_files:
    filename = os.path.basename(full_file_path)
    id_match = re.search(r'\d+', filename)
    if not id_match:
        continue
        
    pid = id_match.group(0).zfill(3)
    if pid not in condition_lookup:
        continue
        
    raw_condition = str(condition_lookup[pid]).strip()

    try:
        with open(full_file_path, 'rb') as binary_file:
            raw_data = np.fromfile(binary_file, dtype=np.float32)
            raw_matrix = raw_data.reshape(-1, 6) # Multi-channel Accel [0:3] and Gyro [3:6]
            
            acc_features = extract_sensor_features(raw_matrix[:, 0:3])
            gyr_features = extract_sensor_features(raw_matrix[:, 3:6])
            
            combined_row = {}
            for k, v in acc_features.items(): combined_row[f"acc_{k}"] = v
            for k, v in gyr_features.items(): combined_row[f"gyr_{k}"] = v
                
            combined_row['participant_id'] = pid
            combined_row['condition'] = raw_condition # Save raw group string ('Healthy', 'PD', or 'DD')
            dataset_rows.append(combined_row)
    except:
        pass # Skip malformed arrays safely

if dataset_rows:
    feature_df = pd.DataFrame(dataset_rows)
    output_path = '/Users/peta_bread/Downloads/PADS_ML/pads_extracted_features.csv'
    feature_df.to_csv(output_path, index=False)
    print(f"Success! Features compiled and saved to '{output_path}'")
else:
    print("Extraction failed. No files found or parsed.")