import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np 

# ==========================
#   DOSYA LİSTESİ
# ==========================

ATTACK_FILES = {
    "DDoS-ICMP_Flood.pcap.csv": 1,   # ICMP Flood
    "DDoS-UDP_Flood.pcap.csv": 2,    # UDP Flood
    "PortScan.pcap.csv": 3,          # Port Scan
    "DoS-SYN_Flood.pcap.csv": 4      # SYN Flood
}

BENIGN_FILES = [
    "BenignTraffic3.pcap.csv"
]

dfs = []

# ==========================
#   SALDIRI DOSYALARI
# ==========================
for fname, label_id in ATTACK_FILES.items():
    if os.path.exists(fname):
        print(f"[OK] Attack file loaded: {fname}")
        df = pd.read_csv(fname)
        df["label"] = label_id
        dfs.append(df)
    else:
        print(f"[!] WARNING: {fname} not found!")

# ==========================
#   BENIGN DOSYALARI
# ==========================
for fname in BENIGN_FILES:
    if os.path.exists(fname):
        print(f"[OK] Benign file loaded: {fname}")
        df = pd.read_csv(fname)
        df["label"] = 0
        dfs.append(df)
    else:
        print(f"[!] WARNING: {fname} not found!")

# Hiç veri yoksa çık
if not dfs:
    raise SystemExit("No CSV files found! Check dataset directory.")

# Hepsini birleştir
df_all = pd.concat(dfs, ignore_index=True)
print("\nTotal dataset shape:", df_all.shape)

# Örnekleme
sample_n = min(20000, len(df_all))
df_sample = df_all.sample(sample_n, random_state=42)
print("Sampled size:", df_sample.shape)
print("\nKOLON ADLARI:")#yeni ekledik
print(df_sample.columns.tolist())# yeni ekledik


cols = df_sample.columns

# Paket hızı
if "Rate" in df_sample.columns:
    df_sample["log_rate"] = np.log1p(df_sample["Rate"])

# Toplam boyutun normalize edilmiş hali
if "Tot size" in df_sample.columns and "IAT" in df_sample.columns:
    df_sample["size_per_time"] = df_sample["Tot size"] / (df_sample["IAT"] + 1)

# Bayrak yoğunluğu (flag davranışı saldırılarda ayırt edici)
flag_cols = [c for c in df_sample.columns if "flag number" in c]
if flag_cols:
    df_sample["flag_sum"] = df_sample[flag_cols].sum(axis=1)

# X / y ayır
X = df_sample.drop(columns=['label'])
y = df_sample['label']

# One-hot kategorik alanlar yeni ekledik
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    print("One-hot encoding:", cat_cols)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE (multi-class destekler)
print("\nOriginal class distribution:")
print(y.value_counts())

smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_resampled).value_counts())

# Kaydet
X_res_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df = pd.concat([X_res_df, pd.Series(y_resampled, name="label")], axis=1)

balanced_df.to_csv("df_smote_balanced.csv", index=False)
print("\n[OK] df_smote_balanced.csv created successfully.")



