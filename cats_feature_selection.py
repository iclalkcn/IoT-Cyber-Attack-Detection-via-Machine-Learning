# cats_feature_selection.py
# Amaç: df_smote_balanced.csv dosyasındaki özelliklerden,
# CAT-S (Cellular Automata + Tabu Search) ile en iyi alt kümeyi seçip
# selected_features.csv olarak kaydetmek.

import os
import random
from collections import deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ===================== Kullanıcı Parametreleri ===================== #
DATA_FILE = "df_smote_balanced.csv"  # SMOTE sonrası veri
SCORING = "accuracy"              # 'accuracy' da deneyebilirsin f1_weighted
POP_SIZE = 12                       # CA popülasyon büyüklüğü
MAX_ITERS = 20                      # iterasyon sayısı
TABU_LEN = 50                        # tabu listesi uzunluğu
FEATURE_PENALTY = 0.02               # daha az özellik seçmeyi teşvik (0-0.05 arası deneyebilirsin)
RANDOM_STATE = 42
CV = 2                              # cross-val kat sayısı
N_TREES = 50                        # RF ağaç sayısı
RULE = 110                           # 1D CA kuralı (ör: 30, 90, 110)
# =================================================================== #

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

if not os.path.exists(DATA_FILE):
    raise SystemExit(f" {DATA_FILE} bulunamadı. Önce merge + SMOTE adımını çalıştır.")

print(f" Veri yükleniyor: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)

# X, y ayrımı
y = df["label"].astype(int)
X = df.drop(columns=["label"])

# Kategorik sütunları one-hot'a çevir (RF numerik ister, SMOTE sonrası da güvenli)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    print(f"ℹ Kategorik sütunlar bulundu, one-hot uygulanıyor: {cat_cols}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

feature_names = X.columns.to_list()
n_feats = len(feature_names)
X_np = X.values

print(f" Özellik sayısı: {n_feats}")
print(f"  Parametreler → POP_SIZE={POP_SIZE}, MAX_ITERS={MAX_ITERS}, TABU_LEN={TABU_LEN}, RULE={RULE}")

# ----- Yardımcılar -----
def rule_to_table(rule_num: int):
    """
    1D CA için 3-bit komşuluk -> bit çıktısı tablo (0..7)
    Örn: RULE=110 -> 0b1101110 (msb: 111, lsb: 000 şablonlarına karşılık)
    """
    bits = [(rule_num >> i) & 1 for i in range(8)]
    return bits

RULE_TABLE = rule_to_table(RULE)

def ca_update(mask: np.ndarray) -> np.ndarray:
    """
    1D CA ile tek bir maskeyi güncelle (binary).
    wrap-around (dairesel) komşuluk kullanılır.
    """
    n = len(mask)
    nxt = np.zeros_like(mask)
    for i in range(n):
        left = mask[(i - 1) % n]
        mid  = mask[i]
        right= mask[(i + 1) % n]
        idx = (left << 2) | (mid << 1) | right  # 0..7
        nxt[i] = RULE_TABLE[idx]
    return nxt

def random_mask(k=None):
    """Rastgele maske (en az 1 özellik). İstersen k özellikli seçebilirsin."""
    m = np.zeros(n_feats, dtype=np.uint8)
    if k is None:
        k = np.random.randint(1, max(2, n_feats // 3))
    idx = np.random.choice(n_feats, size=k, replace=False)
    m[idx] = 1
    return m

def hash_mask(mask: np.ndarray) -> str:
    return mask.tobytes().hex()

def fitness(mask: np.ndarray) -> float:
    """Seçilen özelliklerle RF + CV skoru (sparse teşviki için ceza)."""
    if mask.sum() == 0:
        return 0.0
    cols = np.where(mask == 1)[0]
    X_sel = X_np[:, cols]
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    scores = cross_val_score(clf, X_sel, y, cv=CV, scoring=SCORING, n_jobs=-1)
    base = float(scores.mean())
    penalty = FEATURE_PENALTY * (mask.sum() / n_feats)
    return base - penalty

def local_mutation(mask: np.ndarray, flips: int = 1) -> np.ndarray:
    """Küçük yerel değişiklik: birkaç biti çevir."""
    m = mask.copy()
    idxs = np.random.choice(n_feats, size=flips, replace=False)
    m[idxs] ^= 1
    if m.sum() == 0:  # en az 1 özellik
        m[np.random.randint(0, n_feats)] = 1
    return m

# ----- Başlangıç Popülasyonu -----
population = [random_mask() for _ in range(POP_SIZE)]
tabu = deque(maxlen=TABU_LEN)

best_mask = None
best_score = -1.0

# ----- Ana Döngü -----
for it in range(1, MAX_ITERS + 1):
    new_population = []
    print(f"\n İterasyon {it}/{MAX_ITERS}")

    for idx, mask in enumerate(population):
        # CA ile komşu üret
        neighbor = ca_update(mask)

        # Bir miktar rastgele mutasyon ekle (çeşitlilik)
        if np.random.rand() < 0.6:
            neighbor = local_mutation(neighbor, flips=1)
        if np.random.rand() < 0.2:
            neighbor = local_mutation(neighbor, flips=2)

        # Tabu kontrol (görülen çözümleri tekrar etme)
        h = hash_mask(neighbor)
        tries = 0
        while h in tabu and tries < 5:
            neighbor = local_mutation(neighbor, flips=1)
            h = hash_mask(neighbor)
            tries += 1

        # Değerlendir
        s1 = fitness(mask)
        s2 = fitness(neighbor)

        # Daha iyiyse kabul et (hill-climb + CA)
        chosen = neighbor if s2 >= s1 else mask
        new_population.append(chosen)
        tabu.append(hash_mask(chosen))

        # Global en iyi
        if s2 > best_score:
            best_score = s2
            best_mask = neighbor.copy()

    population = new_population
    print(f"   ↳ Şu ana kadarki en iyi skor: {best_score:.4f} | seçilen özellik: {int(best_mask.sum())}/{n_feats}")

# ----- Sonuç -----
if best_mask is None or best_mask.sum() == 0:
    # Güvenlik ağı: en azından en bilgilendirici birkaç özelliği seç
    print("Geçerli bir maske oluşmadı, yedek strateji ile seçim yapılıyor.")
    imp_model = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    imp_model.fit(X_np, y)
    importances = imp_model.feature_importances_
    topk = max(5, n_feats // 10)
    best_idx = np.argsort(importances)[::-1][:topk]
    best_mask = np.zeros(n_feats, dtype=np.uint8)
    best_mask[best_idx] = 1

selected = [feature_names[i] for i in np.where(best_mask == 1)[0]]

pd.Series(selected).to_csv("selected_features.csv", index=False)
print("\n CAT-S tamamlandı.")
print(f"Seçilen özellik sayısı: {len(selected)} / {n_feats}")
print(" Kaydedildi: selected_features.csv")
