import os
import pandas as pd
import lightgbm as lgb 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lightgbm import LGBMClassifier

# 1) Veriyi yükle
if os.path.exists("df_smote_balanced.csv"):
    df = pd.read_csv("df_smote_balanced.csv")
    print(" SMOTE ile dengelenmiş veri yüklendi: df_smote_balanced.csv")
elif os.path.exists("df_merged_sample.csv"):
    df = pd.read_csv("df_merged_sample.csv")
    print(" Dengesiz örnek veri yüklendi: df_merged_sample.csv")
else:
    raise SystemExit(" Veri dosyası bulunamadı! df_smote_balanced.csv veya df_merged_sample.csv olmalı.")

# 2) X / y ayrımı
X = df.drop(columns=["label"])
y = df["label"]


# 3) CAT-S feature selection varsa uygula
#if os.path.exists("selected_features.csv"):
 #   selected_features = pd.read_csv("selected_features.csv").iloc[:, 0].tolist()
  #  print(f"CAT-S seçimi kullanılıyor: {len(selected_features)} özellik")
   # X = X[selected_features]
#else:
#    print("selected_features.csv bulunamadı, tüm özellikler kullanılacak.")


# 3) Mutual Information (MI) feature selection varsa uygula  YENİİ KISIM
#if os.path.exists("selected_features_mi.csv"):
    #selected_features = pd.read_csv("selected_features_mi.csv").iloc[:, 0].tolist()
if os.path.exists("selected_features_rfimp.csv"):
    selected_features = pd.read_csv("selected_features_rfimp.csv").iloc[:, 0].tolist()

    print(f"MI seçimi kullanılıyor: {len(selected_features)} özellik")
    X = X[selected_features]
else:
    print("selected_features_mi.csv bulunamadı, tüm özellikler kullanılacak.")

# 4) Train / Test bölme
#X_train, X_test, y_train, y_test = train_test_split(
 #   X,
  #  y,
   # test_size=0.2,
    #random_state=42,
    #stratify=y
#)
# 4) Train / Test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_classes = y.nunique()
print(f"\nSınıf sayısı: {num_classes}")
print("Sınıflar:", sorted(y.unique()))

# 5) LightGBM modelini tanımla
#lgbm_model = LGBMClassifier(
 #   objective="multiclass",
  #  num_class=num_classes,
   # n_estimators=1500,
    #learning_rate=0.05,
    #num_leaves=384,
    #max_depth=-1,
   # min_child_samples=10,
    #subsample=0.8,
    #colsample_bytree=0.8,
    #n_jobs=-1,
    #random_state=42
#)
model = LGBMClassifier(
    n_estimators=500, #toplam ağaç sayısı,
    learning_rate=0.05,#her ağacın modele katkı sayısı
    max_depth=8,#
    num_leaves=64,
    subsample=0.8,#bir ağacın eğitilirken kullanılırken eğitim oranı
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,#rastgelelik 
    n_jobs=-1
)
#####yeni kısım


# 6) Eğit
evals_result = {}

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric="multi_logloss",
    callbacks=[lgb.record_evaluation(evals_result)]
)

# 7) Tahmin ve değerlendirme
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n Model Sonuçları (LightGBM)")
print("------------------------------")
print("Doğruluk (Accuracy):", round(acc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import matplotlib.pyplot as plt

# Loss değerlerini al
train_loss = evals_result["training"]["multi_logloss"]
val_loss = evals_result["valid_1"]["multi_logloss"]

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Multi-class Log Loss")
plt.title("LightGBM Eğitim Süreci (Loss vs Iteration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Feature sayıları ve elde edilen accuracy değerleri
feature_counts = [20, 25, 30, 35]
accuracies = [0.9723, 0.9709, 0.9754, 0.9761]  # senin elde ettiklerin

plt.figure(figsize=(7,5))
plt.plot(feature_counts, accuracies, marker='o')
plt.xlabel("Feature Sayısı")
plt.ylabel("Accuracy")
plt.title("Feature Sayısına Göre Model Doğruluğu (LightGBM)")
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix hesapla
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix (LightGBM – Final Model)")
plt.tight_layout()
plt.show()
