import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Veri yükle
df = pd.read_csv("df_smote_balanced.csv")
X = df.drop(columns=["label"])
y = df["label"]

# RF ile importance hesapla
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)

top_k = 35 # modelin kullanacağı özellik sayısı 
selected = importances.sort_values(ascending=False).head(top_k).index

pd.Series(selected).to_csv("selected_features_rfimp.csv", index=False)
print("selected_features_rfimp.csv oluşturuldu. Özellik sayısı:", len(selected))
