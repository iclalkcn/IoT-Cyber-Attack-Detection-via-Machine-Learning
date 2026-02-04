Önemli Not: Veri seti boyut limiti (25MB+) nedeniyle GitHub'a yüklenmemiştir. Projeyi çalıştırmak için CIC-IoT-2023 veri setinden "merge-label-and-sample.py" dosyasındaki belirtilen saldırı/normal trafik verilerini indirip ana dizine eklemelisiniz.

Çalıştırma Sırası (Pipeline)

Sistemin çalışması için şu adımları sırasıyla çalıştırın:

merge-label-and-sample.py: Ham verileri birleştirir ve etiketler. --  python merge_label_and_sample.py

rf-importance.py: En kritik 35 özelliği seçer. -- python rf_importance_selection.py

lightgbm-model.py: SMOTE ve LightGBM ile eğitimi başlatır. -- python lightgbm_model.py



🛠️ Teknik Detaylar

Veri Seti: CIC-IoT-2023
Veri Dengeleme: SMOTE
Özellik Seçimi: RF-Importance
Model: LightGBM
Dil: Python
