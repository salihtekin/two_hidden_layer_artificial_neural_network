
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Veri setini yükleme
kanser = load_breast_cancer()
X = kanser.data
y = kanser.target

# Veri kümesini eğitim ve test kümelerine ayırma
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendirme
olcekleyici = StandardScaler()
X_egitim_olcekle = olcekleyici.fit_transform(X_egitim)
X_test_olcekle = olcekleyici.transform(X_test)

# Model 1: Daha fazla iterasyon kullanarak
model_1 = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', random_state=42, max_iter=1000)
model_1.fit(X_egitim_olcekle, y_egitim)

# Model 2: Toleransı ayarlayarak
model_2 = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', random_state=42, tol=1e-4, max_iter=1000)
model_2.fit(X_egitim_olcekle, y_egitim)

# Model 3: Early stopping kullanarak
model_3 = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', random_state=42, early_stopping=True, max_iter=1000)
model_3.fit(X_egitim_olcekle, y_egitim)

# Tahminleri yapma ve başarı metriklerini hesaplama
for i, model in enumerate([model_1, model_2, model_3], start=1):
    y_tahmin = model.predict(X_test_olcekle)
    dogrulukDegeri = accuracy_score(y_test, y_tahmin)
    duyarlilikDegeri = recall_score(y_test, y_tahmin)
    özgüllükDegeri = precision_score(y_test, y_tahmin)
    print(f"Model {i}:")
    print("Doğruluk (Accuracy): {:.2f}".format(dogrulukDegeri))
    print("Duyarlılık (Recall): {:.2f}".format(duyarlilikDegeri))
    print("Özgüllük (Precision): {:.2f}".format(özgüllükDegeri))
