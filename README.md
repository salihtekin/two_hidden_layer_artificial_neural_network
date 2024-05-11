# two_hidden_layer_artificial_neural_networkü

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/1c09a6d1-06e7-4440-a870-d5aa96b650e8)

	İlk adım olarak kütüphaneler yüklenir. Yukarıdaki kütüphanelerden;
	Satır1 -  ‘sklearn.datasets’ kütüphanesinden ‘load_breast_cancer’ fonksiyonu ile açık kaynak olan “ Breast Cancer Wisconsin” veri seti yüklenmiştir.
	Satır2 - Veri setini eğitim ve test kümelerine bölmek için ‘train_test_split’ fonksiyonunu yüklüyoruz.
	Satır3 - Verileri ölçeklendirmek için ‘SatndartScaler’ yüklendi. Bu kütüphane verilerin özelliklerinin ölçeklendirilmesine ve ortalaması sıfır,standart sapması bir olacak şekilde dönüştürülmesine olanak sağlar.
	Satır4 - Çok katmalı yapay sinir ağı modelini (MLP) oluşturmak için ‘MLPClassifier’ kütüphanesi yüklenir.
	Satır5 – Modelin başarım durumunu ölçmek için doğruluk,duyarlılık ve özgüllük metriklerini hesaplayan kütüphane eklenir.
 
![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/9796997d-a8d5-4d27-b8d7-efe2b0364ddd)

	İkinci adım , “Breast Cancer Wisconsin” veri seti çağrıp X değişkenine özelliklerini Y değişkenine de hedef değişkeni atanır.

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/11b49fbd-1a74-428e-89e5-0f867673b37d)

	Üçüncü adım, veri setini eğitim ve test kümelerine bölmek için ‘train_test_split’ fonksiyonu kullanılır. Elimizde bulunana verilerin %80’i eğitim kümesine %20’side test kümesi oalrak belirlenir.

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/8014d982-3a48-47e6-8ffa-fe57ee67f1af)

	Dördüncü adım, özellikleri ölçeklindirmek için ‘Standart Scaler’ kullanılmıştır. Öncelikle ‘fit_transform’ yöntemini eğitim kümesine uygulayarak eğitim işlemi yapılılır ve dönüştürülür. Sonra aynı dönüşüm test verilerine uygulamak için ‘transform’ yöntemi kullanılmıştır.

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/d9e81729-3c0b-49f8-8367-9ebe0be16d3e)

	Beşinci olarak, iki gizli katman ve 100 nörona sahip MLP modeli oluşturulmuştur. “activation=’relu’” ile ReLu aktivasyon fonksiyonunu, ‘random_state=42’ ile tekraralanabilir sonuçları elde etmek için rastgele durumu ve ‘max_iter=1000’ ile maksimum iterasyon sayısı belirlenir. Ve daha sonra model eğitilir.

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/b3a009ed-46c6-4f1b-a293-85183da1f172)

	Altıncı olarak, İki gizli katmana ve 100 nörona sahip bir MLP modeli oluşturulur. Ayrıca, tol=1e-4 ile optimizasyon algoritmasının tolere edilebilir hata miktarını belirlenir. Ve daha sonra bu model eğitilir.

![image](https://github.com/salihtekin/two_hidden_layer_artificial_neural_network/assets/63247968/84fec75c-1df0-4b92-b2c6-c1662e4e431e)

	İki gizli katmana ve 100 nörona sahip bir MLP modeli oluşturulmuştur. early_stopping=True ile erken durdurmayı kullanarak, eğitim sırasında doğruluk skorunun artmaya başlaması durumunda eğitim durdurulmaya ayarlanmıştır. Ve daha sonra bu model eğitilmiştir.

# Üç ayrı model kullanılma nedeni;
**Model 1:** Daha fazla iterasyon kullanmak.
Bu model, çok katmanlı bir yapay sinir ağı (MLP) oluşturur ve maksimum iterasyon sayısını (max_iter=1000) artırarak daha fazla iterasyon gerçekleştirir. Bu, modelin daha uzun süre eğitilmesini sağlar ve eğitimin istenen sonuca ulaşması ( konverjans) elde edilene kadar daha fazla eğitim yapmasını sağlar.

**Model 2:** Tolerans ayarlamak.
Bu model de aynı şekilde MLP kullanır, ancak tol=1e-4 parametresiyle, optimizasyon algoritmasının tolere edilebilir hata miktarını belirler. Bu, eğitim sırasında modelin ne kadar hatayı kabul edeceğini ayarlar. Daha düşük bir tolerans değeri, daha hassas bir eğitim süreci sağlar.

**Model 3:** Early stopping kullanmak.
Bu model de aynı şekilde MLP kullanır, ancak early_stopping=True parametresiyle, eğitim sırasında erken durdurmayı (early stopping) etkinleştirir. Bu, eğitim sırasında modelin performansının artmaya başladığı anda eğitimi durdurarak aşırı uyum (overfitting) riskini azaltır.
Bu üç model, farklı optimizasyon türlerini kullanarak aynı problemi çözmeye çalışır. İlk model daha uzun süre eğitilirken, ikinci model daha hassas bir eğitim yapar ve üçüncü model aşırı uyum riskini azaltmak için erken durdurma stratejisini kullanır. Bu, farklı eğitim yöntemlerinin performansının karşılaştırılmasını sağlar.

Model 1 ve Model 2'nin doğruluk (accuracy), duyarlılık (recall) ve özgüllük (precision) metriklerinin tamamen aynı olması, bu iki modelin aynı performansı sergilediğini gösterir. Bu, tolerans değerini ayarlamakla (Model 2'de kullanılan tol parametresi) modelin performansında belirgin bir değişiklik olmadığını gösterir.
Model 3'ün doğruluk ve özgüllük metrikleri diğer iki modele kıyasla biraz daha yüksektir. Özellikle, özgüllük metriği (precision) Model 3'te daha yüksektir. Bu, Model 3'ün erken durdurma stratejisini (early stopping) kullanarak daha iyi bir aşırı uyum kontrolü sağladığını ve bu nedenle test verilerinde daha iyi performans gösterdiğini gösterir.
Sonuç olarak, Model 3, erken durdurma stratejisi kullanarak diğerlerine kıyasla biraz daha iyi performans gösterirken, Model 1 ve Model 2 aynı düzeyde performans sergilemektedir. Bu sonuçlar, farklı model ayarlarının performansı nasıl etkileyebileceğini analiz etmemizi sağlar.

