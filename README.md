# Segmenting Retinal Blood Vessels with CNN

*Created for Informatics in Medicine classes*
  
Cel: Przekształcenie obrazu oryginalnego w obraz podobny do maski eksperta, eksponując jedynie naczynia krwionośne:

![Original Image](https://github.com/user-attachments/assets/5f971a8c-e773-4e19-9652-0154cec87b09)

Do projektu wykorzystano bazy HRF oraz DRIVE.

## 1. Transformacje morfologiczne i progowanie

Zadanie segmentacji, które sprowadza się do binarnej klasyfikacji każdego piksela, przeprowadzono za pomocą różnych technik, z których pierwszą było progowanie. Proces obejmował kolejno:

- usunięcie tła
- zastosowanie filtru gamma
- konwersję do skali szarości
- wyostrzenie obrazu
- optymalizację przy użyciu CLAHE
- użycie filtru bilateralnego

Na tym etapie, poprzedzającym segmentację, przykładowy obraz wygląda następująco:

![Preprocessed Image](https://github.com/user-attachments/assets/7197eee6-be71-4593-9ca5-53e717cdc85f)

Segmentacja poprzez progowanie, z ustawianym ręcznie (eksperymentalnie) progiem. Rezultat dla przykładowego obrazu:

![Thresholding Result](https://github.com/user-attachments/assets/da7034c8-dc8c-4aff-9da2-ddf8ad5935c1)

**Wyniki klasyfikacji:**  
Accuracy: 0.8750  
Precision: 0.2081  
Recall: 0.5199

Wynikowy obraz cechuje dość dużą dokładność dla większych (grubszych) naczyń krwionośnych, jednak zauważalny jest duży szum w jaśniejszych obszarach zdjęcia i trudności z wykryciem cienkich naczyń.

## 2. Proste modele klasyfikacyjne

Drugim etapem było przetestowanie prostych modeli: DecisionTree, kNN, RandomForest oraz MLP z wykorzystaniem własnej ekstrakcji cech. Cechy obejmowały:

- jasność piksela
- kontrast
- średnią z otoczenia (ramka 5x5)
- odchylenie standardowe

Należy zwrócić uwagę na dużą dysproporcję klas (support), co istotnie wpływa na proces uczenia. Aby zrównoważyć klasy, ekstrachowano informacje dla 20% pikseli czarnych (klasa 'brak naczynia').

Modele cechowały przeciętne wyniki, najlepiej sprawdził się model MLP (z dwoma warstwami ukrytymi i 1000 iteracji).

- **Wielkość zbioru treningowego:** 36 obrazów
- **Wielkość zbioru testowego:** 9 obrazów

Wyniki modelu MLP dla oryginalnych zdjęć (bez przetwarzania wstępnego):

![MLP Original](https://github.com/user-attachments/assets/7e85ee92-3b7f-4dfd-80bc-99b530e943ac)

Wyniki dla tego samego modelu MLP, wytrenowanego na zbiorze zdjęć po przetworzeniu obrazów (z podejścia 1.):

![MLP Preprocessed](https://github.com/user-attachments/assets/f701c9ba-16ef-4f72-bf24-c4eafe59cf17)

Jak prezentują wyniki, wstępne przetworzenie obrazów treningowych istotnie podniosło miary klasyfikacji.

## 3. Sieć konwolucyjna typu U-Net

Trzecią metodą była implementacja sieci konwolucyjnej na kształt U-Netu. Model liczy 200 tys. parametrów, 12 warstw konwolucyjnych, 1 warstwę Max Pooling, 1 warstwę transponowanej konwolucji oraz 1 warstwę konkatenacji. Warstwa wyjściowa posiada funkcję sigmoidalną, co pozwala przekształcić output modelu na prawdopodobieństwo przynależności każdego piksela do klasy świadczącej o obecności naczynia.

Obrazy z bazy HRF zostały podzielone na zbiór treningowy (36) i testowy (9). Każdy z nich przeszedł przez redukcję rozdzielczości w celu późniejszego zwiększenia wydajności treningu i inferencji.

Trening modelu obejmował 35 epok, po których model dobrze dopasował się do danych (accuracy: 0.9479, loss: 0.1196)

Predykcja na zbiorze walidacyjnym skutecznie określa wartość każdego piksela w zakresie 0-1 (oznaczając prawdopodobieństwo):

![Validation Output](https://github.com/user-attachments/assets/c6dbe9ed-c114-4179-8bbc-266176bc8e63)

Przykładowa segmentacja wykonana przez predykcję modelu:

![Segmentation Result](https://github.com/user-attachments/assets/c936ba2a-0309-44c9-8f9d-72a0007569d6)

Wyniki klasyfikacji na zbiorze walidacyjnym (9 obrazów):

![Validation Results](https://github.com/user-attachments/assets/f029e413-cb64-46ef-939a-db5c465b102c)

Wyniki klasyfikacji można uznać za zadowalające. Model samodzielnie wykrywa obszar oka, eliminując skutecznie jego problematyczny obwód z klasy naczyń, a także uwidacznia mniejsze (cieńsze) naczynia krwionośne. Pozostały minimalny szum w ich obszarze może być spowodowany przede wszystkim zmniejszeniem rozdzielczości oryginalnych obrazów na wejściu modelu oraz zastosowaniem zbyt małej ilości filtrów.

## 4. Testy na bazie DRIVE

Model wytrenowany na bazie HRF został także przetestowany na 20 zdjęciach z bazy DRIVE.

Przykładowy output:

![DRIVE Output](https://github.com/user-attachments/assets/3722ad26-0f02-41f3-8953-9b0b82d8082f)

Wyniki klasyfikacji (20 obrazów testowych):

![DRIVE Results](https://github.com/user-attachments/assets/e4fb467a-6bc1-4438-a46c-d1db3257527c)

Jak można się było spodziewać, model wytrenowany wcześniej na bazie HRF i testowany na bazie DRIVE cechuje mniejsza skuteczność. Obrazy w bazie DRIVE cechują się mniejszą rozdzielczością (oraz szczegółowością ręcznie oznaczanych masek). Są one także bardziej zróżnicowane pod względem jasności i kontrastu niż obrazy w bazie HRF. Co za tym idzie, ciężko ustalić uniwersalny próg (inny niż 0.5) dla wszystkich testowanych zdjęć. Model dalej skutecznie identyfikuje większość naczyń, jednak nie radzi sobie z tzw. obwodem oka na zdjęciu.

### Trening i Walidacja na Bazie DRIVE

Wytrenowano kolejny model w oparciu o 15 zdjęć z bazy DRIVE. Początkowo trening liczył 30 epok, ustawienie batch_size=5 i posiadał domyślną pozostałą konfigurację. Jednak wyniki nie były zadowalające:

![Initial Training Results](https://github.com/user-attachments/assets/6fa2b3f3-3d92-4ed1-ab7f-81fe04778667)

Z uwagi na małą liczność danych dokonano prostej augmentacji - poprzez lustrzane odbicie każdego zdjęcia podwojono liczność zbioru treningowego do 30 obrazów. Zastosowano także niwelowanie learning_rate w trakcie uczenia oraz early_stopping. Po tych zmianach wyniki uległy istotnej poprawie:

Wyniki treningu dla 25 epoki, dla której zachowane zostały wagi:

![Augmented Training Results](https://github.com/user-attachments/assets/7b4b4bb7-6587-4b0c-b59f-e5c8643fca16)

Przykładowy output:

![Augmented Output](https://github.com/user-attachments/assets/0d776b75-8c1d-4d5d-8199-b0d225920870)

Wyniki klasyfikacji (5 zdjęć):

![Augmented Results](https://github.com/user-attachments/assets/97421015-07fd-4739-86f8-0366cdec1fbd)

Po treningu na bazie DRIVE, model nauczył się eliminować obwód oka z klasy naczyń, jednak zdolności identyfikacji cieńszych naczyń zostały ograniczone. Zaimplementowana architektura znacznie lepiej spisuje się na bazie HRF, gdzie model osiągnął 96% dokładności. Wyniki uzyskane dla obu baz danych z obrazami podkreślają kluczową rolę bogatego i poprawnego dataset'u w tworzeniu efektywnych modeli uczenia maszynowego.
