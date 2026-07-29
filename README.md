# Predicția prețului mașinilor second-hand

Proiect de regresie care estimează prețul (priceUSD) unei mașini second-hand pe baza caracteristicilor sale (marcă, model, an, kilometraj, tip combustibil, transmisie, tracțiune, segment etc.).

Platformele de vânzare mașini second-hand se confruntă cu o problemă recurentă: vânzătorii stabilesc prețuri fie prea mari (anunțul stagnează), fie prea mici (pierd bani). Acest proiect antrenează un model de regresie care propune un preț orientativ de piață pornind de la caracteristicile tehnice ale mașinii.

Proiectul acoperă întregul flux de lucru pentru o problemă de regresie:

EDA → curățarea datelor → ingineria caracteristicilor → preprocesare → antrenare → evaluare → comparare modele
Setul de date

Fișierul data/cars.csv conține 56.244 anunțuri de mașini second-hand, cu următoarele coloane:

# Descriere Coloane
    make	marca mașinii
    model	modelul mașinii
    priceUSD	variabila țintă — prețul mașinii în USD
    year	anul fabricației
    condition	starea mașinii
    mileage(kilometers)	kilometrajul
    fuel_type	tipul de combustibil
    volume(cm3)	capacitatea motorului
    color	culoarea mașinii
    transmission	tipul transmisiei
    drive_unit	tipul tracțiunii
    segment	segmentul mașinii
    Structura proiectului
    car-price-prediction/

#Structura
├── data/
│   ├── cars.csv                  # date brute
│   ├── cleaned_cars.csv          # date după curățare
│   └── features_data.csv         # date după ingineria caracteristicilor
├── notebooks/
│   └── 01_eda.ipynb              # analiza exploratorie a datelor
├── src/
│   ├── data_cleaning.py          # curățarea datelor
│   ├── feature_engineering.py    # ingineria caracteristicilor
│   ├── data_preprocessing.py     # pipeline de preprocesare (impute, scale, encode)
│   ├── model_training.py         # antrenarea mai multor modele
│   ├── model_evaluation.py       # evaluarea unui model specific
│   └── model_comparison.py       # compararea modelelor și alegerea celui mai bun
├── models/                       # modelele antrenate (.joblib) — vezi Limitări
├── requirements.txt
└── README.md
# Instalare
    bash
    git clone <link-repo>
    cd car-price-prediction
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    Cum se rulează proiectul

# Toate scripturile se rulează din rădăcina proiectului (folosesc căi relative de tipul data/..., models/...).

bash
# 1. Curățarea datelor: data/cars.csv -> data/cleaned_cars.csv
python src/data_cleaning.py

# 2. Ingineria caracteristicilor: data/cleaned_cars.csv -> data/features_data.csv
python src/feature_engineering.py

# 3. Antrenarea modelelor -> salvează câte un .joblib per model în models/
python src/model_training.py

# 4. Evaluarea unui model specific (implicit: Linear Regression)
python src/model_evaluation.py

# 5. Compararea tuturor modelelor din models/ și alegerea celui mai bun
python src/model_comparison.py

# Pentru explorare și vizualizări, deschide notebooks/01_eda.ipynb.

# Fluxul de lucru
1. EDA (01_eda.ipynb)
Verificarea formei setului de date, a tipurilor și a valorilor lipsă (volume(cm3), drive_unit, segment aveau valori lipsă).
Identificarea valorilor imposibile: kilometraje de 0 pe mașini marcate „with mileage", kilometraje peste 5,25 milioane km (recordul mondial cunoscut), ceea ce indică erori de introducere a datelor.
Calculul unei metrici auxiliare, km_per_year (kilometraj / vârsta mașinii), pentru a detecta mai fin înregistrările nerealiste (prea puțini sau prea mulți km pe an).
Analiza corelației vizuale (boxplot) dintre priceUSD și variabilele categorice (drive_unit, transmission, fuel_type, segment).
Identificarea a 87 de rânduri duplicate.
Concluzie: coloana color nu influențează relevant prețul unei mașini second-hand și a fost eliminată.
2. Curățarea datelor (data_cleaning.py)
Eliminarea duplicatelor și a rândurilor cu valori lipsă.
Standardizarea numelor de coloane (mileage(kilometers) → mileage_kilometers, volume(cm3) → volume_cm3 etc.).
Normalizarea valorilor text (strip, lowercase) și a valorilor de tip „missing-like" ("null", "none", "" → NA).
Standardizarea valorilor categorice (make, model, condition, fuel_type, transmission, drive_unit, segment).
Eliminarea coloanei color.
Filtrarea outlierilor de kilometraj folosind km_per_year (sub 100 km/an sau peste 100.000 km/an sunt considerate valori suspecte).
3. Ingineria caracteristicilor (feature_engineering.py)
Caracteristică nouă	Cum se calculează	Motivație
car_age	2019 - year	vârsta influențează direct deprecierea
mileage_per_year	mileage_kilometers / car_age	normalizează kilometrajul relativ la vechime, mai informativ decât kilometrajul brut
engine_volume_liters	volume_cm3 / 1000	scală mai intuitivă și mai apropiată de cum sunt descrise mașinile comercial
4. Preprocesarea (data_preprocessing.py)
Coloane numerice (year, mileage_kilometers, volume_cm3, car_age, mileage_per_year, engine_volume_liters): SimpleImputer (mediana) + StandardScaler.
Coloane categorice (make, model, condition, fuel_type, transmission, drive_unit, segment): SimpleImputer (cea mai frecventă valoare) + OneHotEncoder.
Totul e combinat printr-un ColumnTransformer, integrat ulterior într-un Pipeline sklearn împreună cu regresorul.
5. Antrenare (model_training.py)

Fiecare model e antrenat într-un Pipeline (preprocessor + regressor) pe același split train_test_split(test_size=0.2, random_state=42), pentru comparație corectă.

6. Evaluare și comparare (model_evaluation.py, model_comparison.py)

Metrici raportate: MAE, MSE, RMSE, R². Se afișează și un tabel cu cele mai mari erori de predicție, pentru interpretare.

Modele testate
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Rezultate

Completează acest tabel cu rezultatele reale obținute după rularea model_comparison.py.

Model	MAE	MSE	RMSE	R²
Linear Regression				
Decision Tree Regressor				
Random Forest Regressor				
Gradient Boosting Regressor				

Interpretare MAE: dacă modelul ales are, de exemplu, MAE = 1200, înseamnă că, în medie, predicția greșește cu aproximativ 1200 USD față de prețul real.

# Modelul ales

Completează: care model a avut scorul cel mai bun (vezi models/ și rezultatul model_comparison.py) și de ce — de exemplu, cel mai bun echilibru între R² ridicat și RMSE scăzut, timp de antrenare rezonabil, robustețe la outlieri etc.

Modelul final este salvat în model/car_price_model.joblib.

Limitări și posibile îmbunătățiri
Rândurile cu valori lipsă la drive_unit, segment sau volume_cm3 au fost eliminate în etapa de curățare, în locul imputării — o abordare mai conservatoare care reduce volumul de date, dar simplifică pipeline-ul.
Modelele antrenate (.joblib) pot depăși limitele de dimensiune pentru push direct pe GitHub; pentru reproducere, rulează src/model_training.py local după instalarea dependențelor.
Caracteristici suplimentare posibile, neincluse încă: is_newer_car, is_high_mileage, brand_model (combinație marcă+model).