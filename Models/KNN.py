import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# functie cu ajuatorul careia se incarca datele
# primeste path-ul catre un fisier csv , folderul cu imagini, dimeninea pentru redimensionare (100x100) 
# si un "flag" care semnalaeaza daca sunt sau nu etichete (de exemplu pt datele de test nu avem etichete)

def incarcaDate(cvsPath, imaginePath, dimensiune = (100,100), areEtichete = True):
    date = pd.read_csv(cvsPath)  # citim datele
    imagini = []  # aici vom stoca imaginile
    etichete = [] if areEtichete else None  # iar aici vom stoca etichetele (daca sunt)
     
    for idImg, et in zip(date['image_id'], date['label'] if areEtichete else [None] * len(date)):
        path = os.path.join(imaginePath, f"{idImg}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, dimensiune).astype('float32') / 255.0 # redimensionare + normalizare
            img = cv2.GaussianBlur(img, (5, 5), 0) # pentru reducerea "zgomotului"
            histograma = [] # aici initializam histograma pentru imaginea curenta 
            for culoare in range(3): # avem 3 culori : Blue, Green, Red - pentru fiecare culoare (canal) vom calcula histograma corespunzatoare
                histograma_ = cv2.calcHist([img], [culoare], None, [16], [0, 1]) # histograma pentru culoarea curenta - se lucreaza cu 16 bin
                histograma_ = cv2.normalize(histograma_, histograma_).flatten() # normalizare histograma + transformarea sa in vector unidimensional 
                histograma.extend(histograma_) # pentru culoarea (canalul) curenta adaugam histograma in lista de histograme (in histograma mare)
            imagini.append(histograma) # acum , imaginile vor fi reprezentate prin histograma lor (adica un vector cu 48 de valori - 16 bin x 3 canale)
            if areEtichete:
                etichete.append(et)

    imagini = np.array(imagini)
    return (imagini, np.array(etichete)) if areEtichete else (imagini, date['image_id'].tolist())



imaginiAntrenare, eticheteAntrenare = incarcaDate("/kaggle/input/unibuc/train.csv", "/kaggle/input/unibuc/train")
imaginiValidare, eticheteValidare = incarcaDate("/kaggle/input/unibuc/validation.csv", "/kaggle/input/unibuc/validation")

# pentru valorile lui k am ales sa folosesc numerele din sirul lui Fibonacci
# mentiune: daca numarul din sir este par, am considerat incremenatrea sa (pentru a avea numar impar de vecini)
incercariK = [1, 3, 5, 9, 13, 21, 35, 55, 89, 145] 
acurateteK = [] # aici salvam acuratatea pentru fiecare model (adica pentru fiecare valoare a lui k)
modeleK = [] # salvam moddelele pentru fiecare valoare a lui k



# modelul KNN
# claisificatorul KNN va primi ca parametri:
# numarul de vecini - se iteresaza prin fiecare valoare posibila a lui k din lista
# weight-ul este dat de distanta - adica fiecare vecin va avea o influenta proportionala cu distanta fata de el
# distanta aleasa (in acest caz) a fost diatnta manhattan

for incercareK in incercariK:
    modelK= make_pipeline(
        StandardScaler(), # scalarea vectorilor 
        KNeighborsClassifier(n_neighbors=incercareK, weights='distance', metric='manhattan')
    )

    # in continuare, antrenarea modelului pe datele de train si preditiile pe datele de validare 
    modelK.fit(imaginiAntrenare, eticheteAntrenare)
    predictieK = modelK.predict(imaginiValidare)

    # calcularea acuratetii pentru modelul curent
    acurateteK.append(accuracy_score(eticheteValidare, predictieK))
    modeleK.append(modelK)
    print(f"k = {incercareK}  -- acuratete: {acurateteK[-1]:.4f}")



# selectam modelul care e avut cea mai buna acuratete pentru datele de validare 
acurateteMaxima = np.argmax(acurateteK)
modelMaxim = modeleK[acurateteMaxima]
kMaxim = incercariK[acurateteMaxima]

# vom mai face inca o data predictiile pentru acest model (pe datele de validare) pentru a afisa "statisticile"
predictieKMaxim = modelMaxim.predict(imaginiValidare)

# matricea de confuzie (pentru cel mai bun k)
matrice = confusion_matrix(eticheteValidare, predictieKMaxim)
afisareMatrice = ConfusionMatrixDisplay(confusion_matrix=matrice)
afisareMatrice.plot(cmap="Purples")
plt.title(f"Matrice de confuzie - KNN (k={kMaxim})")
plt.show()

# graficul acuratetii in functie de k
plt.plot(incercariK, acurateteK, marker='x', color="purple")
plt.xlabel("k (numar de vecini)")
plt.ylabel("Acuratete")
plt.title("Acuratete KNN")
plt.grid(True)
plt.show()
