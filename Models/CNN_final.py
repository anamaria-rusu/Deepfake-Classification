import os 
import cv2 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.losses import CategoricalCrossentropy


# functie cu ajuatorul careia se incarca datele
# primeste path-ul catre un fisier csv , folderul cu imagini, dimeninea pentru redimensionare (100x100) 
# si un "flag" care semnalaeaza daca sunt sau nu etichete (de exemplu pt datele de test nu avem etichete)

def incarcaDate(csvPath, imaginePath, dimensiune=(100, 100), areEtichete=True):
    date = pd.read_csv(csvPath) # citim datele 
    imagini = [] # aici vom stoca imaginile
    etichete = [] if areEtichete else None # iar aici vom stoca etichtele (daca sunt)
    for idImg , et in zip(date['image_id'], date['label'] if areEtichete else [None] * len(date)):
        path = os.path.join(imaginePath, f"{idImg}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, dimensiune).astype('float32') / 255.0 # redimensionare + normalizare 
            imagini.append(img)
            if areEtichete:
                etichete.append(et)
    imagini = np.array(imagini)
    return (imagini, np.array(etichete)) if areEtichete else (imagini, date['image_id'].tolist())




# incarcam(citim) datele : de antrenare(train), validare(validation) si test
imaginiAntrenare, eticheteAntrenare = incarcaDate("/kaggle/input/unibuc/train.csv", "/kaggle/input/unibuc/train")
imaginiValidare, eticheteValidare = incarcaDate("/kaggle/input/unibuc/validation.csv", "/kaggle/input/unibuc/validation")
imaginiTest, idTest = incarcaDate("/kaggle/input/unibuc/test.csv", "/kaggle/input/unibuc/test", areEtichete=False)


# convertim etichetele in formatul one-hot encoding (adica in format de vectori binari)
# de exemplu eticheta 3 va fi convertita in [0, 0, 0, 1, 0]
eticheteAntrenare = to_categorical(eticheteAntrenare, num_classes=5)  
eticheteValidare = to_categorical(eticheteValidare, num_classes=5)  



# functie pentru augumentarea datelor 
def augumentare(img, ech):
    img = tf.image.random_flip_left_right(img)  # flip 
    img = tf.image.random_brightness(img, 0.2)  # ajustare luminozitate
    img = tf.image.random_contrast(img, 0.6, 1.4)  # ajustare contrast
    img = tf.image.resize_with_crop_or_pad(img, 110, 110)  # redimensionare
    img = tf.image.random_crop(img, size=[100, 100, 3])  # decupare 
    return img, ech


# crearea dataset-ului de antrenare (fiecare element va fi un tuplu (imagine, eticheta)) 
# aplicarea augumentarii, amestecarea datelor (pentru a evita invatarea unei ordini anume)
# impartirea in batch-uri 

dateAntrenare = tf.data.Dataset.from_tensor_slices((imaginiAntrenare, eticheteAntrenare))
dateAntrenare = dateAntrenare.map(augumentare, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
dateValidare = tf.data.Dataset.from_tensor_slices((imaginiValidare, eticheteValidare)).batch(64).prefetch(tf.data.AUTOTUNE)
dateTest = tf.data.Dataset.from_tensor_slices(imaginiTest).batch(64).prefetch(tf.data.AUTOTUNE)


# scheduler CosineDecayRestarts pentru rata de invatare 
scheduler = CosineDecayRestarts(
    initial_learning_rate=1e-3,  # rata de invatare initiala (0.001 = 1e-3)
    first_decay_steps=10 * 196, # 196 pasi epoca / 10 epoci 
    t_mul=2.0,  # factorul de multiplicare - fiecare ciclu e mai lung de 2 ori decat precedentul
    m_mul=0.9,  # factorul de multiplicare pentru rata de invatare - fiecare ciclu nou are 0.9 din val maxima anterioara 
    alpha=1e-4  # val minima a ratei de inv (0.0001 = 1e-4)
)


# se opreste antrenarea daca pierderea pe setul de validare nu se imbunatateste timp de 10 epoci
earlyStopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,  # asteptam 10 epoci 
    restore_best_weights=True  # restauram cele mai bune weights
)


# modeulul CNN
# este impartiti in 3 blocuri  
# fiecare are doua convolutii 3x3 cu activare ReLU si rgularizare L2
# dupa fiecare convolutie se va face o normalizare batch
# scrt (scurtatura) - sare peste 2 convolutii


input = tf.keras.Input(shape=(100, 100, 3))  
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input)  
x = layers.BatchNormalization()(x)  
scrt = x  
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x) 
x = layers.BatchNormalization()(x) 
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)  
x = layers.BatchNormalization()(x)  
x = layers.Add()([x, scrt]) 
x = layers.MaxPooling2D()(x) 

scrt = layers.Conv2D(64, (1, 1), padding='same')(x) 
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x) 
x = layers.BatchNormalization()(x) 
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x) 
x = layers.BatchNormalization()(x) 
x = layers.Add()([x, scrt])  
x = layers.MaxPooling2D()(x)  

scrt = layers.Conv2D(128, (1, 1), padding='same')(x) 
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x) 
x = layers.BatchNormalization()(x)  
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)  
x = layers.BatchNormalization()(x) 
x = layers.Add()([x, scrt])  
x = layers.GlobalAveragePooling2D()(x)  

x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x) 
x = layers.Dropout(0.5)(x) 
output = layers.Dense(5, activation='softmax')(x)  

model = tf.keras.Model(input, output)  


# compilam modelul
model.compile(
    optimizer=Adam(learning_rate=scheduler), # optimizer Adam cu scheduler
    loss=CategoricalCrossentropy(label_smoothing=0.05), 
    metrics=['accuracy']  
)

# antrenam modelul
istoric = model.fit(
    dateAntrenare,
    validation_data=dateValidare,
    epochs=60,
    callbacks=[earlyStopping],
    verbose=1
)

# evaluare pe vallidare
pierdere , acuratete = model.evaluate(dateValidare, verbose=0)
print(f"Loss pe setul de validare: {pierdere:.4f}, Acuratete: {acuratete:.4f}")

etichetePred = np.argmax(model.predict(dateValidare), axis=1)
eticheteCorect = np.argmax(eticheteValidare, axis=1)


# matricea de confuzie
matrice = confusion_matrix(eticheteCorect, etichetePred)
afisare = ConfusionMatrixDisplay(confusion_matrix=matrice)
afisare.plot(cmap='Greens')
plt.title("Matricea de confuzie: ")
plt.show()

# plotarea pierderii si acuratetii in timpul antrenarii
plt.plot(istoric.history['accuracy'], label='Acuratete antrenare')
plt.plot(istoric.history['val_accuracy'], label='Acuratete validare')
plt.xlabel('Epoci')
plt.ylabel('Acuratete')
plt.legend()
plt.title("Evuloutie")
plt.grid(True)
plt.show()


# predicii pe datele de test + creare fisier csv pentru submit
predictiiTest = model.predict(dateTest)
eticheteTest = np.argmax(predictiiTest, axis=1)

submit = pd.DataFrame({
    'image_id': idTest,
    'label': eticheteTest
})

submit.to_csv('sub.csv', index=False)
