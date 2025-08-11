import os 
import cv2 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# functie pentru incarcarea datelor
def incarcaDate(csvPath, imaginePath, dimensiune=(100, 100), areEtichete=True):
    date = pd.read_csv(csvPath)
    imagini = []
    etichete = [] if areEtichete else None
    for idImg , et in zip(date['image_id'], date['label'] if areEtichete else [None] * len(date)):
        path = os.path.join(imaginePath, f"{idImg}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, dimensiune).astype('float32') / 255.0
            imagini.append(img)
            if areEtichete:
                etichete.append(et)
    imagini = np.array(imagini)
    return (imagini, np.array(etichete)) if areEtichete else (imagini, date['image_id'].tolist())

# incarcarea datelor
imaginiAntrenare, eticheteAntrenare = incarcaDate("/kaggle/input/unibuc/train.csv", "/kaggle/input/unibuc/train")
imaginiValidare, eticheteValidare = incarcaDate("/kaggle/input/unibuc/validation.csv", "/kaggle/input/unibuc/validation")
imaginiTest, idTest = incarcaDate("/kaggle/input/unibuc/test.csv", "/kaggle/input/unibuc/test", areEtichete=False)

# convertirea etichetelor in format one-hot 
eticheteAntrenare = to_categorical(eticheteAntrenare, num_classes=5)
eticheteValidare = to_categorical(eticheteValidare, num_classes=5)

# augmentare
def augumentare(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.resize_with_crop_or_pad(img, 110, 110)
    img = tf.image.random_crop(img, size=[100, 100, 3])
    return img, label

# dataset-uri
dateAntrenare = tf.data.Dataset.from_tensor_slices((imaginiAntrenare, eticheteAntrenare))
dateAntrenare = dateAntrenare.map(augumentare).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
dateValidare = tf.data.Dataset.from_tensor_slices((imaginiValidare, eticheteValidare)).batch(64).prefetch(tf.data.AUTOTUNE)
dateTest = tf.data.Dataset.from_tensor_slices(imaginiTest).batch(64).prefetch(tf.data.AUTOTUNE)

# early stopping
earlyStopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# scheduler
scheduler = CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=10 * 196,
    t_mul=2.0,
    m_mul=0.9,
    alpha=1e-4
)

# model CNN 
model = models.Sequential([
    layers.Input(shape=(100, 100, 3)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

# compilare si antrenare
model.compile(optimizer=Adam(learning_rate=scheduler), loss='categorical_crossentropy', metrics=['accuracy'])
istoric = model.fit(dateAntrenare, validation_data=dateValidare, epochs=60, callbacks=[earlyStopping], verbose=1)

# evaluare
pierdere, acuratete = model.evaluate(dateValidare, verbose=0)
print(f"Loss pe validare: {pierdere:.4f}, Acuratete: {acuratete:.4f}")

# matrice de confuzie
etichetePred = np.argmax(model.predict(dateValidare), axis=1)
eticheteCorect = np.argmax(eticheteValidare, axis=1)

matrice = confusion_matrix(eticheteCorect, etichetePred)
ConfusionMatrixDisplay(confusion_matrix=matrice).plot(cmap='Greens')
plt.title("Matricea de confuzie (validare)")
plt.show()

# plot evolutie acuratete
plt.plot(istoric.history['accuracy'], label='Acuratete antrenare')
plt.plot(istoric.history['val_accuracy'], label='Acuratete validare')
plt.xlabel('Epoci')
plt.ylabel('Acuratete')
plt.legend()
plt.title("Evolutie acuratete")
plt.grid(True)
plt.show()
