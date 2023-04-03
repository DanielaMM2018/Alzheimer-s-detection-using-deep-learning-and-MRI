import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from datetime import datetime

hora1 = datetime.now()

# Función para cargar los datos
def pickload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Carga los datos
X = np.array(pickload("D:/Datas_8/Pickles/data_peq_igual.plk"))
y = pickload("D:/Datas_8/Pickles/diag_peq_igual.plk")

print("ya")

# División de los datos de prueba en validación y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Transformar las formas de las entradas
X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=128
)

test_generator = validation_datagen.flow(
    X_test, y_test, batch_size=128
)


# Modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(200, 200, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2, activation="softmax"))


#Compilación del modelo
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-4),
    loss="categorical_crossentropy",
    metrics=["acc"],
)


checkpoint = ModelCheckpoint("alzheimer_detection.h5", monitor="val_loss", save_best_only=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 128,
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(X_test) // 128,
    verbose=1,
    callbacks=[checkpoint]
)


val_steps = len(X_test) // 128 + 1
y_true = []
y_pred_prob = []
for i in range(val_steps):
    batch_X, batch_y = next(test_generator)
    y_true.extend(batch_y)
    y_pred_prob.extend(model.predict(batch_X))

# Convertir las predicciones en etiquetas
y_pred = np.round(y_pred_prob)

# Nuevo calculo
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred_prob, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:\n", cm)

# Reporte de clasificación
cr = classification_report(y_true, y_pred)
print("Reporte de clasificación:\n", cr)

# Métricas adicionales
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)
f1_score = 2 * (precision * sensibilidad) / (precision + sensibilidad)
kappa = cohen_kappa_score(y_true, y_pred)


# Obtener las etiquetas verdaderas y las predicciones
val_steps = len(X_test) // 128 + 1
y_true = []
y_pred_prob = []
for i in range(val_steps):
    batch_X, batch_y = next(test_generator)
    y_true.extend(batch_y)
    y_pred_prob.extend(model.predict(batch_X))

# Convertir las predicciones en etiquetas
y_pred = np.round(y_pred_prob)

auc = roc_auc_score(y_true, y_pred_prob)
loss, accuracy = model.evaluate(test_generator, verbose=0)


print("Matriz de confusión:\n", cm)
print("Reporte de clasificación:\n", cr)


print("Métricas adicionales:")
print(f"   Pérdida logarítmica: {loss:.4f}")
print(f"   Precisión: {precision:.4f}")
print(f"   Sensibilidad: {sensibilidad:.4f}")
print(f"   Especificidad: {especificidad:.4f}")
print(f"   F1-Score: {f1_score:.4f}")
print(f"   Área bajo la curva (AUC): {auc:.4f}")
print(f"   Cohen's Kappa: {kappa:.4f}")


# Precisión en conjunto de entrenamiento y validación
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='lower right')
plt.show()

# Pérdida en conjunto de entrenamiento y validación
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss del modelo')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
plt.show()


# Definir las etiquetas y los valores de las métricas
labels = ['Loss', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC', 'Cohen\'s Kappa']
values = [loss, precision, sensibilidad, especificidad, f1_score, auc, kappa]

# Crear el gráfico de barras
plt.figure(figsize=(8,6))
plt.bar(labels, values, color=['gray', 'blue', 'green', 'red', 'purple', 'orange', 'brown'])
plt.ylim([0,1]) # Establecer los límites del eje y
plt.title('Métricas adicionales')
plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.show()



from sklearn.metrics import classification_report
import json

# Suponiendo que tienes las etiquetas verdaderas en una lista llamada "y_true" 
# y las predicciones del modelo en una lista llamada "y_pred"
cr = classification_report(y_true, y_pred, output_dict=True)

# Guardar el reporte de clasificación en un archivo JSON
with open('classification_report.json', 'w') as f:
    json.dump(cr, f)


import xlsxwriter

# Crear un objeto de libro de trabajo de Excel
workbook = xlsxwriter.Workbook('resultados.xlsx')

# Agregar una hoja de cálculo
worksheet = workbook.add_worksheet()

# Escribir encabezados en la hoja de cálculo
worksheet.write('A1', 'Matriz de confusión')
worksheet.write('A5', 'Reporte de clasificación')
worksheet.write('A6', 'precision')
worksheet.write('A7', 'recall')
worksheet.write('A8', 'f1-score')

# Escribir encabezados de métricas en la hoja de cálculo
worksheet.write('D1', 'Pérdida logarítmica')
worksheet.write('E1', 'Precisión')
worksheet.write('F1', 'Sensibilidad')
worksheet.write('G1', 'Especificidad')
worksheet.write('H1', 'F1-Score')
worksheet.write('I1', 'Área bajo la curva (AUC)')
worksheet.write('J1', "Cohen's Kappa")

# Guardar las métricas y los resultados en un archivo CSV
metrics = {
    'Pérdida logarítmica': loss,
    'Precisión': precision,
    'Sensibilidad': sensibilidad,
    'Especificidad':  especificidad,
    'F1-Score': f1_score,
    'Área bajo la curva (AUC)': auc,
    "Cohen's Kappa":  kappa,
}

# Escribir valores de métricas en la hoja de cálculo
worksheet.write('D2', loss)
worksheet.write('E2', precision)
worksheet.write('F2', sensibilidad)
worksheet.write('G2', especificidad)
worksheet.write('H2', f1_score)
worksheet.write('I2', auc)
worksheet.write('J2', kappa)

try:
    # Abrir archivo JSON con las métricas
    with open('classification_report.json', 'r') as f:
        cr = f.read()

    # Convertir el archivo JSON en un diccionario
    cr_dict = json.loads(cr)

    # Escribir métricas y resultados en la hoja de cálculo
    worksheet.write('B6', cr_dict['weighted avg']['precision'])
    worksheet.write('B7', cr_dict['weighted avg']['recall'])
    worksheet.write('B8', cr_dict['weighted avg']['f1-score'])

except FileNotFoundError:
    print('No se encontró el archivo classification_report.json')

# Escribir matriz de confusión en la hoja de cálculo
for i in range(cm.shape[0]):
    for j in range(cm.shape[0]):
        worksheet.write(1+i, 0+j, cm[i][j])

# Cerrar el libro de trabajo
workbook.close()

# almacenar las métricas
train_loss = history.history['loss']
train_acc = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

# graficar la precisión
plt.plot(train_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
plt.savefig('Accuracy.png')
plt.show()

# graficar la pérdida
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.savefig('Loss.png')
plt.show()


# Definir las etiquetas y los valores de las métricas
labels = ['Loss', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC', 'Cohen\'s Kappa']
values = [loss, precision, sensibilidad, especificidad, f1_score, auc, kappa]

# Crear el gráfico de barras
plt.figure(figsize=(8,6))
plt.bar(labels, values, color=['gray', 'blue', 'green', 'red', 'purple', 'orange', 'brown'])
plt.ylim([0,1]) # Establecer los límites del eje y
plt.title('Métricas adicionales')
plt.xlabel('Métricas')
plt.ylabel('Valores')
plt.savefig('metricas.png')
plt.show()

hora2 = datetime.now()
hora = hora2 - hora1
print("Tiempo del codigo: ", hora)
