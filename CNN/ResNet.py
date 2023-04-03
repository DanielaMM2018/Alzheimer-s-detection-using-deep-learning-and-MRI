import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, cohen_kappa_score
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Dense, Flatten, AveragePooling2D, Add, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.initializers import glorot_uniform
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

hora1 = datetime.now()

# Función para cargar los datos
def pickload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Carga los datos
X = np.array(pickload("E:/Datas_8/Pickles/data_peq_igual.plk"))
y = pickload("E:/Datas_8/Pickles/diag_peq_igual.plk")

print("ya")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

#Transformar las formas de las entradas
X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)

# Normalización de datos
train_mean = np.mean(X_train)
train_std = np.std(X_train)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

#Data preprocessing
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

val_datagen = ImageDataGenerator()

# Aumento de datos
train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=256,
    shuffle=True,
)

test_generator = val_datagen.flow(
    X_test,
    y_test,
    batch_size=256,
    shuffle=False,
)

#Definición del modelo
def res_block(X, filters, stage):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='res_' + stage + 'conv_a',
    kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='res' + stage + 'bn_a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1), padding='same', name='res' + stage + 'conv_b',
    kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='res' + stage + 'bn_b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='res' + stage + 'conv_c',
    kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='res' + stage + '_bn_c')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X



def ResNet(input_shape, classes):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = res_block(X, filters=[64, 64, 256], stage='2a')
    X = res_block(X, filters=[64, 64, 256], stage='2b')
    X = res_block(X, filters=[64, 64, 256], stage='2c')
    
    X = res_block(X, filters=[128, 128, 512], stage='3a', stride=(2, 2))
    X = res_block(X, filters=[128, 128, 512], stage='3b')
    X = res_block(X, filters=[128, 128, 512], stage='3c')
    X = res_block(X, filters=[128, 128, 512], stage='3d')
    
    X = res_block(X, filters=[256, 256, 1024], stage='4a', stride=(2, 2))
    X = res_block(X, filters=[256, 256, 1024], stage='4b')
    X = res_block(X, filters=[256, 256, 1024], stage='4c')
    X = res_block(X, filters=[256, 256, 1024], stage='4d')
    X = res_block(X, filters=[256, 256, 1024], stage='4e')
    X = res_block(X, filters=[256, 256, 1024], stage='4f')
    

    
    X = res_block(X, filters=[512, 512, 2048], stage='5a', stride=(2, 2))
    X = res_block(X, filters=[512, 512, 2048], stage='5b')
    X = res_block(X, filters=[512, 512, 2048], stage='5c')
    
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)
    X = Flatten()(X)
    X = Dropout(0.3)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    
    return model


# definición del modelo
def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(200, 200, 1)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# compilación del modelo
model = create_model()
# 
#Compilación del modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5)])

#Definición de callbacks
filepath = "C:/Users/CAP-01/Desktop/Computador_2/Checkpoint/model_{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

callbacks = [early_stopping, checkpoint]

history = model.fit(train_generator,
                    epochs=3000,
                    validation_data=test_generator,
                    callbacks=callbacks)


# callbacks_list = [checkpoint]

# #Entrenamiento del modelo
# history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks_list)


#Evaluar el modelo
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("F1 score: {:.2f}".format(f1_score(y_test, y_pred)))
print("Cohen kappa score: {:.2f}".format(cohen_kappa_score(y_test, y_pred)))
print("ROC AUC score: {:.2f}".format(roc_auc_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#Gráficos de la función de pérdida y la precisión
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




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
auc = roc_auc_score(y_true, y_pred_prob)
# loss, accuracy = model.evaluate(test_generator, verbose=0)
evaluation = model.evaluate(test_generator, verbose=0)
loss = evaluation[0]
accuracy = evaluation[1]
kappa = cohen_kappa_score(y_true, y_pred)
print("Métricas adicionales:")
print(f"   Pérdida logarítmica: {loss:.4f}")
print(f"   Precisión: {precision:.4f}")
print(f"   Sensibilidad: {sensibilidad:.4f}")
print(f"   Especificidad: {especificidad:.4f}")
print(f"   F1-Score: {f1_score:.4f}")
print(f"   Área bajo la curva (AUC): {auc:.4f}")
print(f"   Cohen's Kappa: {kappa:.4f}")


# almacenar las métricas
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

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



hora2 = datetime.now()
hora = hora2 - hora1
print("Tiempo del codigo: ", hora)
