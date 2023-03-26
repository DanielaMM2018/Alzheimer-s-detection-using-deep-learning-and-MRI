import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skTrans
from pathlib import Path
import csv
import nibabel as nib
import numpy as np
import _pickle as pickle
def pickload(ruta):
    with open (ruta,"rb") as file:
        return pickle.load(file)


#     #importar los datos

array1 = pickload("E:/1Yr_ruta5/ruta5_AD.plk")
def ls(ruta = Path.cwd()):
    return [arch.name for arch in Path(ruta).iterdir() if arch.is_file()]


ruta = "E:/1Yr_ruta5/"
rutass=ls(ruta)
rutas = rutass[:-3] #se le resta por si tiene codigos o cosas que no son de archivos extra
List = []
R = []
L1 = len(array1)-1
L2 = len(rutas)-1

for i in range(L2):
    for j in range(L1):
        if array1[j] == rutas[i]:
            List.append(array1[j])
            R.append(rutas[i])
            
            
List2 = []
rut = []
elemtos = []
for j in R:
    elemtos.append(j)   #para verificar por donde paso
    im = nib.load(ruta + j)  #con esto recorre la carpeta y carga cada imagen 
    img = im.get_fdata()
    data = skTrans.resize(img, (200,200,160), order=1, preserve_range=True) #todas las imagenes quedan de 200x200 y serian 160
    for i in range(40,120):
        z_slice = data[:,:,i]  #con esto recorre las imagenes que selecciones por mayor informacion 
        dataReshape = np.reshape(z_slice,(1,40000)) #convierto la matriz de la imganes en un vector para que esta informacion en un futuro se maneje con mayor facilidad
        rut.append(R[i])
        List2.append(dataReshape) #las guardo con los arreglos anteriores
pickle.dump(List2, open('./data_una.plk', 'wb')) #guardo las imagenes adecuadas y seleccionadas
pickle.dump(rut, open('./rutas_una.plk', 'wb')) #gurardo la ruta para en un futuro juntarla en un excel con cada fila de imagen y mantener un orden


        