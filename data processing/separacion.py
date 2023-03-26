import numpy as np
import _pickle as pickle
import pandas as pd
import csv
def pickload(ruta):
    with open (ruta,"rb") as file:
        return pickle.load(file)

#     #importa las imagenes
array0 = pickload("E:/Datas_8/data_3_AD.plk")
array1 = pickload("E:/Datas_8/data_5_AD.plk")
array2 = pickload("E:/Datas_8/data1.plk")
array3 = pickload("E:/Datas_8/data2.plk")

    #importar las direcciones para la futura distribuccion

array8 = pickload("E:/Datas_8/rutas_3_AD.plk")
array81 = pickload("E:/Datas_8/rutas_5_AD.plk")
array7 = pickload("E:/Datas_8/rutas1.plk")
array71 = pickload("E:/Datas_8/rutas2.plk")



# #     #crear los arrays de los diagnoticos para la distribuccion a futuro

array = []
array3 = []

#AD

for i in range(len(array0)):
    array3.append(array0[i])
    
for i in range(len(array1)):
    array3.append(array1[i])
    
#CN
    

for i in range(len(array2)):
    array.append(array2[i])

for i in range(len(array3)):
    array.append(array3[i])
    
    
#CN
  
array0 = []
for i in range(59760):
    array0.append(0)
    

#AD
    
array1 = []
for i in range(53440):
    array1.append(1)
    

#Direcciones

#CN

for i in range(len(array71)):
    array7.append(array71[i])
    
#AD

for i in range(len(array81)):
    array8.append(array81[i])
    
    
    
    
#     # separar los datos de manera odenada para acceder a ellos  (hasta 56320 porque el AD tiene ese numero de datos)

array4 = []
array5 = []
array6 = []  

for i in range(53440):
    a = array0[i]    # los unos del diag de CN
    b = array1[i]   # los ceros del diag de AD
    c = array[i]   # los datos de imagenes del diag de CN
    d = array3[i]   # los datos de imagenes del diag de AD
    e = array7[i]   # Las direcciones de los diag de CN
    f = array8[i]   # Las direcciones de los diag de AD
    array4.append(f)   # AD
    array4.append(e)   # CN
    array5.append(d)   #imagenes AD
    array5.append(c)   #imagenes CN
    array6.append(b)   #diag AD
    array6.append(a)   #diag CN

#      # separar los datos restantes que faltaron de CN

for i in range(53440,59760):
    b = array0[i]   # los ceros del diag de CN
    d = array[i]   # los datos de imagenes del diag de CN
    f = array7[i]   # Las direcciones de los diag de CN
    array4.append(f)
    array5.append(d)
    array6.append(b)

#106680 COMBINADO
#113200 con CN más, -->6320 extra

# #     # separar los datos a evaluar (para poder importar los picos) IMAGENES

array9 = []
array10 = []
array11 = []
array12 = []



array13 = []
array14 = []
array15 = []



#iguales

for i in range(23232):
    array9.append(array5[i])

for i in range(23232,69696):
    array10.append(array5[i])
    
for i in range(60416,106880):  #-->9280 iguales que el otro
    array11.append(array5[i])

for i in range(23232,92928):  
    array12.append(array5[i])

#desigual


for i in range(89968,113200):
    array13.append(array5[i])

for i in range(66736,113200):
    array14.append(array5[i])


for i in range(43504,113200):  
    array15.append(array5[i])
    
    

    
#     # separar los datos a evaluar (para poder importar los picos) DIAG


array16 = []
array17 = []
array18 = []
array19 = []



array20 = []
array21 = []
array22 = []


#iguales

for i in range(23232):
    array16.append(array6[i])

for i in range(23232,69696):
    array17.append(array6[i])
    
for i in range(60416,106880):  #-->9280 iguales que el otro
    array18.append(array6[i])

for i in range(23232,92928):  
    array19.append(array6[i])

#desigual


for i in range(89968,113200):
    array20.append(array6[i])
    
for i in range(66736,113200):
    array21.append(array6[i])

for i in range(43504,113200):  
    array22.append(array6[i])
    
    
#     # separar los datos a evaluar (para poder importar los picos) DIR

array23 = []
array24 = []
array25 = []
array26 = []



array27 = []
array28 = []
array29 = []


#iguales

for i in range(23232):
    array23.append(array4[i])

for i in range(23232,69696):
    array24.append(array4[i])
    
for i in range(60416,106880):  #-->9280 iguales que el otro
    array25.append(array4[i])

for i in range(23232,92928):  
    array26.append(array4[i])


#desigual

for i in range(89968,113200):
    array27.append(array4[i])

for i in range(66736,113200):
    array28.append(array4[i])

for i in range(43504,113200):  
    array29.append(array4[i])
    
pickle.dump(array9, open('E:/Datas_8/Pickles/data_peq_igual.plk', 'wb'))
pickle.dump(array10, open('E:/Datas_8/Pickles/data_igual1.plk', 'wb'))
pickle.dump(array11, open('E:/Datas_8/Pickles/data_igual2.plk', 'wb'))
pickle.dump(array12, open('E:/Datas_8/Pickles/data_grande_igual.plk', 'wb'))

pickle.dump(array13, open('E:/Datas_8/Pickles/data_peq_desigual.plk', 'wb'))
pickle.dump(array14, open('E:/Datas_8/Pickles/data_desigual.plk', 'wb'))
pickle.dump(array15, open('E:/Datas_8/Pickles/data_grande_desigual.plk', 'wb'))






pickle.dump(array16, open('E:/Datas_8/Pickles/diag_peq_igual.plk', 'wb'))
pickle.dump(array17, open('E:/Datas_8/Pickles/diag_igual1.plk', 'wb'))
pickle.dump(array18, open('E:/Datas_8/Pickles/diag_igual2.plk', 'wb'))
pickle.dump(array19, open('E:/Datas_8/Pickles/diag_grande_igual.plk', 'wb'))

pickle.dump(array20, open('E:/Datas_8/Pickles/diag_peq_desigual.plk', 'wb'))
pickle.dump(array21, open('E:/Datas_8/Pickles/diag_desigual.plk', 'wb'))
pickle.dump(array22, open('E:/Datas_8/Pickles/diag_grande_desigual.plk', 'wb'))






pickle.dump(array23, open('E:/Datas_8/Pickles/dir_peq_igual.plk', 'wb'))
pickle.dump(array24, open('E:/Datas_8/Pickles/dir_igual1.plk', 'wb'))
pickle.dump(array25, open('E:/Datas_8/Pickles/dir_igual2.plk', 'wb'))
pickle.dump(array26, open('E:/Datas_8/Pickles/dir_grande_igual.plk', 'wb'))


pickle.dump(array27, open('E:/Datas_8/Pickles/dir_peq_desigual.plk', 'wb'))
pickle.dump(array28, open('E:/Datas_8/Pickles/dir_desigual.plk', 'wb'))
pickle.dump(array29, open('E:/Datas_8/Pickles/dir_grande_desigual.plk', 'wb'))

