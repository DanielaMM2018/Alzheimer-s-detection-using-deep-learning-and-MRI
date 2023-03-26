#librerias de siempre
import numpy as np
import _pickle as pickle
import pandas as pd
import csv


Excel3 = pd.read_csv("ruta3.csv")
Excel5 = pd.read_csv("ruta5.csv")    
        

c3 = 0
c33 = 0
c333 = 0
c_3 = []

c5 = 0
c55 = 0
c555 = 0
c_5 = []

array = []
array2 = []
array3 = []

w = 80
l = Excel3["Grupo"]
l = list(l)
for i in range(len(l)):
  c333 += 1
  if Excel3['Grupo'][i] == 'CN':
      c3 += 1
  else:
    c_3.append(c333)
    array2.append(Excel3['ruta'][i])
    for j in range(w):
        array.append(Excel3['ruta'][i]) 
        c33 += 1
        
    
l = Excel5["Grupo"]
l = list(l)
for i in range(len(l)):
  c555 += 1
  if Excel5['Grupo'][i] == 'CN':
      c5 += 1
  else:
    c_5.append(c555)
    array3.append(Excel5['ruta'][i])
    for j in range(w):
        array.append(Excel5['ruta'][i])
        c55 += 1
        
        
pickle.dump(array, open('E:/rutas_AD.plk', 'wb'))
pickle.dump(array2, open('E:/1Yr_ruta3/ruta3_AD.plk', 'wb'))
pickle.dump(array3, open('E:/1Yr_ruta5/ruta5_AD.plk', 'wb'))
