#librerias de siempre
import numpy as np
import _pickle as pickle
import pandas as pd
import csv


Excel1 = pd.read_csv("ruta1.csv")
Excel2 = pd.read_csv("ruta2.csv")
Excel3 = pd.read_csv("ruta3.csv")
Excel4 = pd.read_csv("ruta4.csv")
Excel5 = pd.read_csv("ruta5.csv")    
        
c1 = 0
c11 = 0
c111 = 0
c_1 = []

c2 = 0
c22 = 0
c222 = 0
c_2 = []

c3 = 0
c33 = 0
c333 = 0
c_3 = []

c4 = 0
c44 = 0
c444 = 0
c_4 = []

c5 = 0
c55 = 0
c555 = 0
c_5 = []

array = []
array2 = []
w = 80

l = Excel1["Grupo"]
l = list(l)
for i in range(len(l)):
  c111 += 1
  if Excel1['Grupo'][i] == 'CN':
      c_1.append(c111)
      array2.append(Excel1['ruta'][i]) 
      for j in range(w):
          array.append(Excel1['ruta'][i]) 
          c1 += 1
  else:
    c11 += 1


l = Excel2["Grupo"]
l = list(l)
for i in range(len(l)):
  c222 += 1
  if Excel2['Grupo'][i] == 'CN':
      c_2.append(c222)
      array2.append(Excel2['ruta'][i]) 
      for j in range(w):
          array.append(Excel2['ruta'][i]) 
          c2 += 1
  else:
    c22 += 1

l = Excel3["Grupo"]
l = list(l)
for i in range(len(l)):
  c333 += 1
  if Excel3['Grupo'][i] == 'CN':
      c_3.append(c333)
      array2.append(Excel3['ruta'][i]) 
      for j in range(w):
          array.append(Excel3['ruta'][i]) 
          c3 += 1
  else:
    c33 += 1
        
    
    
l = Excel4["Grupo"]
l = list(l)
for i in range(len(l)):
  c444 += 1
  if Excel4['Grupo'][i] == 'CN':
      c_4.append(c444)
      array2.append(Excel4['ruta'][i]) 
      for j in range(w):
          array.append(Excel4['ruta'][i]) 
          c4 += 1
  else:
    c44 += 1

l = Excel5["Grupo"]
l = list(l)
for i in range(len(l)):
  c555 += 1
  if Excel5['Grupo'][i] == 'CN':
      c_5.append(c555)
      array2.append(Excel5['ruta'][i]) 
      for j in range(w):
          array.append(Excel5['ruta'][i])
          c5 += 1
  else:
      c55 += 1
        
      
pickle.dump(array, open('E:/rutas_CN.plk', 'wb'))
pickle.dump(array2, open('E:/3Yr/ruta_CN.plk', 'wb'))
