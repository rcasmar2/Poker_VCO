# segmentat_imagen.py
#
# Programa pasa realizar operaciones de umbrallización global con imágenes de niveles de gris y extracción de las cartas
# de la imagen
#
# Autor: José M Valiente    Fecha: marzo 2023
#
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os
import random as rng

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_labels= 'Lables image'
cv2.namedWindow(window_original,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_labels,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

low_H = 155

def label2rgb(label_img):
# Función para conversión de etiquetas a colores    
    label_hue = np.uint8(179*(label_img)/np.max(label_img))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_ids==0] = 0
    return labeled_img

# Selección de una carpeta mediante un diálogo de la biblioteca 'tkinter'
folders = '../VxC FOTOS'     # Poner la ruta de la carpeta de cartas de póker
path = filedialog.askdirectory(initialdir=folders, title="Seleccione una carpeta")

# Hacemos una lista vacía de cartas 'Cards' para ir añadiendo items mediante Cards.append(Card)
Cards = []

for root,  dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not(name.endswith('.jpg')):
            continue
        filename = os.path.join(root, name)
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow(window_original,img)
        fret,thresh1 = cv2.threshold(img_gray,low_H,255,cv2.THRESH_BINARY_INV)
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresh1, 4, cv2.CV_32S)
        
        output = np.zeros(img_gray.shape, dtype="uint8")
          # Bucle para cada objeto 'i'
        for i in range(1, totalLabels):
                # Área del objeeto
            area = values[i, cv2.CC_STAT_AREA]
   
            if (area > 300000):  # Filtro de tamaño   NUEVA CARTA
                componentMask = (label_ids == i).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)
                print(area)
             # A completar: Contornos del objeto ‘i’ con área mayor que el mínimo indicado
             
             
             
              # A completar: extraer el boundig box de la variable values
              
              
              
               # Recortar la imagen de gris por el boundign box
               
               
               
               # Recortar la imagen original de color por el boundign box
               
               
               
                # Crear y apuntar los datos de la carta
                   
                # Añadir la carta a la lista de cartas
                #    Cards.append(c)
                #    icard+=1      
               
        print('\n')
        key = -1
        while (key == -1):
             key=cv2.pollKey()
             # Aquí va la función  cv2.inRange(....)
             cv2.imshow(window_original, img) #, cmap='gray')
             cv2.imshow(window_labels,label2rgb(label_ids))
             cv2.imshow(window_threshold, output) #, cmap='gray')
        if key == ord('q') or key == 27:    # 'q' o ESC para acabar
             break

 # Guardar las cartas en una archivo 'cartas.npz'
 
 
 
cv2.destroyAllWindows()       

 