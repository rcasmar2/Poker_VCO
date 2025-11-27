# segmentat_cartas.py
#
# Programa pasa realizar operaciones de umbrallización global con imágenes de niveles de gris y extracción de las cartas
# de la imagen y obtención de las cartas.
#
# Autor: José M Valiente    Fecha: marzo 2023
#
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os
from clases_cartas import Card, Motif

    
def segmentar_objetos_carta(car):
        lowH = 185
        _ ,thresh_roi = cv2.threshold(car.grayImage,lowH,255,cv2.THRESH_BINARY_INV)
        (totalLabels_roi, lab_img, val, centr) = cv2.connectedComponentsWithStats(thresh_roi, 4, cv2.CV_32SC1)
       # contours_roi, hi = cv2.findContours(thresh_roi, cv2.RETR_LIST ,  cv2.CHAIN_APPROX_SIMPLE)
        if VISUALIZAR:
            cv2.imshow(window_threshold, thresh_roi)
        imot=0
        MIN_AREA = 1000
        MAX_AREA = 40000
        
        for k in range(1, totalLabels_roi):
            
            area = val[k,cv2.CC_STAT_AREA]
            if (area > MIN_AREA) and (area < MAX_AREA):
                componentMask2 = (lab_img == k).astype("uint8") * 255
                contours_roi, hi = cv2.findContours(componentMask2, cv2.RETR_LIST ,  cv2.CHAIN_APPROX_NONE)
                cnt = max(contours_roi, key=lambda k: len(k))   # contorno de mayor longitud
                # Datos del motivo  AÑADIR los datos que queramos
                x0,y0,w0,h0 =    	# Bounding box
                mu = 					# Momentos
                # Otros
                
                centroide = np.array([mu['m10'] / mu['m00'], mu['m01'] / mu['m00']], dtype = np.intc)
                center, radious = cv2.minEnclosingCircle(cnt)   # Círculo 
                center = np.array(list(center), dtype=np.intc)
                radious = int(radious)
                
                # Creación y relleno del motivo
                motif = Motif()
                motif.contour = cnt
                motif.motifId = imot
                motif.area = .............
                motif.perimeter = ................. 
                motif.centroid = centroide
                motif.moments = mu
                motif.huMoments = .................
                motif.circleCenter = center
                motif.circleRadious = radious
                car.motifs.append(motif)     # Se añade a la lista de motivos
                imot+=1
                
               # Visualización de resultados
                if VISUALIZAR:
                    img = roi_color.copy()
                    cv2.rectangle(img,(x0,y0),(x0+w0,y0+h0),(0,255,0),2)
                    cv2.drawContours(img, [cnt], -1, (255,0,0), 5) 
                    cv2.circle(img,(centroide[0],centroide[1]),4, (255,0,0),2)
                    cv2.circle(img,center,radious, (255,0,0),2)
                    print(' * Ínidice %d, Area global = %.2f - Nº puntos %d- Area motif: %.2f - Perimeter: %.2f' %
                          (k,val[k,cv2.CC_STAT_AREA],len(cnt),motif.area, motif.perimeter))
                    cv2.imshow(window_roi, img)
                    cv2.waitKey()
        print(f'Nº de motivos: {imot} \n') 
                
                                
############### Programa principal  ###############################

VISUALIZAR = True

filecard = 'cartas.npz'   # 'trainCards.npz  testCards.npz

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_roi = 'ROI image'
cv2.namedWindow(window_original,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_roi,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

low_H = 155

 # Para ver 'Card' con una sintaxis como si fuera un Struct de C lo asinamos a un objeto de clase 'Card'  c = Card()
 # de esta form apodemos acceder a los elementos con c.realSuit = Card.DIAMONDS  o c.grayImage = roi ....
 
 # Hacemos una lista vacía de cartas 'Cards' para ir añadiendo items mediante Cards.append(Card)
Cards = []

# Selección de una carpeta mediante un diálogo de la biblioteca 'tkinter'
path = filedialog.askdirectory(initialdir="./../../VxC FOTOS/", title="Seleccione una carpeta")
icard = 0   # Numeración de las cartas

for root,  dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not(name.endswith('.jpg')):
            continue
        filename = os.path.join(root, name)
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if VISUALIZAR:
            cv2.imshow(window_original,img)
        fret,thresh1 = cv2.threshold(img_gray,low_H,255,cv2.THRESH_BINARY_INV)
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(255-thresh1, 4, cv2.CV_32S)
       
            # Bucle para cada objeto 'i'
        for i in range(1, totalLabels):
                # Área del objeeto
            area = values[i, cv2.CC_STAT_AREA]
          
            if (area > 300000):  # Filtro de tamaño   NUEVA CARTA
                componentMask = (label_ids == i).astype("uint8") * 255
                # Contornos del objeto ‘i’ con área mayor que el mínimo indicado
                contours, hi = cv2.findContours(componentMask, cv2.RETR_EXTERNAL ,  cv2.CHAIN_APPROX_SIMPLE)
                                
                # A completar por el alumno: extraer el boundig box de la variable values
                x1 = values[i, cv2.CC_STAT_LEFT]
                y1 = values[i, cv2.CC_STAT_TOP]
                w  = values[i, cv2.CC_STAT_WIDTH]
                h  = values[i, cv2.CC_STAT_HEIGHT]
                
                # Recortar la imagen de gris por el boundign box
                roi = img_gray[int(y1):int(y1+h), int(x1):int(x1+w)].copy()
                
                # Recortar la imagen original de color por el boundign box
                roi_color = img[int(y1):int(y1+h), int(x1):int(x1+w)].copy()
                rows, cols = roi.shape[:2]
                
                if len(contours)>=1:
                    #minRect = [None]*len(contours)
                    c = contours[0]
                    minRect = cv2.minAreaRect(c)
                    ww,hh = minRect[1]
                    if (ww < hh):
                        angle = minRect[2]
                    else:
                        angle = minRect[2] - 90
                    
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    roi_warped = cv2.warpAffine(roi,M, (cols ,rows))
                    
                    # apuntamos los datos de la carta
                    c = Card()
                    c.boundingBox.x = x1
                    c.boundingBox.y = y1
                    c.boundingBox.width = w
                    c.boundingBox.height = h
                    c.angle = angle
                    c.grayImage = roi
                    c.colorImage = roi_color
                    c.cardId = icard                    
                    
                    segmentar_objetos_carta(c)
                    Cards.append(c)
                    icard+=1 

        if VISUALIZAR:
            print('\n')
            key = -1
            while (key == -1):
                 key=cv2.pollKey()
                 
                 cv2.imshow(window_original, img)
                 cv2.imshow(window_threshold, roi) #, cmap='gray')
                 cv2.imshow(window_roi, roi_color) #, Ojo: la imagen en color está en formato BGR
            if key == ord('q') or key == 27:    # 'q' o ESC para acabar
                 break

# Guardar las cartas en una archivo filecard
np.savez(filecard, Cartas=Cards)
# npzfile = np.load('cartas.npz', allow_pickle=True)   # es un ndarray
# Cartas = npzfile['Cartas']                     # es una lista
# acceso mediante x = Cartas[idx].boundingBox.x
#                 bbox = Cartas[idx].boundingBoxx
#                 suit = Cartas[idx].realSuit
cv2.destroyAllWindows()       
