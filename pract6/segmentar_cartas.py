# segmentar_cartas.py
#
# Programa pasa realizar operaciones de umbralización global con imágenes de niveles de gris y extracción de las cartas
# de la imagen
#
# Autor: David Bernad López   Fecha: diciembre 2025
#
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os
import random as rng
from clases_cartas import Card, Motif

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_labels= 'Lables image'
window_roi = 'Image_ROI'

cv2.namedWindow(window_original,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_labels,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_roi,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

low_H = 155
VISUALIZAR = False
MIN_AREA = 500
MAX_AREA = 40000

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

def segmentar_objetos_carta(Carta):

    carta_gris = Carta.grayImage
    carta_color = Carta.colorImage
    carta_color_id = Carta.cardId

   # Hacemos threshold y connectedComponents sobre la imagen roi
    fret, thresh2 = cv2.threshold(carta_gris,low_H,255,cv2.THRESH_BINARY_INV)
    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(thresh2, 4, cv2.CV_32S)

    # Bucle para cada objeto 'i'
    for i in range(1, totalLabels):
        # Área del objeeto
        area = values[i, cv2.CC_STAT_AREA]
    
        if (area > MIN_AREA and area < MAX_AREA):  # Filtro de tamaño de los motivos
            componentMask = (label_ids == i).astype("uint8") * 255
            
            # Encontramos un motivo si supera el area minima
            motif = Motif()
            imot = 0

            motif.motifId = imot
            motif.area = area
            motif.centroid = centroid[i]
            # A completar: Contornos del objeto ‘i’ con área mayor que el mínimo indicado
            contours_roi, hi = cv2.findContours(componentMask, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
            cnt = max(contours_roi, key=lambda k: len(k)) # contorno de mayor longitud
            motif.contour = cnt
            
            motif.perimeter = cv2.arcLength(motif.contour, True)

            color_medio = cv2.mean(carta_color, mask=componentMask)
            
            # moments = centro de gravedad y la inercia de la mancha de tinta
            motif.moments = cv2.moments(motif.contour)

            # Hu moments (mejor función para reconocer formas) -> Lo pasamos a lista unidmensional
            # Devuelve 7 numeros invariantes a traslación, escala y rotación
            motif.huMoments = cv2.HuMoments(motif.moments).flatten()
            # Lo hacemos logaritmico para que no sean tan pequeños (los hu moments)
            hu_log = -1 * np.sign(motif.huMoments) * np.log10(np.abs(motif.huMoments) + 1e-10)

            # A. Calcular el círculo mínimo que encierra el contorno
            (x, y), radius = cv2.minEnclosingCircle(motif.contour)
            
            # B. Guardar datos en el objeto (tal cual, floats)
            motif.circleCenter = (x, y)
            motif.circleRadious = radius  # Ojo con el nombre en tu clase (Radious vs Radius)
            
            # C. Dibujar el círculo (necesita Enteros)
            center_int = (int(x), int(y))
            radius_int = int(radius)

            # --- CARACTERÍSITCAS A PONER EN EL VECTOR DE FEATURES DEL MOTIVO ---
            # Aspect Ratio (Relación Ancho/Alto) -> INVARIANTE A ESCALA
            # Distingue cosas alargadas (como un 1) de cosas cuadradas (como un 8)
            x, y, w, h = cv2.boundingRect(motif.contour)
            aspect_ratio = float(w) / h

            # Rectangularidad (Extent) -> INVARIANTE A ESCALA
            # Área del objeto dividida por el área del rectángulo que lo encierra.
            rect_area = w * h
            extent = float(motif.area) / rect_area

            # Solidez (Solidity) -> INVARIANTE A ESCALA
            # Ayuda a distinguir formas con huecos o formas cóncavas (como Picas) de formas convexas (Rombos).
            hull = cv2.convexHull(motif.contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: 
                solidity = 0
            else:
                solidity = float(motif.area) / hull_area

            # --- VECTOR FINAL ---
            # Solo metemos datos que describen "CÓMO ES" la forma, no "CUÁNTO OCUPA".
            # Vector de características (features) del motivo
            motif.features = np.hstack([
                hu_log, 
                aspect_ratio, 
                extent, 
                solidity, 
            ])

            Carta.motifs.append(motif)
            imot += 1

            if VISUALIZAR:
                img = carta_color.copy()

                x0 = values[i, cv2.CC_STAT_LEFT]
                y0 = values[i, cv2.CC_STAT_TOP]
                w0 = values[i, cv2.CC_STAT_WIDTH]
                h0 = values[i, cv2.CC_STAT_HEIGHT]

                cv2.rectangle(img, (x0, y0), (x0+w0,y0+h0), (0, 255, 0), 2)
                cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
                cv2.circle(img, center_int, radius_int, (255, 0, 0), 2)

                print(f' * Índice {carta_color_id} - Nº puntos {len(contours)} , - Área motivo: {motif.area} - Perímetro:{motif.perimeter} \n')
                cv2.imshow(window_roi, img)
                cv2.waitKey()
              
    return Carta


# Selección de una carpeta mediante un diálogo de la biblioteca 'tkinter'
folders = '../Training'     # Poner la ruta de la carpeta de cartas de póker
path = filedialog.askdirectory(initialdir=folders, title="Seleccione una carpeta")

# Hacemos una lista vacía de cartas 'Cards' para ir añadiendo items mediante Cards.append(Card)
Cards = []
icard = 0   # Manejar el número de cartas que vamos a ir introduciendo

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
                contours, jerarquia = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

              # A completar: extraer el boundig box de la variable values
                x1 = values[i, cv2.CC_STAT_LEFT]
                y1 = values[i, cv2.CC_STAT_TOP]
                w = values[i, cv2.CC_STAT_WIDTH]
                h = values[i, cv2.CC_STAT_HEIGHT]
              
               # Recortar la imagen de gris por el boundign box
                roi = img_gray[int(y1):int(y1+h), int(x1):int(x1+w)].copy() 
                rows_gray, cols_gray = roi.shape[:2]
               
               # Recortar la imagen original de color por el boundign box
                roi_color = img[int(y1):int(y1+h), int(x1):int(x1+w)].copy()                
                rows_color, cols_color = roi_color.shape[:2]
    
                # Crear y apuntar los datos de la carta
                card = Card()
                card.cardId = icard
                
                card.grayImage = roi
                card.colorImage = roi_color

                card.boundingBox[0].x = x1
                card.boundingBox[0].y = y1
                card.boundingBox[0].width = w
                card.boundingBox[0].height = h

                Cards.append(card)
                icard += 1  

                minRect = cv2.minAreaRect(contours[0])  
                angulo_giro_rect_min = minRect[2]
                # // = división entera
                M = cv2.getRotationMatrix2D((cols_gray // 2, rows_gray // 2), angulo_giro_rect_min, 1.0)
                dst = cv2.warpAffine(roi, M, (cols_gray, rows_gray))

                card.angle = angulo_giro_rect_min

                segmentacion_carta = segmentar_objetos_carta(card)
                
                # Convertimos el mapa de IDs (que es casi negro) a colores vivos
                imagen_coloreada_etiquetas = label2rgb(label_ids)


        print('\n')
        key = -1
        while (key == -1):
             key=cv2.pollKey()
             # Aquí va la función  cv2.inRange(....)
             cv2.imshow(window_original, img) #, cmap='gray')
             #cv2.imshow(window_roi, dst)
             cv2.imshow(window_threshold, segmentacion_carta.colorImage) #, cmap='gray')
             cv2.imshow(window_labels, imagen_coloreada_etiquetas)
        if key == ord('q') or key == 27:    # 'q' o ESC para acabar
             break

    # Guardar las cartas en una archivo 'cartas.npz'
    np.savez('trainCards.npz', Cartas=Cards)  

cv2.destroyAllWindows()       

 