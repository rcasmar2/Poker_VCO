# modelor_kNN.py
#
# Programa pasa realizar el entrenamiento de un clasificador k-Nearest Neighbour (kNN) con los motivos
# de las cartas de poker, obtenidos desde un archivo de tipo *.npz
# Después se realiza la validación del modelo kNN con los motivos de la cartas de test.
# 
#
# Autor:  José M Valiente + alumno .............    Fecha: mayo 2023
#

import cv2
import os
import numpy as np
from clases_cartas import Card, Motif
import warnings


FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K') # Se accede mediantge Carta.FIGURES[i]
SUITS = ('Rombos','Picas','Corazones','Treboles')
MOTIF_LABELS = ('Rombos','Picas','Corazones','Treboles','0','2','3','4','5','6','7','8','9','A','J','Q','K','Others')   

filecard = 'trainCards.npz'

npzfile = np.load(filecard, allow_pickle=True) 
cards = npzfile['Cartas']
llen = cards.size

# Listas vacias
samples = []     # Lista de características de cada muestra
responses = []   # Lista de etiqueta real de cada muestra


############## TRAINING ############################
j=0
for i in range(0,llen):   # para todas las cartas
    motifs = cards[i].motifs
    for mot in motifs:
        lbl = mot.motifLabel
        if lbl == 'i':
            continue       # si el motivo no está etiquetado se descarta
        idx = MOTIF_LABELS.index(lbl)    # etiqueta real del motivo
        responses.append(idx)
        j +=1
        print(j, idx, lbl)       
        # Añadir a samples todas las características del motivo que consideremos oportunas.
        # Ojo que debe ser una fila o vector de números reales
        ...............
        samples.append(.............)


# Convertir las listas en arrays
sampl = np.asarray(samples_vect).astype(np.float32)
resp = np.asarray(responses)

# Creación del modelo kNN
knn = .......................       # A comletar

# Entrenar el modelo kNN


knn.train(....................... )      # A comletar


######## TEST ##########

filecardTest = 'testCards.npz'
npzfileT = np.load(filecardTest, allow_pickle=True) 
cardsTest = npzfileT['Cartas']


samplesTest = []
responsesTest = []

j=0
for i in range(0,le):   # para todas las cartas
    motifs = cardsTest[i].motifs
    for mot in motifs:
       
        lbl = mot.motifLabel
        if lbl == 'i':    # si el motivo no está etiquetado se le pone 'Others'
            lbl = 'Others'
        idx = MOTIF_LABELS.index(lbl)
        responsesTest.append(idx)
        j +=1
        print(j, idx, lbl)
        # Añadir a samplesTest todas las características del motivo que consideremos oportunas.
        # Ojo que debe ser una fila o vector de números reales
        ...............
        samplesTest.append(.............)
     
        
        
        # Convertir las listas en arrays
samplTest = np.asarray(samplesTest).astype(np.float32)
respTest = np.asarray(responsesTest)

# Predicción con k=3

ret, results, neighbours ,dist = knn.................................  # A completar por el alumno

# Visualización de resultados

le = len(results)
j=0
pred = np.zeros(le)
real = np.zeros(le)

for i in range(0,le):
    pred[i] = int(results[i][0])
    real[i] = respTest[i]
    pred_str = MOTIF_LABELS[int(results[i][0])]
    real_str = MOTIF_LABELS[respTest[i]]
    print(f"result: {pred_str}  real:  {real_str}" )
    if pred[i]==real[i]:
        j+=1

print(f'Tasa aciertos:  {j/le}')


# VISUALIZACIÓN

from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

# Obtenemos el report
CLS_REP=classification_report(real, pred, target_names=MOTIF_LABELS)
print('Classification report ', CLS_REP) 
CONF_MAT = confusion_matrix(real,pred)
print('Confusion Matrix', CONF_MAT)
MCC = matthews_corrcoef(real, pred)
print('MCC: ', MCC)

import matplotlib.pyplot as plt
cm_display = metrics.ConfusionMatrixDisplay(...................)  # A completar
cm_display.plot(xticks_orientation='vertical')
plt.show()

        
        