# clases_cartas.py
#
# Definición de las clases 'Motif' y 'Card' para el trabajo con las cartas de póker.
#
# Se importa mediante  "from clases_cartas import Card, Motif"
#
# Autor: José M Valiente    Fecha: abril 2023
#
import numpy as np

###### CLASES #################

class Motif:
    
    MOTIF_LABELS = ('Diamonds','Spades','Hearts','Clubs','0','2','3','4','5','6','7','8','9','A','J','Q','K','Others')
    
    def __init__(self):  # Constructor
        self.motifId = 0
        self.motifLabel = 'i'
        self.motifPredictedLabel = 'iii'
        self.area = 0.0
        self.contour = []
        self.perimeter = 0.0
        self.features = []
        self.moments = []
        self.huMoments = []
        self.centroid = []
        self.circleCenter = []
        self.circleRadious = 0.0
        
    def __repr__(self):
        rep = f"Motif number: {str(self.motifId)} --  Motif Label:  {self.motifLabel} -- Predicted Motif Label: {self.motifPredictedLabel}"
        bb = f"Area: {str(self.area)}   Perimeter: {str(self.perimeter)}"
        ims = f"Contour: {self.contour}  Features:  {str(self.features)}"
        _new_line = "\n"
        return rep + _new_line + bb + _new_line + ims
       
class Card:
      
      # Suits.  Palos de las cartas de póker
      DIAMONDS = 'Diamonds'   # Rombos
      SPADES = 'Spades'       # Picas
      HEARDS = 'Hearts'       # Corazones
      CLUBS ='Clubs'          # Tréboles
      # Figuras y cifras de las cartas de póker
      SUITS = ('Rombos','Picas','Corazones','Treboles')
      FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K') # Se accede mediantge Carta.FIGURES[i]
      
      def __init__(self):  # Constructor
        self.cardId = 0
        self.realSuit = 'i'
        self.realFigure = 'iii'
        self.predictedSuit = 'o'
        self.predictedFigure = 'ooo'
        bboxType = [('x', np.intc),('y',np.intc),('width',np.intc),('height',np.intc)]
        self.boundingBox = np.zeros(1, dtype=bboxType).view(np.recarray)
        self.angle = 0.0
        self.grayImage = np.empty([0,0], dtype=np.uint8)
        self.colorImage = np.empty([0,0,0], dtype=np.uint8)
        self.motifs = []
        
      def __repr__(self):
        rep = f"Card number: {str(self.cardId)} --  Real Suit/Figure:  {self.realSuit} / {self.realFigure} -- Predicted Suit/Fifure: {self.predictedSuit} / {self.predictedFigure}"
        bb = f"Bounding Box: {str(self.boundingBox)}   Rect angle: {str(self.angle)}"
        ims = f"Gray image:  {str(self.grayImage.shape)}   Color image:  {str(self.colorImage.shape)}"
        new_line = "\n"
        return rep + new_line + bb + new_line + ims