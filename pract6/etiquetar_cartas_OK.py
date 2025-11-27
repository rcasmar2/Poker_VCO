# etiquetar_cartas.py
#
# Programa pasa realizar el etiquetado de las cartas y sus motivos, obtenidos desde un archivo de tipo *.npz
# 
#
# Autor: José M Valiente    Fecha: abril 2023
#

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
from clases_cartas import Card, Motif
import warnings

FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K') # Se accede mediantge Carta.FIGURES[i]
SUITS = ('Rombos','Picas','Corazones','Treboles')
MOTIF_LABELS = ('Rombos','Picas','Corazones','Treboles','0','2','3','4','5','6','7','8','9','A','J','Q','K','Others')   

filecard = 'cartas.npz'

class App(tk.Tk):
    #FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K') # Se accede mediantge Carta.FIGURES[i]
        
    def __init__(self):
        super().__init__()
        self.title('Widget Demo')
        self.resizable(True, True)
        
        self.cards = []
        self.card_idx = -1
        self.motifs = []
        self.motif_idx = -1
   
        self.create_widgets()
            # Crear la estructura de cartas
        npzfile = np.load(filecard, allow_pickle=True) 
        self.cards = npzfile['Cartas']
            # Averiguar cual es la primera carta sin etiquetar
        le = len(self.cards)
        for i in range(0,le):
            if self.cards[i].realSuit == 'i':
                    self.card_idx = i
                    break
        
        if (i == (le-1)):
            res = messagebox.askquestion('Atención', 'No quedan cartas sin etiquetar\n ¿Terminar la aplicación?')
            if res == 'yes':
                quit()
            else:
                self.show_card(self.cards[0])   # Carta inicial
        else:
            self.show_card(self.cards[i])   # Carta inicial
            print(f'{len(self.cards)} Cartas. Carta inicial {self.card_idx} ')    
              
    def get_photo(self, img):
        (h,w,_) = img.shape
        im = Image.fromarray(img)
        print(im.getbands())
        scale_div = 3
        if (w > 4000) or (h > 4000):
            scale_div = 6
        if (w < 200) or (h < 200):
            scale_div = 0.5
        width = int(w / scale_div)
        height = int(h / scale_div)
        im_resized = im.resize((width, height))
        photo = ImageTk.PhotoImage(im_resized)
        return photo        
    
    def show_image(self,  img):
        photo = self.get_photo(img)
        self.panel = ttk.Label(self, image = photo)
        self.panel.image = photo
        self.panel.grid(column=0, row=0, padx=10, pady=10,columnspan = 3)

    def show_card(self, card):
        # Actualiza los datos de la carta actual y la muestra
        self.panel.grid_forget()
        self.motif_idx = 0
        self.motifs = card.motifs
        self.img = card.colorImage
        if (len(self.motifs) > 0):
            self.show_motif(self.motifs[0])   # Motivo inicial
            print(f'{len(self.motifs)} motivos en esta carta')
            self.show_buttons(True, True, True, False)
            self.show_suit()
        else:
            print('No hay motivos en esta carta')
            self.show_image(self.img)
            self.show_buttons(True, True, False, False)
        
    def show_suit(self):
        
        car = self.cards[self.card_idx]
        
        for ll in car.motifs:
            if ll.motifLabel in SUITS:
                car.realSuit = ll.motifLabel
                break
        for ll in car.motifs:
            if ll.motifLabel in FIGURES:
                car.realFigure = ll.motifLabel
                break
         # Muestra el palo (suit) y la figura de la carta actual
        self.labelSuit.grid_forget()
        self.labelSuit = ttk.Label(self, text=f"Palo: {car.realSuit} ")
        self.labelSuit.grid(column=1, row=1, padx=5, pady=5)
        self.labelFigure.grid_forget()
        self.labelFigure = ttk.Label(self, text=f"Figura: {car.realFigure} ")
        self.labelFigure.grid(column=2, row=1, padx=5, pady=5)
           
    def show_motif(self, motif):
        # Actualiza los datos de la carta actual y la muestra
        self.panel.grid_forget() 
        cnt = motif.contour
        img = self.img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.drawContours(img, [cnt], -1, (0,0,255), 5)
        self.show_image(img)
        self.label.grid_forget()
        self.label = ttk.Label(self, text=f"Motivo actual: {motif.motifLabel} ")
        self.label.grid(column=0, row=1, padx=5, pady=5)
    
    def motif_changed(self, value):
        if (self.motif_idx != -1):
            self.motifs[self.motif_idx].motifLabel = value
            print(f'Motivo {self.motif_idx} de clase {value}')
            self.next_motif()
        
    def next_card(self):
        self.card_idx +=1
        if self.card_idx == len(self.cards):
            self.exit_app()
        self.show_card(self.cards[self.card_idx])
        #self.show_buttons(True, True, True, True)
        
    def prev_card(self):
        self.card_idx -=1
        if self.card_idx <0:
            self.exit_app()
        self.show_card(self.cards[self.card_idx])
        #self.show_buttons(True, True, True, True)
                
    def next_motif(self):
        self.motif_idx +=1
        if self.motif_idx == len(self.motifs):
            self.show_buttons(True, True, False, True)
            self.motif_idx -=1
            # Con el último motivo se obtiene los datos de palo y figura de la carta
            self.show_suit()
        else:
            self.show_motif(self.motifs[self.motif_idx])
            self.show_buttons(True, True, True, True)

        
    def prev_motif(self):
        self.motif_idx -=1
        if self.motif_idx < 0:
            self.show_buttons(True, True, True, False)
            self.motif_idx +=1
        else:
            self.show_motif(self.motifs[self.motif_idx])
            self.show_buttons(True, True, True, True)
            
    def exit_app(self):
        np.savez(filecard, Cartas=self.cards)
        quit()
   
    
    def create_widgets(self):

        self.panel = ttk.Label(self, text = ' ')
        self.panel.grid(column=0, row=0, padx=10, pady=10, columnspan = 3)  

        self.label = ttk.Label(self, text="Indique el tipo de motivo:")
        self.label.grid(column=0, row=1, padx=5, pady=5)

        self.labelSuit = ttk.Label(self, text="Palo:  ")
        self.labelSuit.grid(column=1, row=1, padx=5, pady=5)
        self.labelFigure = ttk.Label(self, text="Figura:  ")
        self.labelFigure.grid(column=2, row=1, padx=5, pady=5)
        
        lfm = ttk.LabelFrame(self, text='Tipos de motivos')
        lfm.grid(column=0, row=2, padx=5, pady=5,columnspan = 3)
        
        for i in range(0, len(MOTIF_LABELS)):
            txt = MOTIF_LABELS[i]
            print(i)
            but = ttk.Button(lfm, text= txt, padding='10 10 10 10', command=lambda x1 = txt: self.motif_changed(x1))
            but.grid(column=i % 6, row= int(i / 6), padx=5, pady=5)
         
        lf = ttk.LabelFrame(self, text='Motivos')
        lf.grid(column=0, row=3, padx=5, pady=5,columnspan = 3)
        
        self.prevMotif_button = ttk.Button(lf, text='MOTIVO previo', command=self.prev_motif)
        self.prevMotif_button.grid(column = 1, row=0, padx=10, pady=5)
        
        self.nextMotif_button = ttk.Button(lf, text='MOTIVO siguiente', command=self.next_motif)
        self.nextMotif_button.grid(column = 2, row=0, padx=10, pady=5)
       
        lf1 = ttk.LabelFrame(self, text='Cartas')
        lf1.grid(column=0, row=4, padx=5, pady=5,columnspan = 3)

        exit_button = ttk.Button(lf1, text='Salir', command=self.exit_app)
        exit_button.grid(column = 0, row=0, pady=5)
        
        self.prevCard_button = ttk.Button(lf1, text='CARTA previa', command=self.prev_card)
        self.prevCard_button.grid(column = 1, row=0, padx=10, pady=5)
                
        self.nextCard_button = ttk.Button(lf1, text='CARTA siguiente', command=self.next_card)
        self.nextCard_button.grid(column = 2, row=0, padx=10, pady=5)
    
    def show_buttons(self, b_nc, b_pc, b_nm, b_pm):
        
        if b_nc: # nextCard_button
            self.nextCard_button.grid(column = 2, row=0, padx=20)
        else:
            self.nextCard_button.grid_forget()
        if b_pc: # prevCard_button
            self.prevCard_button.grid(column = 1, row=0, padx=20)
        else:
            self.prevCard_button.grid_forget()

        if b_nm: # nextMotif_button
            self.nextMotif_button.grid(column = 2, row=0, padx=20)
        else:
            self.nextMotif_button.grid_forget()

        if b_pm: # prevMotif_button
            self.prevMotif_button.grid(column = 1, row=0, padx=20)
        else:
            self.prevMotif_button.grid_forget()

if __name__ == "__main__":
    app = App()
    app.mainloop()
    



