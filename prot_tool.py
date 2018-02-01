#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:57:56 2018

@author: soumyadipghosh
"""

import sys
from PyQt5 import QtWidgets, uic
import pickle
import pandas as pd
import numpy as np
 
qtCreatorFile = "ProteinDisorder.ui"
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class Ui(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('ProteinDisorder.ui', self)
        
        filename = 'mlp_model.sav'
        self.mlp = pickle.load(open(filename, 'rb'))
        filename_1="minmaxscaler.sav"
        self.scaler = pickle.load(open(filename_1, 'rb'))
        
        df = pd.read_excel("data_final.xlsx", sheetname=None)
        self.prot_data = df["Sheet1"]
        self.prot_data.drop(self.prot_data.columns[len(self.prot_data.columns)-1], axis=1, inplace=True)
        self.prot_data.replace([np.inf, -np.inf], np.nan)
        self.prot_data.dropna(inplace=True)
        
        labels = np.array(self.prot_data.columns.values,dtype="str")
        self.prot_data[labels[21:]] = self.prot_data[labels[21:]].astype(str).replace("-", "0").astype(float)
        
        self.uni_protid = self.prot_data["UniProt ID"].tolist()
        self.prot_data.drop(["UniProt ID"], axis=1, inplace=True)
        self.prot_data.drop(["LenDR"], axis=1, inplace=True)
        self.prot_data = self.prot_data.as_matrix()

        self.initUI()
        self.show()
 
    def initUI(self):
        self.setWindowTitle("UniProt Protein Disorder")
        self.cal_button.clicked.connect(self.buttonClicked) 
        
    def buttonClicked(self):
        input_id = self.input.text()
        idx = self.uni_protid.index(input_id)
        in_data = self.prot_data[idx]
        in_data = self.scaler.transform([in_data])
        pred = self.mlp.predict(in_data)
        self.output.setText(str(pred))
        

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater) # if using IPython Console
    window = Ui()
    sys.exit(app.exec_())
    