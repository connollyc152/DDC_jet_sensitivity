import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import netCDF4 as nc
import math as mt

import Plots as p


def CreateGaussianBlob(array, y_loc, pf_loc, y_gsize, pf_gsize, anom, y_size, pf_size, y, pf):
    HEATING_blab = np.empty(shape = (pf_size, y_size))
    HEATING_blab.fill(0)
    for l, lat in enumerate(y):
        for h, height in enumerate(pf): 
            HEATING_blab[h,l] = HEATING_blab[h,l] + ((mt.exp(-((lat - y_loc)**2/(y_gsize)**2) - ((height - pf_loc)**2/(pf_gsize)**2))) * (anom))   
    return(array + HEATING_blab)

def GaussianBlab(anom, model, y, pf, initAO, y_size, pf_size, out_mean, out_std, in_mean, in_std, FILL):
    print("COMPLETING GAUSSIANBLAB")
    data  = np.empty(shape = (pf_size, y_size))
    Predicted_AO_array = np.empty(shape=(pf_size, y_size))
    Predicted_AO_array.fill(0)
    Uncertainty_AO_array = np.empty(shape=(pf_size, y_size))
    Uncertainty_AO_array.fill(0)
    label = np.empty(shape=(1))
    label.fill(initAO)

    latSize = 15
    heightSize = 200
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            data.fill(FILL)
            data = CreateGaussianBlob(data, AOlat, AOheight, latSize, heightSize, anom, y_size, pf_size, y, pf)
  
            data_stand = (data - in_mean)/in_std 
            dataInTrain = data_stand.reshape(-1, pf_size, y_size, 1)
            
            dataInTrain = dataInTrain.reshape(1, np.size(pf), np.size(y), 1)
            
            pred = model.predict([dataInTrain,label])
            Uncertainty_AO_array[AOh, AOl] = pred[:,1]
            Predicted_AO_array[AOh,AOl] =(pred[:,0] * out_std) + out_mean
    p.MakeContourPlot(Predicted_AO_array, "", y, pf)
    
    sample_data  = np.empty(shape = (pf_size, y_size))
    sample_data.fill(0)
    for l, lat in enumerate(y):
        for h, height in enumerate(pf): 
            sample_data[h,l] = (mt.exp(-((550 - height)**2/(heightSize)**2) - ((10 - lat)**2/(latSize)**2))) * anom 
    p.TwoByOnePlot2(sample_data, Predicted_AO_array, Uncertainty_AO_array, "", y , pf)
    return(Predicted_AO_array)
