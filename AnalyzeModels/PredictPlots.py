import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import netCDF4 as nc
import matplotlib.colors as colors
import tensorflow as tf
import random
import math as mt

import Plots as p
import Stats as s

import keras
import tensorflow_probability as tfp
#CREATE COLORBAR###################
lower = plt.cm.bwr(np.linspace(0,.5, 45))
white = plt.cm.bwr(np.ones(10)*0.5)
upper = plt.cm.bwr(np.linspace(0.5, 1, 45))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

def Saliency(amp, mode, temp, AO, labels, pred, model, y, pf, mean, std, in_mean, in_std):
    zero_array = np.empty(shape=(25, 32))
    zero_array.fill(0)
    if mode == 'ONEINPUT':
        n = 1
        Full_predicted_array = np.empty(shape=(1,25, 32))
        samples = [500]
    if mode == 'RANDOM':
        n = 1
        Full_predicted_array = np.empty(shape=(n,25, 32))
        samples = random.sample(list(np.arange(0,int(np.size(temp[:,0,0])),1)), n)

    for s, sample in enumerate(samples):
        print("COMPLETELING SAILENCY SAMPLE " + str(s))
        temp_sample = temp[sample,:,:]
        label_sample2 = labels[sample]
        
        label_sample = np.empty(shape = (1,))
        label_sample[0] = label_sample2
        prediction_sample = pred[sample]
        
        Predicted_AO_array = np.empty(shape=(25, 32))
        Predicted_AO_array.fill(0)
        data  = np.empty(shape = (25, 32))
        data[:,:] = temp_sample

        for AOh, AOheight in enumerate(pf):
            for AOl, AOlat in enumerate(y):
                data[:,:] = temp_sample
                data[AOh,AOl] = data[AOh,AOl] + amp
        
                data_stand = (data - in_mean)/in_std
                dataInTrain = data_stand.reshape(-1, 25, 32, 1)
                
                pred = model.predict([dataInTrain, label_sample])
                predvalue = ((pred[:,0] * std) + mean)

                Predicted_AO_array[AOh,AOl] = predvalue - prediction_sample

        p.MakeContourPlot(Predicted_AO_array,"Difference from original prediction: " + str(prediction_sample),y, pf)
        Full_predicted_array[s,:,:] = Predicted_AO_array
        
    average_saliency = np.mean(Full_predicted_array, axis = 0)
    p.MakeContourPlot(average_saliency,"How heating changes predicted AO Value", y, pf)
    p.MakeContourPlotColorMesh(average_saliency,"How heating changes predicted AO Value", y, pf)

def Nmaxelements(data, N):
    max_list = np.argsort(data)[-N:]
    return(max_list)

def Nminelements(data, N):
    max_list = np.argsort(data)[:N]
    return(max_list)
    
def PullingSamples(n, temp, labels, true, pred, Initmin, Initmax, y, pf, Ttype, metric):
    Predicted_AO_array = np.empty(shape=(25, 32))
    Predicted_AO_array.fill(0)

    pred = pred[labels > Initmin]
    temp = temp[labels > Initmin]
    labels = labels[labels > Initmin]

    pred = pred[labels < Initmax]
    temp = temp[labels < Initmax]
    labels = labels[labels < Initmax]
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            if Ttype == 'max':
                max_list = Nmaxelements(temp[:,AOh,AOl], n)
            if Ttype == 'min':
                max_list = Nminelements(temp[:,AOh,AOl], n)
            Predicted_AO_value = np.empty(shape = (n))
            for i, m in enumerate(max_list):
                Predicted_AO_value[i] = pred[m]
            
            if metric == 'mean':  
                calculated_array = np.mean(Predicted_AO_value, axis = 0)
            if metric == 'median':  
                calculated_array = np.median(Predicted_AO_value, axis = 0)
            if metric == 'std':  
                calculated_array = np.std(Predicted_AO_value, axis = 0)
            if metric == 'var':  
                calculated_array = np.var(Predicted_AO_value, axis = 0)
            if metric == 'min':  
                calculated_array = np.min(Predicted_AO_value, axis = 0)
            if metric == 'max':  
                calculated_array = np.max(Predicted_AO_value, axis = 0)
            
            Predicted_AO_array[AOh,AOl] = calculated_array
    return(Predicted_AO_array)

def PullingSamplesTrue(n, temp, labels, true, pred, Initmin, Initmax, y, pf, Ttype, metric):
    Predicted_AO_array = np.empty(shape=(25,32))
    Predicted_AO_array.fill(0)

    pred = pred[labels > Initmin]
    temp = temp[labels > Initmin]
    labels = labels[labels > Initmin]

    pred = pred[labels < Initmax]
    temp = temp[labels < Initmax]
    labels = labels[labels < Initmax]
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            if Ttype == 'max':
                max_list = Nmaxelements(temp[:,AOh,AOl], n)
            if Ttype == 'min':
                max_list = Nminelements(temp[:,AOh,AOl], n)
            Predicted_AO_value = np.empty(shape = (n))
            for i, m in enumerate(max_list):          
                Predicted_AO_value[i] = true[m]
            
            if metric == 'mean':  
                calculated_array = np.mean(Predicted_AO_value, axis = 0)
            if metric == 'median':  
                calculated_array = np.median(Predicted_AO_value, axis = 0)
            if metric == 'std':  
                calculated_array = np.std(Predicted_AO_value, axis = 0)
            if metric == 'var':  
                calculated_array = np.var(Predicted_AO_value, axis = 0)
            if metric == 'min':  
                calculated_array = np.min(Predicted_AO_value, axis = 0)
            if metric == 'max':  
                calculated_array = np.max(Predicted_AO_value, axis = 0)
            
            Predicted_AO_array[AOh,AOl] = calculated_array
    return(Predicted_AO_array)

def PullingSamplesReturnArray(n, temp, labels, true, pred, Initmin, Initmax, y, pf, Ttype):
    predicted_array = []
    true_array = []
    
    pred = pred[labels > Initmin]
    temp = temp[labels > Initmin]
    labels = labels[labels > Initmin]

    pred = pred[labels < Initmax]
    temp = temp[labels < Initmax]
    labels = labels[labels < Initmax]
    
    for AOl, AOlat in enumerate(y[:-1]):
        for AOh, AOheight in enumerate(pf[:-1]):
            if Ttype == 'max':
                max_list = Nmaxelements(temp[:,AOh,AOl], n)
            if Ttype == 'min':
                max_list = Nminelements(temp[:,AOh,AOl], n)
            for i, m in enumerate(max_list):
                predicted_array.append(pred[m])
                true_array.append(true[m])
            
    return(predicted_array, true_array)

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
    
    # latSize = 10
    # heightSize = 100

    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            data.fill(FILL)
            # data = FILL

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
    #p.TwoByOnePlot(sample_data, Uncertainty_AO_array, "Uncertainty", y , pf)
    
    return(Predicted_AO_array)
                
def PullingSamplesPercent(p, temp, labels, true, pred, Initmin, Initmax, y, pf, Ttype, metric):
    n = int(int(np.size(labels)) * p)
    Predicted_AO_array = np.empty(shape=(25, 32))
    Predicted_AO_array.fill(0)

    pred = pred[labels > Initmin]
    temp = temp[labels > Initmin]
    labels = labels[labels > Initmin]

    pred = pred[labels < Initmax]
    temp = temp[labels < Initmax]
    labels = labels[labels < Initmax]
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            if Ttype == 'max':
                max_list = Nmaxelements(temp[:,AOh,AOl], n)
            if Ttype == 'min':
                max_list = Nminelements(temp[:,AOh,AOl], n)
            Predicted_AO_value = np.empty(shape = (n))
            for i, m in enumerate(max_list):
                Predicted_AO_value[i] = pred[m]
            
            if metric == 'mean':  
                calculated_array = np.mean(Predicted_AO_value, axis = 0)
            if metric == 'median':  
                calculated_array = np.median(Predicted_AO_value, axis = 0)
            if metric == 'std':  
                calculated_array = np.std(Predicted_AO_value, axis = 0)
            if metric == 'var':  
                calculated_array = np.var(Predicted_AO_value, axis = 0)
            if metric == 'min':  
                calculated_array = np.min(Predicted_AO_value, axis = 0)
            if metric == 'max':  
                calculated_array = np.max(Predicted_AO_value, axis = 0)
            
            Predicted_AO_array[AOh,AOl] = calculated_array
    return(Predicted_AO_array)
   
def Nrandomelements(data, N):
    max_list = random.choice(data, N)
    return(max_list)

def mae_avg(test, predict):
    return (np.mean(test-predict))
 
            
def GaussianBlabCorrelation(anom, model, y, pf, initAO, y_size, pf_size, traingdata):
    data  = np.empty(shape = (pf_size, y_size))
    data.fill(0)
    Predicted_AO_array = np.empty(shape=(pf_size, y_size))
    Predicted_AO_array.fill(0)
    initAI = np.array(initAO)
    BLOB_HEIGHT = 150
    BLOB_LENGTH = 10
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            data.fill(0)
    
            for l, lat in enumerate(y):
                for h, height in enumerate(pf): 
                    data[h,l] = (mt.exp(-((AOheight - height)**2/(BLOB_HEIGHT)**2) - ((AOlat - lat)**2/(BLOB_LENGTH)**2))) * anom 
            
            Predicted_AO_array[AOh,AOl] = s.MaxSpatialCorrelation(data, traingdata, pf_size, y_size, AOh, AOl )
        print(str(AOl) + " out of " + str(np.size(y)))
            
    
    sample_data  = np.empty(shape = (pf_size, y_size))
    sample_data.fill(0)
    for l, lat in enumerate(y):
        for h, height in enumerate(pf): 
            sample_data[h,l] = (mt.exp(-((500 - height)**2/(BLOB_HEIGHT)**2) - ((10 - lat)**2/(BLOB_LENGTH)**2))) * anom 
    p.TwoByOnePlot(sample_data, Predicted_AO_array, "", y , pf)

    return(Predicted_AO_array)

def PullingSamplesAddingGaussianBlab(n, temp, labels, true, pred, Initmin, Initmax, y, pf, pf_size, y_size, anom, model, out_std, out_mean):
    data  = np.empty(shape = (pf_size, y_size))
    data.fill(0)
    Predicted_AO_array = np.empty(shape=(pf_size, y_size))
    Predicted_AO_array.fill(0)
    lab = np.empty(shape=(1))
    lab.fill(0)
    
    for AOl, AOlat in enumerate(y):
        for AOh, AOheight in enumerate(pf):
            data.fill(0)    
            for l, lat in enumerate(y):
                for h, height in enumerate(pf): 
                    data[h,l] = data[h,l] + ((mt.exp(-((lat - AOlat)**2/(150)**2) - ((height - AOheight)**2/(10)**2))) * (anom))

            max_list = random.sample(list(np.arange(0,int(np.size(temp[:,0,0])),1)), n)

            true_pred  = np.empty(shape = (n))
            pred  = np.empty(shape = (n))
            i = 0
            for sample in max_list:
                sam = temp[sample,:,:]
                true_pred[i] = true[sample]
                lab.fill(labels[sample])
                sam = sam + data
                
                sam_stand = (sam - 0)/.1 
                sam_standF = sam_stand.reshape(-1, pf_size, y_size, 1)
                
                predi = model.predict([sam_standF,lab])
                pred[i] = ((pred[:,0] * out_std) + out_mean)
                i = i + 1
                
                dif = pred - true_pred
                avg_dif = np.mean(dif)
        
            Predicted_AO_array[AOh,AOl] = avg_dif
    return(Predicted_AO_array)
      