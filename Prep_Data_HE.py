"""Used to prep the forced Heating Experiement output
The hearing experiements saved in doi (unavailable at this time)
is derived from this script

Raw data to run this script is available upon request."""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import matplotlib.colors
import matplotlib.colors as mcolors

#CREATE COLORBAR###################
lower = plt.cm.bwr(np.linspace(0,.5, 45))
white = plt.cm.bwr(np.ones(10)*0.5)
upper = plt.cm.bwr(np.linspace(0.5, 1, 45))
colors = np.vstack((lower,white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

colors = [mcolors.to_rgb('#401404'),mcolors.to_rgb('#BFB68A'),mcolors.to_rgb('#81A64B'),mcolors.to_rgb('#86A0A6'),mcolors.to_rgb('#D9CCCD')]
####################

runs = ["TU_TTP1_PS_TTN1", "TU_TTP1_PS_TTNP5",
              "TU_TTP1_PS_TT0", "TU_TTP1_PS_TTP5",
              "TU_TTP1_PS_TT1", "TU_TT0_PS_TTN1",
              "TU_TT0_PS_TTNP5", "TU_TT0_PS_TTP5",
              "TU_TT0_PS_TT1","TU_TTNP1_PS_TTN1",
              "TU_TTNP1_PS_TTNP5","TU_TTNP1_PS_TT0",
              "TU_TTNP1_PS_TTP5", "TU_TTNP1_PS_TT1",
              "30L_15LS_1000HPA_200HPAS_P25K", "30L_15LS_700HPA_200HPAS_P25K",
              "60L_15LS_1000HPA_200HPAS_P25K", "60L_15LS_700HPA_200HPAS_P25K",]

def MakeContourPlot(data, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    levels = np.linspace(-maxi, maxi, 17)
    cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.02f', label = "")
    plt.title(title)
    plt.show()

def OpenData_nc(run, r):
    file = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/raw_heatingFiles/" + runs[r] + ".nc"
    f = nc.Dataset(file, 'r')
    temp = f['temp'][3998:19998,:,32:]  #3998:19998
    wind = f['wind'][3998:19998,:,32:] 

    pf = f["pf"][:]#Pa
    y = f["y"][32:][:] #latitude (degreees)
    f.close()

    print("data remaining after removing spin-up")
    print(np.shape(np.mean(np.mean(temp, axis = 1), axis = 1)))
    plt.plot(np.mean(np.mean(temp, axis = 1), axis = 1))
    plt.show()
    
    return(temp,wind,pf,y)

def PrepData(wind_data, temp_data):       
    for_dif_temp = (temp_data[1:,:,:] - temp_data[:-1,:,:]) * 4
    
    wind_field = wind_data[1:,:,:]

    u4xDaily = np.array(wind_field) #put into a numpy array
    for_dif_temp = np.array(for_dif_temp)#put into a numpy array
    
    return(u4xDaily, for_dif_temp)

def LatPolyFit(winds, latitude):
    wind850 = winds[:,4,:]  # 25(850)  33Pedram
    max_lat = np.empty(shape = int(np.size(wind850[:,0])))
    for w in np.arange(0,int(np.size(wind850[:,0])), 1):
        max_value = np.max(wind850[w,:])
        result = np.where(wind850[w,:] == max_value)[0]

        result = result[0]
        wind_sample = wind850[w,(int(result) - 2):(int(result) + 3)]
        lat_sample = latitude[(int(result) - 2):(int(result) + 3)]
        
        fit = np.polyfit(lat_sample, wind_sample, 2)
        a = fit[0]
        b = fit[1]
        c = fit[2]

        high_res_lat = np.arange(min(lat_sample), max(lat_sample), .1)
        fit_equation = a * np.square(high_res_lat) + b * high_res_lat + c
        max_value = np.max(fit_equation)
        
        result = np.array(np.where(fit_equation == max_value)[0])
        max_lat[w] = high_res_lat[int(result)]
        
    return(max_lat)

total_temp = []
for r, run in enumerate(runs):

    T4xDaily,u4xDaily,pf,y = OpenData_nc(run, r)
    
    wind_field, for_dif_temp = PrepData(u4xDaily, T4xDaily) 
    
    del(T4xDaily)

    jet_loc = LatPolyFit(wind_field, y)
 
    pos_temp = wind_field[(jet_loc >= np.mean(jet_loc)), :, :]
    pos_temp_avg = np.mean(pos_temp, axis = 0)
    MakeContourPlot(pos_temp_avg, "pos", y, pf)
    plt.show()
   
    neg_temp = wind_field[(jet_loc <= np.mean(jet_loc)), :, :]
    neg_temp_avg = np.mean(neg_temp, axis = 0)
    MakeContourPlot(neg_temp_avg, "neg", y, pf)
    plt.show()

    pos_temp = for_dif_temp[(jet_loc >= np.mean(jet_loc)), :, :]
    pos_temp_avg = np.mean(pos_temp, axis = 0)
    MakeContourPlot(pos_temp_avg, "pos", y, pf)
    plt.show()
   
    neg_temp = for_dif_temp[(jet_loc <= np.mean(jet_loc)), :, :]
    neg_temp_avg = np.mean(neg_temp, axis = 0)
    MakeContourPlot(neg_temp_avg, "neg", y, pf)
    plt.show()
    
    plt.plot(jet_loc)
    
    ts = nc.Dataset("/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/Averaged_HeatingFiles/Heating_Experiment_"+str(run)+".nc", 'w' , format='NETCDF4')
    ts_pf = ts.createDimension('pf',30)
    ts_y = ts.createDimension('y',32)
    time = ts.createDimension('time',15999)
    output_ts = ts.createDimension('output_ts',np.size(jet_loc))
 
    ts_pf = ts.createVariable('pf','f4',('pf'))
    ts_y = ts.createVariable('y','f4',('y'))
    output_ts = ts.createVariable('output_ts','f4',('time'))
    ts_de_ts = ts.createVariable('de_ts','f4',('time','pf','y'))
    windfield = ts.createVariable('wind','f4',('time','pf','y'))

    ts_pf[:] = pf
    ts_y[:]  = y
    ts_de_ts[:,:,:]  = for_dif_temp
    windfield[:,:,:]  = wind_field
    output_ts[:] = jet_loc
    
    ts.close()
    print("Finish run" + str(run))
    del(ts_pf,ts_y,ts_de_ts,windfield)   
    
    



