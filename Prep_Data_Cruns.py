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

run_avg = 240
runs = ["2","3","4","5","6","7","8","9","10"]
RUN_AVG = True
HEM = "NH"

def MakeContourPlot(data, title, y, pf):
    plt.figure(figsize=(7.5,6), dpi=300)
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    levels = np.linspace(-maxi, maxi, 17)
    cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.02f', label = "")
    plt.title(title)
    plt.show()

def OpenData_nc(run, r):
    file = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/raw_data/run" + run + "fixed.nc"
    f = nc.Dataset(file, 'r')
    if HEM == "NH":
        temp = f['de_ts'][:99995,:,32:] #20000:99995 to remove spinup
        wind = f['wind'][:99995,:,32:]  #:99995 for all other files
    if HEM == "SH":
        temp = f['de_ts'][:99995,:,:32] #20000:99995 to remove spinup
        wind = f['wind'][:99995,:,:32]  #:99995 for all other files
        temp = np.flip(temp, axis = 2)
        wind = np.flip(wind, axis = 2)

    pf = f["pf"][:]#Pa
    y = f["y"][32:][:] #latitude (degreees)
    f.close()
    
    return(temp,wind,pf,y)

def PrepData(wind_data, temp_data):       
    for_dif_temp = (temp_data[1:,:,:] - temp_data[:-1,:,:]) * 4
    
    wind_field = wind_data[1:]

    u4xDaily = np.array(wind_field) #put into a numpy array
    for_dif_temp = np.array(for_dif_temp)#put into a numpy array
    
    return(u4xDaily, for_dif_temp)

def RunningAvg(wind_data, temp_data, moving_avg):
    wind_runavg = []
    temp_runavg = []
    for i, x in enumerate(wind_data[:-moving_avg]):
        wind_runavg.append(np.mean(wind_data[i:(i+moving_avg)], axis = 0))
        temp_runavg.append(np.mean(temp_data[i:(i+(moving_avg)),:,:], axis = 0))
        
    temp_runavg = np.array(temp_runavg)
    wind_runavg = np.array(wind_runavg)
    
    return(wind_runavg, temp_runavg)

def LatPolyFit(winds, latitude):
    wind850 = winds[:,4,:]  
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

for r, run in enumerate(runs):
    
    #######################
    ###Processing used in## 
    #########DOI###########
    # T4xDaily,u4xDaily,pf,y = OpenData_nc(run, r)
    
    # wind_field, for_dif_temp = PrepData(u4xDaily, T4xDaily) 
    
    # del(u4xDaily, T4xDaily)
  
    # avg_wind_field, avg_Dtemp_field = RunningAvg(wind_field, for_dif_temp, run_avg)

    # max_lat = LatPolyFit(avg_wind_field, y)
    
    # plt.plot(max_lat)
    # plt.show()
    
    # print(np.shape(max_lat), np.shape(avg_wind_field), np.shape(avg_Dtemp_field))
    
    ###########################
    ##Reorganizing Processing## 
    ##########order############
    
    T4xDaily,u4xDaily,pf,y = OpenData_nc(run, r)
    
    wind_field, for_dif_temp = RunningAvg(u4xDaily, T4xDaily, run_avg)
    
    avg_wind_field, avg_Dtemp_field = PrepData(wind_field, for_dif_temp) 
    
    del(u4xDaily, T4xDaily)

    max_lat = LatPolyFit(avg_wind_field, y)
    
    print(np.shape(max_lat), np.shape(avg_wind_field), np.shape(avg_Dtemp_field))

    ###########################
    
    pos_temp = avg_wind_field[(max_lat >= np.mean(max_lat)), :, :]
    pos_temp_avg = np.mean(pos_temp, axis = 0)
    MakeContourPlot(pos_temp_avg, "pos", y, pf)
    plt.show()
   
    neg_temp = avg_wind_field[(max_lat <= np.mean(max_lat)), :, :]
    neg_temp_avg = np.mean(neg_temp, axis = 0)
    MakeContourPlot(neg_temp_avg, "neg", y, pf)
    plt.show()

    pos_temp = avg_Dtemp_field[(max_lat >= np.mean(max_lat)), :, :]
    pos_temp_avg = np.mean(pos_temp, axis = 0)
    MakeContourPlot(pos_temp_avg, "pos", y, pf)
    plt.show()
   
    neg_temp = avg_Dtemp_field[(max_lat <= np.mean(max_lat)), :, :]
    neg_temp_avg = np.mean(neg_temp, axis = 0)
    MakeContourPlot(neg_temp_avg, "neg", y, pf)
    plt.show()
    
    sys.exit(0)
    if HEM == "NH":
        file_save = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/runs_averaged/OneHemisphere_run"+str(run)+"Runningavg" + str(run_avg) + "_lat.nc"
    if HEM == "SH":
        file_save = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/runs_averaged/OneHemisphere_run1"+str(run)+"Runningavg" + str(run_avg) + "_lat.nc"
    print(file_save)
    ts = nc.Dataset(file_save, 'w' , format='NETCDF4')
    ts_pf = ts.createDimension('pf',30)
    ts_y = ts.createDimension('y',32)
    time = ts.createDimension('time',np.size(max_lat))
    output_ts = ts.createDimension('output_ts',np.size(max_lat))
 
    ts_pf = ts.createVariable('pf','f4',('pf'))
    ts_y = ts.createVariable('y','f4',('y'))
    output_ts = ts.createVariable('output_ts','f4',('time'))
    ts_de_ts = ts.createVariable('de_ts','f4',('time','pf','y'))
    windfield = ts.createVariable('wind','f4',('time','pf','y'))

    ts_pf[:] = pf
    ts_y[:]  = y
    ts_de_ts[:,:,:]  = avg_Dtemp_field
    windfield[:,:,:]  = avg_wind_field
    output_ts[:] = max_lat
    
    ts.close()
    print("Finish run" + str(run))
    del(ts_pf,ts_y,ts_de_ts,windfield)   
    
    



