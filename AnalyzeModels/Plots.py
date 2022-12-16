import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.cm as cm
tmap_uncertainty = "bone" #cmr.savanna_r

lower = plt.cm.RdBu_r(np.linspace(0,.49, 49))
white = plt.cm.RdBu_r(np.ones(2)*0.5)
upper = plt.cm.RdBu_r(np.linspace(0.51, 1, 49))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

cmap = matplotlib.cm.get_cmap('Reds')
####################
white = plt.cm.bwr(np.ones(10)*0.5)
upper = plt.cm.bwr(np.linspace(0.5, 1, 90))
colors2 = np.vstack((white, upper))
tmap_zero = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors2)
####################
ModelColor = "black"
SteadyEvolutionColor = "#35D6E8" #"#81C8FF"
PersistenceColor = "orange"
#####################

axis_font = {'fontname':'Noto Sans JP', 'size':'13', 'color':'black', 'weight':'normal'}

predict_DeltaAO = True
fontPath = '/Users/cconn/Library/Fonts/'
import matplotlib.font_manager as fm
for font in fm.findSystemFonts(fontPath):
    fm.fontManager.addfont(font)


label_font = {'fontname':'Noto Sans JP', 'size':'8', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
title_font = {'fontname':'Noto Sans JP', 'size':'18', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
axis_font = {'fontname':'Noto Sans JP', 'size':'30', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
legend_font = {'family':'Noto Sans JP', 'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
axisTick_font = {'family':'Noto Sans JP', 'size':'15', 'color':'black', 'weight':'light'}


def mae_avg(test, predict):
    return np.mean(np.abs(np.ndarray.flatten(test)-np.ndarray.flatten(predict)))

def mae(test, predict):
    return np.mean(np.abs(test-predict))

def MakeContourPlotColorMesh(data, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    colormap = tmap
    if np.min(data) > 0:
        mini = 0
        colormap = tmap_zero
    if np.min(data) < 0:
        mini = -maxi
        colormap = tmap  
    
    plt.figure(figsize=(8,5), dpi=300)
    cs = plt.pcolormesh(X, Y, data ,cmap=colormap, vmin = mini, vmax = maxi, shading='auto' )
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.3f')
    plt.title(title, **title_font)
    plt.ylabel("Pressure Height (hPa)", **label_font)
    plt.xlabel("Latitude", **label_font, labelpad=12)
    #plt.show()

def MakeContourPlot(data, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    plt.figure(figsize=(7.5,6), dpi=300)
    maxi = np.max(np.abs(data))
    levels = np.linspace(-maxi, maxi, 17)
    # levels = np.linspace(0, .5, 17)
    cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.ylim(np.max(pf),np.min(pf))
    # plt.colorbar(cs, format='%.0f', label = "")
    plt.colorbar(cs, format='%.2f', label = "")
    plt.title(title)
    #plt.show()

def MakeContourPlotLog(data, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    levels = np.linspace(-maxi, maxi, 17)
    cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.yscale('log')
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.2f', label = "")
    plt.title(title)
    plt.show()

def MakeContourPlot_outline(data, title, y, pf, color, levels):
    X, Y = np.meshgrid(y, pf)
    plt.figure(figsize=(7.5,6), dpi=300)
    cs = plt.contour(y, pf, data, colors="black", levels = levels)#[0.1, .25, .4])#, linestyles='dashed')
    plt.clabel(cs, inline=True, fontsize=10)
    plt.ylim(np.max(pf),np.min(pf))
    plt.title(title)
    plt.show()

def MakeContourPlotNoFill(data, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    #maxi = np.max(np.abs(data))
    #levels = np.linspace(-maxi, maxi, 17)
    # cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.figure(figsize=(7.5,6), dpi=300)
    cs = plt.contour(X, Y, data, colors='black', levels = [-4, -.01, .01, 4], linestyles='-')
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.02f', label = "")
   # plt.colorbar(cs, label = "")
    plt.title(title)
    plt.show()

def TwoD_PDF_dist(predicted, actual):
    # MAE_train2 = np.round(mae(actual,predicted),3)
    # plt.axline((0, .10621649904093333), slope=0.5, color='green', label='true slope')
    plot = sns.jointplot(x = actual,y = predicted, kind='kde', cmap="Greys", shade=True,) #, kind='kde'
    
    plot.ax_joint.plot([-10,10],[((-10 * .5) + -0.1),((10 * .5) + -0.1)], color = 'gray')
    plot.ax_joint.plot([-10,10],[-10,10], color = 'gray')
    plot.ax_joint.set_xlim(-10,10)
    plot.ax_joint.set_ylim(-10,10)
    
    plt.show()

import matplotlib.colors as mcolors
def truncate_colormap(cmap2, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap2.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap2.name, a=minval, b=maxval),
         cmap2(np.linspace(minval, maxval, n)))
    return new_cmap

greys = plt.get_cmap('Greys')
greys_sub = truncate_colormap(greys, 0.15, 1)
def TwoD_PDF(predicted, actual):
    plt.figure(figsize=(6.5,6), dpi=300)
    # MAE_train2 = np.round(mae(actual,predicted),3)
    # plt.axline((0, .10621649904093333), slope=0.5, color='green', label='true slope')
    plot = sns.kdeplot(x = actual,y = predicted, cmap=greys_sub, shade=True, levels=[0.001,0.03,0.06, 0.1,0.2,0.3,0.4,0.6,0.8,1], cbar=True,) #, kind='kde'
    # plot = sns.kdeplot(x = actual,y = predicted, cmap=greys_sub, shade=True, shade_lowest = True) #, kind='kde'
    plot.set_aspect('equal')
    plot.plot([-11,11],[((-11 * .5) + -0.1),((11 * .5) + -0.1)], color = 'gray')
    plot.plot([-11,11],[-11,11], color = 'gray')
    plot.set_xlim(-11,11)
    plot.set_ylim(-11,11)
    
    
    plt.show()

def ScatterPlot(predicted, true, title):
    MAE_train2 = np.round(mae_avg(predicted,true),3)
    plt.figure(figsize=(6.5,6), dpi=300)
    plt.text(8,-9.4, str(MAE_train2), c="blue")
    plt.plot(true,predicted,'o',markersize=1, label='', color = "blue")
    plt.plot([-10, 10],[-10, 10], color = 'gray')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title(title, **title_font)
    plt.xlabel('True', **label_font, labelpad=12)
    plt.ylabel('Predicted', **label_font)
    #plt.show()

import scipy
def ThreeByOnePanelScatter_trend(true, predicted, title, x_points, y_points):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22,6), dpi=100)

    ax1.plot(true, predicted[0],'o',markersize=1, label='', color = ModelColor)
    # print(scipy.stats.spearmanr(true, predicted[0]))
    MAE_V = np.round(mae_avg(predicted[0],true),3)
    ax1.text(8,-9.3, str(MAE_V), c=ModelColor)
    ax1.plot([-11,11],[-11,11], color = 'gray')
    ax1.set_xlim(-11,11)
    ax1.set_ylim(-11,11)
    ax1.set_xlabel("True", **label_font, labelpad=12)
    ax1.set_ylabel("Predicted", **label_font)
    ax1.set_title(title[0], **title_font)
    ax1.plot(x_points, y_points, 'o', markersize=15, color = 'limegreen')
    # ax1.axline((0, intercept), slope=slope, color='limegreen', label='ideal slope')
    plt.setp(ax1.get_xticklabels(), **axisTick_font)
    plt.setp(ax1.get_yticklabels(), **axisTick_font)
    ax1.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax1.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax1.set_aspect('equal')
    
    ax2.plot(true, predicted[1],'o',markersize=1, label='', color = SteadyEvolutionColor)
    # print(scipy.stats.spearmanr(true, predicted[1]))
    MAE_V = np.round(mae_avg(predicted[1],true),3)
    ax2.text(8,-9.3, str(MAE_V), c=SteadyEvolutionColor)
    ax2.plot([-11,11],[-11,11], color = 'gray')
    ax2.set_xlim(-11,11)
    ax2.set_ylim(-11,11)
    ax2.set_xlabel("True", **label_font, labelpad=12)
    ax2.set_ylabel("Predicted", **label_font)
    ax2.set_title(title[1], **title_font)
    plt.setp(ax2.get_xticklabels(), **axisTick_font)
    plt.setp(ax2.get_yticklabels(), **axisTick_font)
    ax2.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax2.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax2.set_aspect('equal')
 
    ax3.plot(true, predicted[2],'o',markersize=1, label='', color = PersistenceColor)
    # print(scipy.stats.spearmanr(true, predicted[2]))
    MAE_V = np.round(mae_avg(predicted[2],true),3)
    ax3.text(8,-9.3, str(MAE_V), c=PersistenceColor)
    ax3.plot([-11,11],[-11,11], color = 'gray')
    ax3.set_xlim(-11,11)
    ax3.set_ylim(-11,11)
    ax3.set_xlabel("True", **label_font, labelpad=12)
    ax3.set_ylabel("Predicted", **label_font)
    ax3.set_title(title[2], **title_font)
    plt.setp(ax3.get_xticklabels(), **axisTick_font)
    plt.setp(ax3.get_yticklabels(), **axisTick_font)
    ax3.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax3.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax3.set_aspect('equal')
    
    plt.show()

def ThreeByOnePanelScatter(true, predicted, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22,6), dpi=1000)

    ax1.plot(true, predicted[0],'o',markersize=1, label='', color = ModelColor)
    MAE_V = np.round(mae_avg(predicted[0],true),3)
    # MAE_V = np.round(mape(predicted[0],true),3)
    ax1.axline((0, -.04), slope=.51, color='limegreen', label='ideal slope')
    ax1.text(8,-9.3, str(MAE_V), c=ModelColor)
    ax1.plot([-11,11],[-11,11], color = 'gray')
    ax1.set_xlim(-11,11)
    ax1.set_ylim(-11,11)
    ax1.set_xlabel("True", **label_font, labelpad=12)
    ax1.set_ylabel("Predicted", **label_font)
    ax1.set_title(title[0], **title_font)
    plt.setp(ax1.get_xticklabels(), **axisTick_font)
    plt.setp(ax1.get_yticklabels(), **axisTick_font)
    ax1.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax1.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax1.set_aspect('equal')
    
    ax2.plot(true, predicted[1],'o',markersize=1, label='', color = SteadyEvolutionColor)
    MAE_V = np.round(mae_avg(predicted[1],true),3)
    # MAE_V = np.round(mape(predicted[1],true),3)
    ax2.text(8,-9.3, str(MAE_V), c=SteadyEvolutionColor)
    ax2.plot([-11,11],[-11,11], color = 'gray')
    ax2.set_xlim(-11,11)
    ax2.set_ylim(-11,11)
    ax2.set_xlabel("True", **label_font, labelpad=12)
    ax2.set_ylabel("Predicted", **label_font)
    ax2.set_title(title[1], **title_font)
    plt.setp(ax2.get_xticklabels(), **axisTick_font)
    plt.setp(ax2.get_yticklabels(), **axisTick_font)
    ax2.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax2.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax2.set_aspect('equal')
 
    ax3.plot(true, predicted[2],'o',markersize=1, label='', color = PersistenceColor)
    MAE_V = np.round(mae_avg(predicted[2],true),3)
    # MAE_V = np.round(mape(predicted[2],true),3)
    ax3.text(8,-9.3, str(MAE_V), c=PersistenceColor)
    ax3.plot([-11,11],[-11,11], color = 'gray')
    ax3.set_xlim(-11,11)
    ax3.set_ylim(-11,11)
    ax3.set_xlabel("True", **label_font, labelpad=12)
    ax3.set_ylabel("Predicted", **label_font)
    ax3.set_title(title[2], **title_font)
    plt.setp(ax3.get_xticklabels(), **axisTick_font)
    plt.setp(ax3.get_yticklabels(), **axisTick_font)
    ax3.set_xticks(ticks = [-10, -5, 0, 5, 10])
    ax3.set_yticks(ticks = [-10, -5, 0, 5, 10])
    ax3.set_aspect('equal')
    
    plt.show()

def CreateSamplePlot(samples, temp, AO, labels, pred, y, pf, lag):
    # samples = [680, 48972, 29787, 38225, 37975]
    for i, m in enumerate(samples):
        array = temp[m,:,:]
        maxi = np.max(np.abs(array))
    
        f, (ax1, ax2) = plt.subplots(2,1, figsize=[5,10], dpi = 300)#, sharey=True)
        X, Y = np.meshgrid(y,pf)
        xs = ax1.contourf(X,Y,array,cmap=tmap,levels = np.linspace(-maxi, maxi, 15))
        plt.colorbar(xs,label="K/day", orientation="horizontal", format='%.3f',)
        ax1.set_title("Predicted \u0394 Latitude: " + str(pred[m]) + "\n True \u0394 Latitude: " + str(AO[m]) + "\nInput Latitude: " + str(labels[m]))
        ax1.set_xlabel("Latitude")
        ax1.set_ylabel("Height (Pa)")
        ax1.set_ylim(975,200)

        print(m)
        plt.plot(labels[m-100:m+(100 + lag)])
        ax2.set_ylabel("Latitude")
        plt.plot((100), labels[m], 'o', c = 'red')
        plt.plot((100 + 360), (labels[m] + pred[m]), 'o', c = 'black')
        plt.plot((100 + 360), (labels[m] + AO[m]), 'o', c = 'red')
        plt.xlabel("timesteps (6hr)")
        
        plt.ylim(35,50)
        plt.show()
        

def HistogramWHeightNormBySample(AO, weight, bins, title, label, color, width, LABELS):
    # bins = np.linspace(mini,maxi,bins_n)
    n = np.histogram(AO, bins = bins, weights=weight)
    n2 = np.histogram(AO, bins = bins)
    n2[0][n2[0]==0] = 1
    values = n[0] / n2[0]
    labels = list(n2[0])
    n, bins1, patches = plt.hist(bins[:-1], weights=values, bins = bins, label=label, histtype="step", color = color, linewidth=width)
    if LABELS:
        for i, N in enumerate(n[:]):
            # plt.text(bins1[i], N + .005, str(np.round(labels[i],0)), **label_font)
            plt.text(bins1[i] + 1, 0, str(np.round(labels[i],0)), horizontalalignment='center', **label_font)
    # plt.legend(loc='upper right', prop={'family':'Assistant', 'size':'12'})    
    plt.xticks(**axisTick_font) #[25, 30, 35, 40, 42.4, 45, 50],
    plt.yticks(**axisTick_font)
    plt.title(title)


def Histogram(AO, bins_n, title, label, color, width):
    bins = np.linspace(30,50,bins_n)
    n, bins1, patches = plt.hist(AO,bins = bins, label=label, histtype="step", color=color, linewidth=width)
    plt.xlabel("\u0394AO", labelpad=20, **axis_font)
    plt.ylabel("count", **axis_font)
    plt.legend()
    plt.title(title)
    # plt.show()
    return(n, bins1, patches)
    
def FourPanelContour(data, plot_labels, title, y, pf):
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    colormap = tmap
    if np.min(data) > 0:
        levels = np.linspace(0, maxi, 21)
        colormap = tmap_zero
    if np.min(data) < 0:
        levels = np.linspace(-maxi, maxi, 21)
        colormap = tmap
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6,6))
    fig.suptitle(title, **title_font, y=.9)
    
    ax1.set_ylim(np.max(pf),np.min(pf))
    ax1.contourf(X, Y, data[0] ,cmap=colormap, levels = levels)
    ax1.text(4,50, plot_labels[0], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax2.set_ylim(np.max(pf),np.min(pf))
    ax2.contourf(X, Y, data[1] ,cmap=colormap, levels = levels)
    ax2.text(4,50, plot_labels[1], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax3.set_ylim(np.max(pf),np.min(pf))
    ax3.contourf(X, Y, data[2] ,cmap=colormap, levels = levels)
    ax3.text(4,50, plot_labels[2], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax4.set_ylim(np.max(pf),np.min(pf))
    xs = ax4.contourf(X, Y, data[3] ,cmap=colormap, levels = levels)
    ax4.text(4,50, plot_labels[3], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    for ax in fig.get_axes():
        ax.label_outer()

    cbar_ax = fig.add_axes([0.05, -0.02, 0.85, 0.04])
    plt.colorbar(xs,label="K/day", orientation="horizontal", cax = cbar_ax)       
     
def SixPanelContour(data, plot_labels, title, y, pf):
    #pf = np.flip(pf)
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    colormap = tmap
    if np.min(data) > 0:
        levels = np.linspace(0, maxi, 21)
        colormap = tmap_zero
    if np.min(data) < 0:
        levels = np.linspace(-maxi, maxi, 21)
        colormap = tmap
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(9,6))
    fig.suptitle(title, **title_font, y=1)
    
    ax1.set_ylim(np.max(pf),np.min(pf))
    ax1.contourf(X, Y, data[0] ,cmap=colormap, levels = levels)
    ax1.text(4,1000, plot_labels[0], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax2.set_ylim(np.max(pf),np.min(pf))
    ax2.contourf(X, Y, data[1] ,cmap=colormap, levels = levels)
    ax2.text(4,1000, plot_labels[1], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax3.set_ylim(np.max(pf),np.min(pf))
    ax3.contourf(X, Y, data[2] ,cmap=colormap, levels = levels)
    ax3.text(4,1000, plot_labels[2], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax4.set_ylim(np.max(pf),np.min(pf))
    ax4.contourf(X, Y, data[3] ,cmap=colormap, levels = levels)
    ax4.text(4,1000, plot_labels[3], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)

    ax5.set_ylim(np.max(pf),np.min(pf))
    ax5.contourf(X, Y, data[4] ,cmap=colormap, levels = levels)
    ax5.text(4,1000, plot_labels[4], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    ax6.set_ylim(np.max(pf),np.min(pf))
    xs = ax6.contourf(X, Y, data[5] ,cmap=colormap, levels = levels)
    ax6.text(4,1000, plot_labels[5], c="black",bbox=dict(facecolor='white', edgecolor='none', pad=4.0), size = 8)
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    cbar_ax = fig.add_axes([0.1, 0, 0.82, 0.03])
    plt.colorbar(xs,label="K/day", orientation="horizontal", cax = cbar_ax)   

def TwoByOnePlot(example, predicted, title, y, pf):
    from matplotlib.patches import Ellipse
    X, Y = np.meshgrid(y, pf)
    maxi_d = np.max(np.abs(predicted))
    maxi_e = np.max(np.abs(example))
    levels_d = np.linspace(-maxi_d, maxi_d, 35) #21
    levels_e = np.linspace(0, maxi_e, 5) #21
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))

    maxi_example = np.max(np.abs(example))
    colormap_example = tmap
    if np.min(maxi_example) > 0:
        # mini_example = 0
        colormap_example = tmap_zero
    if np.min(maxi_example) < 0:
        # mini_example = -maxi_example
        colormap_example = tmap  
    #xs1 = ax1.contourf(X, Y, example ,cmap=colormap_example, levels = levels_e)
    ax1.set_ylim(np.max(pf),np.min(pf))
    ax1.set_title("Exmple Gaussian Blob")
    
    xs1 = ax1.contour(X, Y, example, colors='black', levels = levels_e)#, linestyles='dashed')
    plt.clabel(xs1, inline=True,fontsize=10)

    maxi_predicted = np.max(np.abs(predicted))
    colormap_example = tmap
    if np.min(maxi_predicted) > 0:
        # mini_predicted = 0
        colormap_predicted = tmap
    if np.min(maxi_predicted) < 0:
        # mini_predicted = -maxi_predicted
        colormap_predicted = tmap      
    xs2 = ax2.pcolormesh(X, Y, predicted ,cmap=colormap_predicted, vmin = -maxi_predicted, vmax = maxi_predicted)
    ax2.set_ylim(np.max(pf),np.min(pf))
    ax2.set_title("Moving Gaussian Blob")
    
    cbar_ax = fig.add_axes([0.1, 0, 0.37, 0.03])
    plt.colorbar(xs1,label="K/day", orientation="horizontal", cax = cbar_ax)   
    cbar_ax = fig.add_axes([0.55,.0, 0.37, 0.03])
    plt.colorbar(xs2,label="K/day", orientation="horizontal", cax = cbar_ax)   
    
    
    circ1 = Ellipse((20, 800), 10, 175, facecolor='None', edgecolor='black', lw=2)
    ax2.add_patch(circ1)

    xs22 = ax2.contour(X, Y, example, colors='black', levels = levels_e)#, linestyles='dashed')
    plt.clabel(xs22, inline=True,fontsize=10)

    plt.show()
      
tmap_uncertainty_dis = cm.get_cmap(tmap_uncertainty,lut=20)
tmap_dis = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors, N = 15)

def TwoByOnePlot2(example, predicted, uncertainty, title, y, pf):
    # from matplotlib.patches import Ellipse
    X, Y = np.meshgrid(y, pf)
    maxi_p = np.max(np.abs(predicted))
    # levels_p = np.linspace(-maxi_p, maxi_p, 35) #21
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6), dpi=300)

    # xs1 = ax1.contourf(X, Y, predicted ,cmap=tmap, levels = levels_p)
    xs1 = ax1.pcolormesh(X, Y, predicted ,cmap=tmap, vmin = -maxi_p, vmax = maxi_p)
    ax1.set_ylim(np.max(pf),np.min(pf))
    ax1.set_title("Exmple Gaussian Blob")
    
    xs12 = ax1.contour(X, Y, example, colors='grey', levels = [0, .1, .2], linewidths = 3)#, linestyles='dashed')
    plt.clabel(xs12, inline=True,fontsize=10)

    maxi_uncertainty = np.max(np.abs(uncertainty))
    
    xs2 = ax2.pcolormesh(X, Y, uncertainty ,cmap=tmap_uncertainty_dis, vmin = 0, vmax = maxi_uncertainty)
    ax2.set_ylim(np.max(pf),np.min(pf))
    ax2.set_title("Moving Gaussian Blob")
    
    xs13 = ax2.contour(X, Y, example, colors='darkgrey', levels = [0, .1, .2], linewidths = 3)#, linestyles='dashed')
    plt.clabel(xs13, inline=True,fontsize=10)
    
    cbar_ax = fig.add_axes([0.1, 0, 0.37, 0.03])
    plt.colorbar(xs1,label="K/day", orientation="horizontal", cax = cbar_ax)   
    cbar_ax = fig.add_axes([0.55,.0, 0.37, 0.03])
    plt.colorbar(xs2,label="K/day", orientation="horizontal", cax = cbar_ax)   
    

    plt.show()

