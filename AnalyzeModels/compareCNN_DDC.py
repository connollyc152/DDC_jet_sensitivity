import numpy as np
import matplotlib.pyplot as plt

text_font = {'fontname':'Noto Sans JP', 'size':'7', 'color':'black', 'weight':'bold'}
label_font = {'fontname':'Noto Sans JP', 'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
label_font2 = {'fontname':'Noto Sans JP', 'size':'9', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
title_font = {'fontname':'Noto Sans JP', 'size':'18', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
axis_font = {'fontname':'Noto Sans JP', 'size':'13', 'color':'black', 'weight':'normal'}
colorbar_font = {'family':'Noto Sans JP', 'size':'10', 'color':'black', 'weight':'light'}
legend_font = {'family':'Assistant', 'size':'8'}
axisTick_font = {'family':'Noto Sans JP', 'size':'10', 'color':'black', 'weight':'light'}

DDC_delta = np.array([2.9293737, 2.3378592, 2.2779036, 0.79852587, 0.972621, 0.410903,
              1.3341055,-0.48264405,-1.9080896,-0.16747375,-2.5479403,-3.622861,
              -3.2531443,-3.5393648])
CNN_delta = np.array([2.1370685, 2.1559293, 1.9132582, 1.4985067, 1.2569493, 0.6527403,
              0.60248685, -0.12237211, -0.31416297, -0.9085139, -1.0140995,
              -1.3726072, -1.7032474, -1.8255893])
CNN_uncertainty = np.array([0.39380658, 0.4629696, 0.6510144, 0.9510937, 1.1769874,
                    0.45130357, 0.54063165, 1.139453, 1.3951901, 0.5303465, 
                    0.6448716, 0.9379413, 1.3494642, 1.624238])

tropical_warming = [.1,   .1,  .1,  .1,  .1, 0, 0,   0, 0, -.1, -.1, -.1, -.1, -.1]
polar_warming =    [ -1, -.5,   0,  .5,   1, -1, -.5, .5, 1, -1, -.5, 0, .5, 1]
heating_exp =    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

tropical_warming_color = [-1,   -1,  -1,  -1,  -1, 0, 0,   0, 0, 1, 1, 1, 1, 1]
polar_warming_color =    [ 1, 1,   0,  -1,   -1, 1, 1, -1, -1, 1, 1, 0, -1, -1]
heating_exp_color  =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
LABEL = True

x_plot = np.arange(0, np.size(DDC_delta), 1)
n = len(polar_warming)
########################################

fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 8), dpi=600,  gridspec_kw={'width_ratios': [1, 4]})
ax2.axvline(0, color = "grey", zorder = 3, linewidth = .5)

bars = plt.barh(x_plot, CNN_delta, color='none', zorder=2, label = "CNN prediction")#, width = .5)
plt.errorbar(CNN_delta, x_plot, xerr = (CNN_uncertainty * 2), fmt='none', elinewidth=20, ecolor='silver', capsize=0, zorder=1)#, label = "CNN uncertainty")

for bar in bars:
    x, y = bar.get_xy()
    w, h = bar.get_width(), bar.get_height()
    plt.plot([x, x], [y, y + h], color='black', lw=0)
    plt.plot([x, x + w], [y + h, y + h], color='black', lw=0)#
    plt.plot([x + w, x + w], [y, y + h], color='black', lw=1)
plt.margins(x=0.02)

plt.scatter(DDC_delta, x_plot, marker = 'o', c='seagreen', zorder=3, label="dry core output")

plt.xlabel("Jet Shift (degrees poleward)", **label_font, labelpad = -460)
ax2.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.set_ylim(-.5,13.5)
ax2.set_xlim(-4.5,4.5)
plt.title("", **title_font)
plt.legend(loc='lower left', prop={'family':'Assistant', 'size':'8'}, frameon=False)


x1, y1 = [-4.2, -3.6], [0.-.15,0.-.15]
ax2.plot(x1, y1, c = "black")
x2, y2 = [-4.1, -3.76], [.60,.60]
ax2.plot(x2, y2, c = "lightgrey", linewidth = 10, zorder = 4)
ax2.text(-3.4, .56, "CNN Uncertainty", **legend_font)vb 

table_array = np.empty(shape = (np.size(polar_warming),3))
table_array[:,0] = (heating_exp)
table_array[:,1] = np.flip(polar_warming)
table_array[:,2] = np.flip(tropical_warming)


table_array2 = np.empty(shape = (np.size(polar_warming),3))
table_array2[:,0] = (heating_exp_color)
table_array2[:,1] = np.flip(polar_warming_color)
table_array2[:,2] = np.flip(tropical_warming_color)

normal = plt.Normalize(table_array2.min()-1, table_array2.max()+1)
colours = plt.cm.RdBu(normal(table_array2))

table_array = np.array(table_array, dtype=object)
table_array[:,0] = heating_exp

the_table = ax2.table(cellText=table_array,
                      colLabels=["Heating \nExperiment", "Polar \nMagnitude \n(K/day)", "Tropical \nMagnitude \n(K/day)"],
                       cellLoc='center',
                       bbox=(-0.9, 0.0, 0.9, ((n+1) / n) + .03), 
                       cellColours=colours)
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
cellDict = the_table.get_celld()
cellDict[(0,0)].set_height(.039)
cellDict[(0,1)].set_height(.039)
cellDict[(0,2)].set_height(.039)

ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

plt.show()

############################################################
#############Figure in Supplementary########################
############################################################
DDC_delta = [4.16, 1.12, -5.87, -.56]
CNN_delta = [5.07, -1.03, -3.15, 1.04]
CNN_uncertainty = [.77, .92, 1.76, .67]

x_plot = np.arange(0, np.size(DDC_delta), 1)

fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(8, 3), dpi=600)#,  gridspec_kw={'width_ratios': [1]})
ax2.axvline(0, color = "grey", zorder = 3, linewidth = .5)
bars = plt.barh(x_plot, CNN_delta, color='none', zorder=2, label = "CNN prediction")#, width = .5)
plt.errorbar(CNN_delta, x_plot, xerr = (np.array(CNN_uncertainty) * 2), fmt='none', elinewidth=20, ecolor='silver', capsize=0, zorder=1)#, label = "CNN uncertainty")

for bar in bars:
    x, y = bar.get_xy()
    w, h = bar.get_width(), bar.get_height()
    plt.plot([x, x], [y, y + h], color='black', lw=0)
    plt.plot([x, x + w], [y + h, y + h], color='black', lw=0)#
    plt.plot([x + w, x + w], [y, y + h], color='black', lw=1)
plt.margins(x=0.02)

plt.scatter(DDC_delta, x_plot, marker = 'o', c='seagreen', zorder=3, label="dry core output")

ax2.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
ax2.set_yticks([])
ax2.set_yticklabels([])
ax2.set_ylim(-.5,3.5)
plt.title("", **title_font)

plt.show()

