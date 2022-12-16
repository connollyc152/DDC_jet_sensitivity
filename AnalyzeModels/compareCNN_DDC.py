import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import netCDF4 as nc
import matplotlib.colors as colors

import sys
sys.path.insert(0, '/Users/cconn/Documents/CNN_proj/AnalyzeModels')

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

f_DDC = nc.Dataset("/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/Averaged_HeatingFiles/DeltaFromHeating_small.nc",'r')
DDC_delta = f_DDC['heatingScenerios'][:,0]
f_CNN = nc.Dataset("/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data/Averaged_HeatingFiles/DeltaFromCNN_small_init0_nolag.nc",'r')
CNN_delta = f_CNN['heatingScenerios'][:,0]
CNN_uncertainty = f_CNN['heatingScenerios'][:,3]
MAE = np.abs(DDC_delta-CNN_delta)

tropical_warming = [.1,   .1,  .1,  .1,  .1, 0, 0,   0, 0, -.1, -.1, -.1, -.1, -.1]
polar_warming =    [ -1, -.5,   0,  .5,   1, -1, -.5, .5, 1, -1, -.5, 0, .5, 1]
heating_exp =    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

tropical_warming_color = [-1,   -1,  -1,  -1,  -1, 0, 0,   0, 0, 1, 1, 1, 1, 1]
polar_warming_color =    [ 1, 1,   0,  -1,   -1, 1, 1, -1, -1, 1, 1, 0, -1, -1]
heating_exp_color  =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
LABEL = True

x_plot = np.arange(0, np.size(DDC_delta), 1)
n = len(polar_warming)
# print(n, np.shape(DDC_delta), np.shape(x_plot), np.shape(CNN_delta))
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
ax2.text(-3.4, .56, "CNN Uncertainty", **legend_font)



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
# DDC_delta = [4.16, 1.12, -5.87, -.56]
# CNN_delta = [5.07, -1.03, -3.15, 1.04]
# CNN_uncertainty = [.77, .92, 1.76, .67]


# ####.5
# # DDC_delta = [7.14, 1.61, -12.30, -2.83]
# # CNN_delta = [7.23, -2.28, -5.21, 1.51]
# # CNN_uncertainty = [.85, 1.01, .81, 0.61]

# x_plot = np.arange(0, np.size(DDC_delta), 1)

# fig, (ax2) = plt.subplots(nrows=1, ncols=1, figsize=(8, 3), dpi=600)#,  gridspec_kw={'width_ratios': [1]})
# ax2.axvline(0, color = "grey", zorder = 3, linewidth = .5)
# # fig = plt.figure()
# # ax = plt.subplot(121)
# # ax2 = plt.subplot(122)
# # plt.axhline(y=0, color='black', linestyle='-', linewidth = .5, zorder = 0)
# # plt.errorbar(x_plot, CNN_delta, yerr = (CNN_uncertainty * 2), fmt='none', elinewidth=20, ecolor='silver', capsize=0, zorder=1, label = "CNN uncertainty")
# bars = plt.barh(x_plot, CNN_delta, color='none', zorder=2, label = "CNN prediction")#, width = .5)
# plt.errorbar(CNN_delta, x_plot, xerr = (np.array(CNN_uncertainty) * 2), fmt='none', elinewidth=20, ecolor='silver', capsize=0, zorder=1)#, label = "CNN uncertainty")

# for bar in bars:
#     x, y = bar.get_xy()
#     w, h = bar.get_width(), bar.get_height()
#     plt.plot([x, x], [y, y + h], color='black', lw=0)
#     plt.plot([x, x + w], [y + h, y + h], color='black', lw=0)#
#     plt.plot([x + w, x + w], [y, y + h], color='black', lw=1)
# plt.margins(x=0.02)

# plt.scatter(DDC_delta, x_plot, marker = 'o', c='seagreen', zorder=3, label="dry core output")


# # plt.title("Heating Experiements", **title_font, labelpad=15)
# # plt.xlabel("Jet Shift (degrees poleward)", **label_font, labelpad = -460)
# ax2.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
# ax2.set_yticks([])
# ax2.set_yticklabels([])
# ax2.set_ylim(-.5,3.5)
# # ax2.set_xlim(-4.5,4.5)
# plt.title("", **title_font)
# # plt.legend(loc='lower left', prop={'family':'Assistant', 'size':'8'}, frameon=False)

# # x1, y1 = [-4.2, -3.6], [0.-.15,0.-.15]
# # ax2.plot(x1, y1, c = "black")
# # x2, y2 = [-4.1, -3.76], [.60,.60]
# # ax2.plot(x2, y2, c = "lightgrey", linewidth = 10, zorder = 4)
# # ax2.text(-3.4, .56, "CNN Uncertainty", **legend_font)


# plt.show()

