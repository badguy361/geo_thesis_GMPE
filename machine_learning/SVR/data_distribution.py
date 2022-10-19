import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import pygmt

df = pd.read_csv("../../../TSMIP_FF.csv")
df['lnVs30'] = np.log(df['Vs30'])
df['lnRrup'] = np.log(df['Rrup'])
df['Mw_size'] = np.zeros(len(df['lnRrup']))
df['lnPGA'] = np.log(df['PGA'] * 980)

################### PGA Vs30 distribution ###########################

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(df['lnVs30'],
            df['lnPGA'],
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
# plt.xscale("log")
plt.yscale("log")
# plt.xlim(3, 8)
# plt.ylim(1e-1, 1e3)
plt.xlabel('lnVs30')
plt.ylabel('lnPGA')
plt.title('TSMIP Vs30 PGA distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Vs30 PGA distribution(1992-2014).png', dpi=300)
plt.show()

################### PGA Mw distribution ###########################

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(df['MW'],
            df['lnPGA'],
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
# plt.xscale("log")
plt.yscale("log")
plt.xlim(3, 8)
# plt.ylim(1e-1, 1e3)
plt.xlabel('Mw')
plt.ylabel('lnPGA')
plt.title('TSMIP Mw PGA distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Mw PGA distribution(1992-2014).png', dpi=300)
plt.show()


################### PGA Rrup distribution ###########################

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(df['lnRrup'],
            df['lnPGA'],
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
# plt.xscale("log")
plt.yscale("log")
# plt.xlim(1e-1, 1e3)
# plt.ylim(1e-1, 1e3)
plt.xlabel('lnRrup')
plt.ylabel('lnPGA')
plt.title('TSMIP Rrup PGA distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Rrup PGA distribution(1992-2014).png', dpi=300)
plt.show()

################### Rrup Mw distribution ###########################

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(df['Rrup'],
            df['MW'],
            marker='o',
            facecolors='none',
            edgecolors='b',
            label='Data')  #數據點
plt.xscale("log")
plt.xlim(1e-1, 1e3)
plt.xlabel('Rrup')
plt.ylabel('Mw')
plt.title('TSMIP Data distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Data distribution(1992-2014).png', dpi=300)
plt.show()

################### Vs30 Number of ground motion ###########################

plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.hist(df['Vs30'], bins=20, edgecolor="yellow", color="green")
plt.xlabel('Vs30(m/s)')
plt.ylabel('Number of record')
plt.title('TSMIP Vs30 distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Vs30 distribution(1992-2014).png', dpi=300)
plt.show()

################### Distribution of the epicenters ###########################
region = [119.30, 124, 21.5, 26.0]
df.loc[(df['MW'] < 4), 'Mw_size'] = 0.1
df.loc[(df['MW'] >= 4) & (df['MW'] < 5), 'Mw_size'] = 0.3
df.loc[(df['MW'] >= 5) & (df['MW'] < 6), 'Mw_size'] = 0.5
df.loc[(df['MW'] >= 6), 'Mw_size'] = 0.7

fig = pygmt.Figure()
# Make a global Mollweide map with automatic ticks
fig.basemap(region=region,
            projection="W15c",
            frame=["af", f'WSne+t"TSMIP epicneters distribution"'])
# Plot the land as light gray, and the water as sky blue
fig.coast(land="#666666", water="skyblue")
fig.plot(x=df['Hyp.Long'],
         y=df['Hyp.Lat'],
         size=df['Mw_size'],
         style="c",
         color="red",
         pen="1p,black")

fig.text(x=122.8, y=25.8, text="Mw")
fig.plot(x=123, y=25.8, color="gray", style="c0.1c", pen="1p,black")
fig.plot(x=123.2, y=25.8, color="gray", style="c0.3c", pen="1p,black")
fig.plot(x=123.4, y=25.8, color="gray", style="c0.5c", pen="1p,black")
fig.plot(x=123.65, y=25.8, color="gray", style="c0.7c", pen="1p,black")

fig.text(x=123, y=25.6, text="<4", font="12p")
fig.text(x=123.2, y=25.6, text="5", font="12p")
fig.text(x=123.4, y=25.6, text="6", font="12p")
fig.text(x=123.65, y=25.6, text=">7", font="12p")

fig.savefig("TSMIP epicneters distribution.png", dpi=300)
fig.show()