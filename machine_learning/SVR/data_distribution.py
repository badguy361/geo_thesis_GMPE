import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import pygmt

df = pd.read_csv("../../../TSMIP_FF.csv")
df['lnVs30'] = np.log(df['Vs30'])
df['lnRrup'] = np.log(df['Rrup'])

################### Rrup Mw distribution ###########################

plt.grid(color='gray', linestyle = '--',linewidth=0.5)
plt.scatter(df['Rrup'], df['MW'], marker='o',facecolors='none',edgecolors='b', label= 'Data') #數據點
plt.xscale("log")
plt.xlim(1e-1,1e3)
plt.xlabel('Rrup')
plt.ylabel('Mw')
plt.title('TSMIP Data distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Data distribution(1992-2014).png',dpi=300)
plt.show()

################### Vs30 Number of ground motion ###########################

plt.grid(color='gray', linestyle = '--',linewidth=0.5)
plt.hist(df['Vs30'], bins=20, edgecolor="yellow", color="green")
plt.xlabel('Vs30(m/s)')
plt.ylabel('Number of record')
plt.title('TSMIP Vs30 distribution(1992-2014)')
plt.legend()
plt.savefig(f'TSMIP Vs30 distribution(1992-2014).png',dpi=300)
plt.show()


################### Distribution of the epicenters ###########################
region = [119.30,124,21.5,26.0]
Mw_min_df = df[(df['MW'] < 4)]['MW']
Mw_5_df = df[(df['MW'] >= 4) & (df['MW'] < 5)]['MW']
Mw_6_df = df[(df['MW'] >= 5) & (df['MW'] < 6)]['MW']
Mw_max_df = df[(df['MW'] >= 6)]['MW']

fig = pygmt.Figure()
# Make a global Mollweide map with automatic ticks
fig.basemap(region=region, projection="W15c", frame=True)
# Plot the land as light gray, and the water as sky blue
fig.coast(land="#666666", water="skyblue")
fig.plot(
    x=df['Hyp.Long'], y=df['Hyp.Lat'], size=df['MW']*0.1,
    style="c", color="red", pen="1p,black"
)
fig.show()