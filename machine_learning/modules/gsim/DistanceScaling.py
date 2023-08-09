import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib import recfunctions
from utils.imt import PGA, SA, PGV
from tqdm import tqdm

from phung_2020 import PhungEtAl2020Asc
from chang_2023 import Chang2023
from lin_2009 import Lin2009
from abrahamson_2014 import AbrahamsonEtAl2014
from campbell_bozorgnia_2014 import CampbellBozorgnia2014
from chao_2020 import ChaoEtAl2020Asc

dataLen = 17 # rrup 總點位
df = pd.read_csv('data/DSC_condition.csv')

"""
calculate Chang2023 total station value
"""
station_id_num = 1 # station_id 總量
ch_mean = [[0] * dataLen] * station_id_num
ch_sig = [[0] * dataLen] * station_id_num
ch_tau = [[0] * dataLen] * station_id_num
ch_phi = [[0] * dataLen] * station_id_num
for i in tqdm(range(station_id_num)):
    ctx = df[df['sta_id']==i+1].to_records()
    ctx = recfunctions.drop_fields(
        ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
    ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                    ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                    ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'), 
                    ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
                    ('sta_id', '<i8'),('hypo_depth', '<f8'),('z2pt5', '<f8')])
    ctx = ctx.view(np.recarray)
    imts = [PGA()]
    chang = Chang2023()
    ch_mean[i], ch_sig[i], ch_tau[i], ch_phi[i] = chang.compute(ctx, imts, [ch_mean[i]], [ch_sig[i]], [ch_tau[i]], [ch_phi[i]])
    ch_mean_copy = np.exp(ch_mean[i][0].copy())
    plt.plot(ctx['rrup'], ch_mean_copy)

"""
calculate total station id mean value
"""
# total = np.array([0]*dataLen)
# for j in range(station_id_num):
#     total = total + np.exp(ch_mean[j][0])
# total_station_mean = total / station_id_num
# plt.plot(ctx['rrup'], total_station_mean, 'gray', label="Chang2023 average")

"""
others GMM
"""

ctx = df[df['sta_id']==1].to_records()
ctx = recfunctions.drop_fields(
    ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'), 
                ('ry0', '<f8'), ('width', '<f8'), ('vs30measured', 'bool'),
                ('sta_id', '<i8'),('hypo_depth', '<f8'),('z2pt5', '<f8')])
ctx = ctx.view(np.recarray)
imts = [PGA()]
    
phung = PhungEtAl2020Asc()
ph_mean = [[0] * dataLen]
ph_sig = [[0] * dataLen]
ph_tau = [[0] * dataLen]
ph_phi = [[0] * dataLen]
ph_mean, ph_sig, ph_tau, ph_phi = phung.compute(ctx, imts, ph_mean, ph_sig, ph_tau, ph_phi)
ph_mean = np.exp(ph_mean)

lin = Lin2009()
lin_mean = [[0] * dataLen]
lin_sig = [[0] * dataLen]
lin_tau = [[0] * dataLen]
lin_phi = [[0] * dataLen]
lin_mean, lin_sig = lin.compute(ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
lin_mean = np.exp(lin_mean)

abrahamson = AbrahamsonEtAl2014()
abr_mean = [[0] * dataLen]
abr_sig = [[0] * dataLen]
abr_tau = [[0] * dataLen]
abr_phi = [[0] * dataLen]
abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
abr_mean = np.exp(abr_mean)

campbell = CampbellBozorgnia2014()
cam_mean = [[0] * dataLen]
cam_sig = [[0] * dataLen]
cam_tau = [[0] * dataLen]
cam_phi = [[0] * dataLen]
cam_mean, cam_sig, cam_tau, cam_phi = campbell.compute(ctx, imts, cam_mean, cam_sig, cam_tau, cam_phi)
cam_mean = np.exp(cam_mean)

choa = ChaoEtAl2020Asc()
choa_mean = [[0] * dataLen]
choa_sig = [[0] * dataLen]
choa_tau = [[0] * dataLen]
choa_phi = [[0] * dataLen]
choa_mean, choa_sig, choa_tau, choa_phi = choa.compute(ctx, imts, choa_mean, choa_sig, choa_tau, choa_phi)
choa_mean = np.exp([choa_mean])

plt.grid(which="both",
        axis="both",
        linestyle="-",
        linewidth=0.5,
        alpha=0.5)
# plt.plot(ctx['rrup'], ch_mean[0] + ch_sig[0], 'b--')
# plt.plot(ctx['rrup'], ch_mean[0] - ch_sig[0], 'b--')
plt.plot(ctx['rrup'], ph_mean[0], 'r', label="Phung2020")
# plt.plot(ctx['rrup'], ph_mean[0] + ph_sig[0], 'r--')
# plt.plot(ctx['rrup'], ph_mean[0] - ph_sig[0], 'r--')
plt.plot(ctx['rrup'], lin_mean[0], 'g', label="Lin2009")
# plt.plot(ctx['rrup'], lin_mean[0] + lin_sig[0], 'g--')
# plt.plot(ctx['rrup'], lin_mean[0] - lin_sig[0], 'g--')
plt.plot(ctx['rrup'], abr_mean[0], 'b', label="Abrahamson2014")
# plt.plot(ctx['rrup'], abr_mean[0] + abr_sig[0], 'r--')
# plt.plot(ctx['rrup'], abr_mean[0] - abr_sig[0], 'r--')
plt.plot(ctx['rrup'], cam_mean[0], 'yellow', label="CampbellBozorgnia2014")
# plt.plot(ctx['rrup'], cam_mean[0] + choa_sig[0], 'r--')
# plt.plot(ctx['rrup'], cam_mean[0] - choa_sig[0], 'r--')
plt.plot(ctx['rrup'], choa_mean[0], 'pink', label="ChaoEtAl2020Asc")
# plt.plot(ctx['rrup'], choa_mean[0] + choa_sig[0], 'r--')
# plt.plot(ctx['rrup'], choa_mean[0] - choa_sig[0], 'r--')
plt.xlabel(f'rrup(km)')
plt.ylabel('PGA(g)')
plt.title(f'Distance Scaling')
plt.ylim(10e-5, 10)
# plt.ylim(-6, 2)
plt.xscale("log")
plt.xticks([1, 10, 50, 100, 200, 300], [1, 10, 50, 100, 200, 300])
plt.yscale("log")
plt.legend()
plt.savefig('Distance Scaling compare.png', dpi=300)
plt.show()