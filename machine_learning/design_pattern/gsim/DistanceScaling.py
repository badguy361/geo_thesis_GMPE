import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib import recfunctions
from imt import PGA, SA, PGV

from phung_2020 import PhungEtAl2020Asc
from chang_2023 import Chang2023
from lin_2009 import Lin2009
from abrahamson_2014 import AbrahamsonEtAl2014

df = pd.read_csv('test.csv')
ctx = df.to_records()
ctx = recfunctions.drop_fields(
    ctx, ['index', 'src_id', 'rup_id', 'sids', 'occurrence_rate', 'mean'])
ctx = ctx.astype([('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
                  ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
                  ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'),
                  ('clon', '<f8'), ('clat', '<f8'), ('width', '<f8'), 
                  ('ry0', '<f8'), ('vs30measured', 'bool')])
ctx = ctx.view(np.recarray)
imts = [PGA()]

chang = Chang2023()
ch_mean = [[0] * 13]
ch_sig = [[0] * 13]
ch_tau = [[0] * 13]
ch_phi = [[0] * 13]
ch_mean, ch_sig, ch_tau, ch_phi = chang.compute(ctx, imts, ch_mean, ch_sig, ch_tau, ch_phi)
# ch_mean = np.exp(ch_mean)

phung = PhungEtAl2020Asc()
ph_mean = [[0] * 13]
ph_sig = [[0] * 13]
ph_tau = [[0] * 13]
ph_phi = [[0] * 13]
ph_mean, ph_sig, ph_tau, ph_phi = phung.compute(ctx, imts, ph_mean, ph_sig, ph_tau, ph_phi)
# ph_mean = np.exp(ph_mean)

lin = Lin2009()
lin_mean = [[0] * 13]
lin_sig = [[0] * 13]
lin_tau = [[0] * 13]
lin_phi = [[0] * 13]
lin_mean, lin_sig = lin.compute(ctx, imts, lin_mean, lin_sig, lin_tau, lin_phi)
# lin_mean = np.exp(lin_mean)

abrahamson = AbrahamsonEtAl2014()
abr_mean = [[0] * 13]
abr_sig = [[0] * 13]
abr_tau = [[0] * 13]
abr_phi = [[0] * 13]
abr_mean, abr_sig, abr_tau, abr_phi = abrahamson.compute(ctx, imts, abr_mean, abr_sig, abr_tau, abr_phi)
# abr_mean = np.exp(abr_mean)

plt.grid(linestyle=':')
plt.plot(ctx['rrup'], ch_mean[0], 'b', label="Chang2023")
plt.plot(ctx['rrup'], ch_mean[0] + ch_sig[0], 'b--')
plt.plot(ctx['rrup'], ch_mean[0] - ch_sig[0], 'b--')
plt.plot(ctx['rrup'], ph_mean[0], 'r', label="Phung2020")
plt.plot(ctx['rrup'], ph_mean[0] + ph_sig[0], 'r--')
plt.plot(ctx['rrup'], ph_mean[0] - ph_sig[0], 'r--')
# plt.plot(ctx['rrup'], lin_mean[0], 'g', label="Lin2009")
# plt.plot(ctx['rrup'], lin_mean[0] + lin_sig[0], 'g--')
# plt.plot(ctx['rrup'], lin_mean[0] - lin_sig[0], 'g--')
# plt.plot(ctx['rrup'], abr_mean[0], 'r', label="Abrahamson2014")
# plt.plot(ctx['rrup'], abr_mean[0] + abr_sig[0], 'r--')
# plt.plot(ctx['rrup'], abr_mean[0] - abr_sig[0], 'r--')
plt.xlabel(f'rrup(km)')
plt.ylabel('ln(PGA)(g)')
plt.title(f'Distance Scaling')
# plt.ylim(10e-5, 10)
plt.ylim(-6, 2)
plt.xscale("log")
plt.xticks([1, 10, 50, 100, 200, 300], [1, 10, 50, 100, 200, 300])
# plt.yscale("log")
plt.legend()
plt.savefig('Distance Scaling.png', dpi=300)
plt.show()