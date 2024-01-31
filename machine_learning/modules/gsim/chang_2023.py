# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2023 GEM Foundation, Chih-Yu Chang
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import math
import numpy as np
import utils.const as const
from utils.coeffs_table import CoeffsTable
from utils.imt import PGA, SA, PGV
from numpy.lib import recfunctions
import queue
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm


class Chang2023():
    #: Supported tectonic region type is active shallow crust, see title!
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration, peak
    #: ground velocity and peak ground acceleration, see tables 4
    #: pages 1036
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, PGV, SA}

    #: Supported intensity measure component is orientation-independent
    #: average horizontal :attr:`~openquake.hazardlib.const.IMC.RotD50`,
    #: see page 1025.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see paragraph "Equations for standard deviations", page
    #: 1046.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
        const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    #: Required site parameters are Vs30 and Z1.0, see table 2, page 1031
    #: Unit of measure for Z1.0 is [m]
    REQUIRES_SITES_PARAMETERS = {'vs30'}

    #: Required rupture parameters are magnitude, rake, dip, ztor, and width
    #: (see table 2, page 1031)
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake'}

    #: Required distance measures are Rrup, Rjb, Ry0 and Rx (see Table 2,
    #: page 1031).
    REQUIRES_DISTANCES = {'rrup'}

    #: Reference rock conditions as defined at page
    DEFINED_FOR_REFERENCE_VELOCITY = 1180

    def __init__(self, model_path):
        ML_model = xgb.Booster()
        ML_model.load_model(model_path)
        self.ML_model = ML_model

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        for m, imt in enumerate(imts):
            predict = self.ML_model.predict(xgb.DMatrix(np.column_stack(
                (np.log(ctx.vs30), ctx.mag, np.log(ctx.rrup), ctx.rake, ctx.sta_id))))
            mean[m] = np.log(np.exp(predict)/980)  # unit : ln(g)
            sig[m], tau[m], phi[m] = 0.35, 0.12, 0.34

        return mean, sig, tau, phi


if __name__ == '__main__':
    dtype = [('dip', '<f8'), ('mag', '<f8'), ('rake', '<f8'),
             ('ztor', '<f8'), ('vs30', '<f8'), ('z1pt0', '<f8'),
             ('rjb', '<f8'), ('rrup', '<f8'), ('rx', '<f8'), ('sta_id', '<i8')]
    rrup_num = [0.1, 0.5, 1, 10, 50, 100, 200, 300]
    station_num = 400
    total_elements = len(rrup_num) * station_num
    ctx = np.empty(total_elements, dtype=dtype)
    index = 0
    for station_id in range(station_num): # 依照station_num、Rrup的順序建recarray
        for rrup in rrup_num:
            ctx[index] = (0, 7.65, 90, 1, 360, 1, 1, rrup, 1, station_id + 1)
            index += 1
    ctx = ctx.view(np.recarray)

    imts = [PGA()]
    mean = [[0]*len(imts)]
    sig = [[0]*len(imts)]
    tau = [[0]*len(imts)]
    phi = [[0]*len(imts)]
    gmm = Chang2023(f"XGB_PGA.json")
    mean, sig, tau, phi = gmm.compute(ctx, imts, mean, sig, tau, phi)
    mean = np.exp(mean)
    split_array = np.split(mean[0], station_num) # 預測結果依station_num總數拆成n組

    for i in range(station_num):
        plt.plot(rrup_num, split_array[i])
    plt.grid(which="both",
                axis="both",
                linestyle="-",
                linewidth=0.5,
                alpha=0.5)
    plt.xlabel(f'Rrup(km)')
    plt.ylabel('PGA(g)')
    plt.title(f'Distance Scaling Chang2023')
    plt.ylim(10e-4, 5)
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks([0.1, 0.5, 1, 10, 50, 100, 200, 300],
                [0.1, 0.5, 1, 10, 50, 100, 200, 300])
    plt.legend()
    plt.savefig('Distance Scaling Chang2023.png',dpi=300)
    plt.show()
