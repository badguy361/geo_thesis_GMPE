import unittest
from process_train import dataprocess
import pandas as pd
import numpy as np


model = dataprocess()
class Testprocess_train(unittest.TestCase):
    def test_preprocess(self):
        target = "PGA"
        data = pd.read_csv("../../TSMIP_FF_test.csv")
        data['lnVs30'] = np.log(data['Vs30'])
        data['lnRrup'] = np.log(data['Rrup'])
        data['log10Vs30'] = np.log10(data['Vs30'])
        data['log10Rrup'] = np.log10(data['Rrup'])
        data['fault.type'] = data['fault.type'].str.replace("RO", "1")
        data['fault.type'] = data['fault.type'].str.replace("RV", "1")
        data['fault.type'] = data['fault.type'].str.replace("NM", "2")
        data['fault.type'] = data['fault.type'].str.replace("NO", "2")
        data['fault.type'] = data['fault.type'].str.replace("SS", "3")
        data['fault.type'] = pd.to_numeric(data['fault.type'])
        data[f'ln{target}(gal)'] = np.log(data[f'{target}'] * 980)
        after_process_data = data[(data["eq.type"] != "subduction interface")
                                  & (data["eq.type"] != "subduction intraslab")
                                  & (data["eq.type"] != "deep crustal")]
        df = pd.read_csv("../../TSMIP_FF_test.csv")
        result = model.preprocess(df, target, True)
        assert after_process_data.equals(result)

suite = unittest.TestSuite()
suite.addTest(Testprocess_train('test_preprocess'))

unittest.TextTestRunner(verbosity=2).run(suite)