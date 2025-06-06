import sys
sys.path.append("..")
from modules.enhancement_plot import SeismicDataPlotter
import pandas as pd

seismic_data_plotter = SeismicDataPlotter(["x_train", "x_test", "y_train", "y_test"])
# seismic_data_plotter.plot_mw_distribution()
# seismic_data_plotter.plot_fault_type_distribution()
# seismic_data_plotter.plot_ln_pga_distribution()
# eq_distribution_file = pd.read_csv("../../../TSMIP_FF_eq_distribution.csv")
# seismic_data_plotter.plot_event_distribution_map(eq_distribution_file)

# total_data = pd.read_csv("../../../TSMIP_FF.csv")
# seismic_data_plotter.get_data_less_than_10km(total_data)
# data_less_than_10km = pd.read_csv("../../../data_less_than_10km.csv")
# seismic_data_plotter.predict_less_than_10km_data(data_less_than_10km)
data_less_than_10km_predicted = pd.read_csv("../../../data_less_than_10km_predicted.csv")
seismic_data_plotter.calculate_and_plot_residual_std(data_less_than_10km_predicted)