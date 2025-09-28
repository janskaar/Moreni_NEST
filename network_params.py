import numpy as np

net_dict = {
    # neuron model
    "neuron_model": "iaf_bw_2001",
    # names of the simulated neuronal populations
    "populations": ("L1VIP",
                    "L23E", "L23PV", "L23SST", "L23VIP",
                    "L4E", "L4PV", "L4SST", "L4VIP",
                    "L5E", "L5PV", "L5SST", "L5VIP",
                    "L6E", "L6PV", "L6SST", "L6VIP"),
    "inhibitory_indices": (0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16),

    "N_tot": 5000,

    # Data from L23-L6 comes from V1_structure.xlsx from Billeh.
    # L1 size is set such that population size matches Moreni paper.
    "original_population_sizes": [4350,
                                  56057, 2927, 2120, 4845,
                                  45761, 4461, 2384, 1231,
                                  33558, 2876, 2538,  508,
                                  57202, 4626, 4626,  841],


    # taken from connectionsPro_final.txt, Cpl1_final.txt, Cptol1_final.txt. L1->L1 found in simulation scripts.
    "conn_probs": np.loadtxt("connectivity.csv", delimiter=","),
    # taken from connectionsStren.txt, Csl1.txt, Cstol1.txt. L1->L1 found in simulation scripts.
    "weights": np.loadtxt("synaptic_weights.csv", delimiter=","),
    "weight_scale": 5.,
    "nu_ext": [650, 930, 1460, 870, 1405, 890, 1980, 2105, 240, 4740, 930, 530, 870, 1770, 1170, 885, 1620],
#     "I_ext": [
    "weight_ext": 1.,
    "delay": 0.5,
    "fraction_AMPA": 0.8,
    "fraction_NMDA": 0.2,
    "neuron_params": {
        "E_L": [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72, -77.5, -76.42, -62.99, -78.85],
        "V_th": [-40.20, -40.53, -56.32, -39.95, -41.34, -47.63, -44.23, -44.07, -40.89, -40.55, -51.2, -47.38, -51.2, -42.31, -49.06, -37.19, -44.81],
        "V_reset": [-65.5, -80.97, -82.35, -69.16, -67.94, -72.53, -70.45, -74.2, -63.14, -68.28, -77.5, -70.01, -72, -77.5, -76.42, -62.99, -78.85],
        "C_m": [37.11, 123.41, 70.95, 82.34, 41.23, 80.16, 81.21, 132.86, 40.3, 149.43, 70.9, 52.32, 59.29, 99.96, 49.65, 96.09, 65.87],
        "t_ref": [3.5, 3, 1.26, 1.85, 2.75, 4.4, 1.5, 2.2, 2.4, 4.25, 1.85, 1.9, 2.55, 3.3, 1.65, 2.1, 2.85],
        "g_L": [4.07, 2.47, 9.49, 3.17, 6.4, 5.16, 9.19, 7.96, 1.87, 16.66, 5.21, 3.43, 6.52, 5.88, 6.86, 2.99, 6.09],
        "tau_AMPA": 2.,
        "tau_GABA": 5.,
        "tau_rise_NMDA": 2.,
        "tau_decay_NMDA": 80.,
        "E_ex": 0.,
        "alpha": 0.5,
        "conc_Mg2": 1.,
    },
}

net_dict["neuron_params"].update(E_in = net_dict["neuron_params"]["E_L"])

