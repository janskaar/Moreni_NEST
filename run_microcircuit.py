from network_params import net_dict
from sim_params import sim_dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from network import Network
import numpy as np

population_rates = []

for i in range(5):
    sim_dict["rng_seed"] = i + 1
    net = Network(sim_dict, net_dict, None)
    net.create()
    net.connect()
    net.simulate(1000.0)
    bins = np.arange(0, 501, 1) - 0.001
    spike_histograms = [
        np.histogram(sr.events["times"], bins=bins)[0] for sr in net.spike_recorders
    ]
    population_rates.append(
        [
            np.mean(hist / net.population_sizes[i] * 1000)
            for i, hist in enumerate(spike_histograms)
        ]
    )
population_rates = np.array(population_rates)
