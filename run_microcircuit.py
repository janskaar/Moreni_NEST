from network_params import net_dict
from sim_params import sim_dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from network import Network
import numpy as np
import sys, os, time

runner_id = int(sys.argv[1])
n_threads = int(os.environ["SLURM_CPUS_PER_TASK"])

sim_dict["rng_seed"] = runner_id + 1
sim_dict["local_num_threads"] = n_threads
sim_dict["data_path"] = os.path.join(sim_dict["data_path"], "runner_" + str(runner_id))
net = Network(sim_dict, net_dict, None)
t0 = time.time()
net.create()
t1 = time.time()
net.connect()
t2 = time.time()
net.simulate()
t3 = time.time()
lines = ["create,connect,simulate,total\n",
        f"{t1-t0},{t2-t1},{t3-t2},{t3-t0}\n"]

with open(os.path.join(sim_dict["data_path"], "time.csv"), "w") as f:
    f.writelines(lines)

# 
