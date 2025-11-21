import os

import nest
import numpy as np


class Network:
    def __init__(self, sim_dict, net_dict, stim_dict=None):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict

        # data directory
        self.data_path = sim_dict["data_path"]
        if nest.Rank() == 0:
            if os.path.isdir(self.data_path):
                message = "  Directory already existed."
                if self.sim_dict["overwrite_files"]:
                    message += " Old data will be overwritten."
            else:
                os.mkdir(self.data_path)
                message = "  Directory has been created."

        self.pop_names = self.net_dict["populations"]
        # derive parameters based on input dictionaries
        self._derive_parameters()

        # initialize the NEST kernel
        self._setup_nest()

    def create(self):
        """
        Creates all network nodes.
        """
        self._create_populations()
        if len(self.sim_dict["rec_dev"]) > 0:
            self._create_recording_devices()
        self._create_poisson_bg_input()

    def connect(self):
        """
        Connects the network.
        """
        self._connect_populations()

        if len(self.sim_dict["rec_dev"]) > 0:
            self._connect_recording_devices()
        self._connect_poisson_bg_input()

        nest.Prepare()
        nest.Cleanup()

    def simulate(self):
        """
        Simulates the microcircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        t_sim = self.sim_dict["t_sim"]
        if nest.Rank() == 0:
            print("Simulating {} ms.".format(t_sim))

        nest.Simulate(t_sim)

    def _derive_parameters(self):
        """
        Derives and adjusts parameters and stores them as class attributes.
        """
        self.num_pops = len(self.net_dict["populations"])
        N_tot = self.net_dict["N_tot"]
        N_tot_orig = sum(self.net_dict["original_population_sizes"][1:])

        self.population_sizes = [
            int(np.round(s * N_tot / N_tot_orig))
            for s in self.net_dict["original_population_sizes"]
        ]

    def _create_poisson_bg_input(self):
        self.bg_poisson_generators = []
        for rate in self.net_dict["nu_ext"]:
            self.bg_poisson_generators.append(
                nest.Create("poisson_generator", params={"rate": rate})
            )

    def _connect_poisson_bg_input(self):
        for poisson_gen, pop in zip(self.bg_poisson_generators, self.pops):
            nest.Connect(
                poisson_gen,
                pop,
                syn_spec={"weight": self.net_dict["weight_ext"], "receptor_type": 1},
            )

    def _setup_nest(self):
        """
        Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        """
        nest.ResetKernel()

        nest.local_num_threads = self.sim_dict["local_num_threads"]
        nest.resolution = self.sim_dict["sim_resolution"]
        nest.rng_seed = self.sim_dict["rng_seed"]
        nest.overwrite_files = self.sim_dict["overwrite_files"]
        nest.print_time = self.sim_dict["print_time"]

        rng_seed = nest.rng_seed
        vps = nest.total_num_virtual_procs

        if nest.Rank() == 0:
            print("RNG seed: {}".format(rng_seed))
            print("Total number of virtual processes: {}".format(vps))

    def _get_neuron_parameters(self, index):
        params = {
            "E_L": self.net_dict["neuron_params"]["E_L"][index],
            "E_ex": self.net_dict["neuron_params"]["E_ex"],
            "E_in": self.net_dict["neuron_params"]["E_in"][index],
            "V_th": self.net_dict["neuron_params"]["V_th"][index],
            "V_reset": self.net_dict["neuron_params"]["V_reset"][index],
            "C_m": self.net_dict["neuron_params"]["C_m"][index],
            "g_L": self.net_dict["neuron_params"]["g_L"][index],
            "t_ref": self.net_dict["neuron_params"]["t_ref"][index],
            "tau_AMPA": self.net_dict["neuron_params"]["tau_AMPA"],
            "tau_GABA": self.net_dict["neuron_params"]["tau_GABA"],
            "tau_rise_NMDA": self.net_dict["neuron_params"]["tau_rise_NMDA"],
            "tau_decay_NMDA": self.net_dict["neuron_params"]["tau_decay_NMDA"],
            "alpha": self.net_dict["neuron_params"]["alpha"],
            "conc_Mg2": self.net_dict["neuron_params"]["conc_Mg2"],
        }
        return params

    def _create_populations(self):
        if nest.Rank() == 0:
            print("Creating neuronal populations.")

        self.pops = []
        for i in np.arange(self.num_pops):
            neuron_params = self._get_neuron_parameters(i)
            population = nest.Create(
                self.net_dict["neuron_model"],
                self.population_sizes[i],
                params=neuron_params,
            )

            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join(self.data_path, "population_nodeids.dat")
            with open(fn, "w+") as f:
                for pop in self.pops:
                    f.write("{} {}\n".format(pop[0].global_id, pop[-1].global_id))

    def _create_recording_devices(self):
        """
        Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        if nest.Rank() == 0:
            print("Creating recording devices.")

        if "spike_recorder" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("  Creating spike recorders.")
            sd_dicts = [
                {
                    "record_to": "ascii",
                    "label": os.path.join(self.data_path, "spike_recorder" + "_" + pop),
                }
                for pop in self.pop_names
            ]
            self.spike_recorders = nest.Create(
                "spike_recorder", n=self.num_pops, params=sd_dicts
            )

        if "multimeter" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("Creating multimeters.")
            mm_dicts = [
                {
                    "interval": self.sim_dict["rec_mm_int"],
                    "record_to": "memory",
                    "record_from": self.sim_dict["rec_from_mm"],
                    "label": os.path.join(self.data_path, "multimeter" + "_" + pop),
                }
                for pop in self.pop_names
            ]
            self.multimeters = nest.Create(
                "multimeter", n=self.num_pops, params=mm_dicts
            )

    def _get_conn_spec_syn_spec(self, source_index, target_index):
        """
        Creates dictionaries for synapse parameters and connection parameters
        for a given connection between a source population and target population.
        Returns list of parameter dictionaries.
        """
        p = self.net_dict["conn_probs"][source_index][target_index]
        if p == 0:
            return None
        if source_index in self.net_dict["inhibitory_indices"]:
            conn_specs = [{"rule": "pairwise_bernoulli", "p": p}]

            w = (
                self.net_dict["weights"][source_index][target_index]
                * self.net_dict["weight_scale"]
            ) / (self.population_sizes[source_index] * p)

            syn_specs = [
                {"delay": self.net_dict["delay"], "weight": w, "receptor_type": 2}
            ]

        else:
            conn_specs = [
                {
                    "rule": "pairwise_bernoulli",  # AMPA
                    "p": self.net_dict["fraction_AMPA"] * p,
                },
                {
                    "rule": "pairwise_bernoulli",  # NMDA
                    "p": self.net_dict["fraction_NMDA"] * p,
                },
            ]

            w_AMPA = (
                self.net_dict["weights"][source_index][target_index]
                * self.net_dict["weight_scale"]
            ) / (
                self.population_sizes[source_index] * p * self.net_dict["fraction_AMPA"]
            )

            w_NMDA = (
                self.net_dict["weights"][source_index][target_index]
                * self.net_dict["weight_scale"]
            ) / (
                self.population_sizes[source_index] * p * self.net_dict["fraction_NMDA"]
            )

            syn_specs = [
                {"delay": self.net_dict["delay"], "weight": w_AMPA, "receptor_type": 1},
                {"delay": self.net_dict["delay"], "weight": w_NMDA, "receptor_type": 3},
            ]

        return conn_specs, syn_specs

    def _connect_populations(self):
        if nest.Rank() == 0:
            print("Connecting neuronal populations recurrently.")

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                params = self._get_conn_spec_syn_spec(j, i)
                if params is None:
                    continue
                conn_specs = params[0]
                syn_specs = params[1]

                for conn_spec, syn_spec in zip(conn_specs, syn_specs):
                    nest.Connect(
                        source_pop, target_pop, conn_spec=conn_spec, syn_spec=syn_spec
                    )

    def _connect_recording_devices(self):
        """Connects the recording devices to the microcircuit."""
        if nest.Rank == 0:
            print("Connecting recording devices.")

        for i, target_pop in enumerate(self.pops):
            if "spike_recorder" in self.sim_dict["rec_dev"]:
                nest.Connect(target_pop, self.spike_recorders[i])
            if "multimeter" in self.sim_dict["rec_dev"]:
                nest.Connect(
                    self.multimeters[i], target_pop[: min(80, len(target_pop))]
                )
