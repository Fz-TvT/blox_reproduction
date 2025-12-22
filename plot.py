import os
import sys
import json
import grpc
import argparse
import numpy as np
from workload import Workload
from concurrent import futures
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from collections import defaultdict
def _get_avg_jct(time_dict):
    """
    Fetch the avg jct from the dict
    """
    values = list(time_dict.values())
    count = 0
    jct_time = 0
    for v in values:
        jct_time += v[1] - v[0]
        count += 1

    return jct_time / count
# something in pylab * screws up with random library, for now just overwriting
import random
if __name__=="__main__":
        """
        Post Simulation Plot Graphs
        """
        # print("plot called")
        jct_dict = defaultdict(dict)
        file_names_job_stats = list()
        file_names_cluster_stats = list()
        scheduler="Fifo"
        print("Generating plots for scheduler {}".format(scheduler))
        for load in np.arange(1.0, 3.0, 1.0):
            stat_fname = f"test_3000_4000_{scheduler}_AcceptAll_load_{load}_job_stats.json"
            print(stat_fname)
            with open(stat_fname, "r") as fin:
                data_job = json.load(fin)
            print(scheduler, load)
            jct_dict[scheduler][load] = _get_avg_jct(data_job)
            # print("Wrote to dict")
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 3)
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        print("Job completion dict {}".format(jct_dict))
        plot_list = list()
        # x_labels = list()
        for load in np.arange(1.0, 3.0, 1.0):
            plot_list.append(jct_dict[scheduler][load])
        ax1.plot(np.arange(1.0, 3.0, 1.0), plot_list, label=scheduler)
        write_folder = f"./plots/test_3000_4000_{scheduler}_AcceptAll_load_{load}"
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
        ax1.set_title(f"{scheduler}_load_{load}_Average_JCT")
        plt.savefig(
                    os.path.join(write_folder, f"{scheduler}_load_{load}_jct.pdf"),
                    format="pdf",
                    dpi=600,
                    bbox_inches="tight",
                )

        # plot cdf

        for load in np.arange(1.0, 3.0, 1.0):
            stat_fname = f"test_3000_4000_{scheduler}_AcceptAll_load_{load}_job_stats.json"
            with open(stat_fname, "r") as fin:
                data_job = json.load(fin)
            vals = data_job.values()
            vals = [val[1] - val[0] for val in vals]
            vals = sorted(vals)
            plot_y_val = list()
            for idx, val in enumerate(vals):
                plot_y_val.append(float(idx) / len(vals))
            print("val",len(vals))
            write_folder = f"./plots/test_3000_4000_{scheduler}_AcceptAll_load_{load}"            
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(10, 3)
            ax1.set_xscale("log")
            ax1.set_title(f"{scheduler}_load_{load}_cdf")
            ax1.plot(vals, plot_y_val)
            plt.savefig(
                os.path.join(write_folder, f"{scheduler}_load_{load}_cdf.pdf"),
                format="pdf",
                dpi=600,
                bbox_inches="tight",
            )

        # plot GPU demand
        for load in np.arange(1.0, 3.0, 1.0):
            stat_fname = f"test_3000_4000_{scheduler}_AcceptAll_load_{load}_cluster_stats.json"
            print(stat_fname)
            with open(stat_fname, "r") as fin:
                data_job = json.load(fin)
            gpu_demand = list()
            free_gpus = list()
            print(data_job)
            for d in data_job.keys():
                gpu_demand.append(data_job[d]["gpu_demand"])
                free_gpus.append(data_job[d]["free_gpus"])
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(10, 3)
            ax1.set_title(f"{scheduler}_load_{load}_gpu_demand")
            ax1.set_xscale("log")
            ax1.plot(data_job.keys(), gpu_demand)

            write_folder = f"./plots/test_3000_4000_{scheduler}_AcceptAll_load_{load}"            
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            plt.savefig(
                os.path.join(
                    write_folder, f"{scheduler}_load_{load}_gpu_demand.pdf"
                ),
                format="pdf",
                dpi=600,
                bbox_inches="tight",
            )
            # plot free GPUs
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(10, 3)
            ax1.set_title(f"{scheduler}_load_{load}_free_GPUs")
            print("data_key",data_job.keys())
            ax1.plot(data_job.keys(), free_gpus)
            plt.savefig(
                os.path.join(write_folder, f"{scheduler}_load_{load}_free_gpu.pdf"),
                format="pdf",
                dpi=600,
                bbox_inches="tight",
            )