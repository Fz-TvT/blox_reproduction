import os
import sys
import json
import grpc
import argparse
import numpy as np
from workload import Workload
from concurrent import futures
from typing import Tuple
from workload.utils import get_job_gpu_demand
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from collections import defaultdict
import heapq
# something in pylab * screws up with random library, for now just overwriting
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "./deployment/grpc_stubs"))
from blox.deployment.grpc_stubs import rm_pb2
from blox.deployment.grpc_stubs import rm_pb2_grpc
from blox.deployment.grpc_stubs import simulator_pb2
from blox.deployment.grpc_stubs import simulator_pb2_grpc
import traceback


class SimulatorRunner(simulator_pb2_grpc.SimServerServicer):
    """
    Run jobs in the simulation mode
    """

    def __init__(
        self,
        cluster_job_log,
        list_jobs_per_hour,
        job_ids_to_track,
        schedulers,
        placement_policies,
        acceptance_policies,
        model_class_split=(20, 70, 10),
        ipaddr_resource_manager="localhost",
        exponential=True,
        multigpu=True,
        small_trace=False,
        placement=True,
        prioritize=False,
        round_duration=6000,
        number_of_machines=16,
        gpus_per_machine=8,
        memory_per_machine=500,
        is_numa_available=False,
        num_cpu_cores=24,
        num_jobs_default=0,
        exp_prefix="test",
    ):
        # self.cluster_job_log = cluster_job_log
        self.cluster_job_log = cluster_job_log
        self.list_jobs_per_hour = list_jobs_per_hour
        self.job_ids_to_track = job_ids_to_track
        self.schedulers = schedulers
        self.placement_policies = placement_policies
        self.acceptance_policies = acceptance_policies

        self.model_class_split = model_class_split
        self.exponential = exponential
        self.multigpu = multigpu
        self.small_trace = small_trace
        self.placement = placement
        self.prioritize = prioritize
        self.round_duration = round_duration
        self.num_jobs_default = num_jobs_default

        # cluster parameters
        self.number_of_machines = number_of_machines
        self.gpus_per_machine = gpus_per_machine
        self.memory_per_machine = memory_per_machine
        self.is_numa_available = is_numa_available
        self.num_cpu_cores = num_cpu_cores
        self.ipaddr_rm = f"{ipaddr_resource_manager}:50051"

        # self.jobs_to_run = Workload(cluster_job_log)
        # setup the training
        self.simulator_config = list()
        self._generate_simulator_configs()
        self.prev_job_time = 0
        self.latest_job = None
        self.prev_job = None
        # first_job_config = self.simulator_config.pop(0)
        # self.workload = self._generate_workload(first_job_config)

        self.random_seed = 1
        self.exp_prefix = exp_prefix

        return None

    def GetConfig(self, request, context):
        """
        Provide new_jobt config to run the simulator.
        """
        # get new job config
        try:
            job_config = self.simulator_config.pop(0)
            # setup new workload
            self.workload = self._generate_workload(job_config)
            job_config_send = rm_pb2.JsonResponse()
            job_config_send.response = json.dumps(job_config)
            self.setup_cluster()
            # reseting the job time
            self.prev_job_time = 0
            self.prev_job = None
            print("Job config {}".format(job_config))
            return job_config_send
        except IndexError:
            # list empty signal to terminate
            job_config = dict()
            job_config["scheduler"] = ""
            job_config["load"] = -1
            job_config["start_id_track"] = 0
            job_config["stop_id_track"] = 0
            job_config_send = rm_pb2.JsonResponse()
            job_config_send.response = json.dumps(job_config)
            # self.setup_cluster()
            # self._plot_graphs()
            return job_config_send

    def GetJobs(self, request, context) -> rm_pb2.JsonResponse:
        """
        Return a dictionary of jobs for simulating.
        """
        simulator_time = request.value
        job_to_run_dict = dict()
        jcounter = 0
        print("Simulator time {}".format(simulator_time))
        new_job = None
        while True:
            try:
                if self.prev_job is None:
                    new_job = self.workload.generate_next_job(self.prev_job_time) ##下一个job
                    new_job_dict = self._clean_sim_job(new_job.__dict__)
                if self.prev_job is not None:
                    print("Self previous job")
                    new_job_dict = self.prev_job
                print(
                    "New job dict arrival time {}".format(
                        new_job_dict["job_arrival_time"]
                    )
                )
                if new_job_dict["job_arrival_time"] <= simulator_time:
                    print("In getting more jobs")
                    job_to_run_dict[jcounter] = new_job_dict
                    self.prev_job_time = new_job_dict["job_arrival_time"]
                    self.prev_job = None
                    jcounter += 1
                if new_job_dict["job_arrival_time"] > simulator_time:
                    # no more jobs for next time
                    print("returning previos job")
                    valid_jobs = rm_pb2.JsonResponse()
                    valid_jobs.response = json.dumps(job_to_run_dict)
                    self.prev_job = new_job_dict
                    self.prev_job_time = new_job_dict["job_arrival_time"]
                    print("Json dump and return")
                    return valid_jobs
            ##modified part        
            except StopIteration: 
                # 不再生成新作业，但如果有缓存作业（self.prev_job）且时间到了，仍可返回
                if self.prev_job and self.prev_job["job_arrival_time"] <= simulator_time:
                    job_to_run_dict[0] = self.prev_job
                    self.prev_job = None
                # 否则返回空字典（表示当前无新作业）
                valid_jobs = rm_pb2.JsonResponse()
                valid_jobs.response = json.dumps(job_to_run_dict)
                return valid_jobs
            ##modified part        
            except Exception as e:
                # somewhere there is logger called in workload. I am just
                # trying to avoid that possibly
                print("Exception e {}".format(e))
                traceback.print_exc()
                # pass

    def _get_avg_jct(self, time_dict):
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

    def _plot_graphs(self):
        """
        Post Simulation Plot Graphs
        """
        # 设置 matplotlib 样式参数
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        matplotlib.rcParams["font.size"] = 11
        matplotlib.rcParams["axes.labelsize"] = 12
        matplotlib.rcParams["axes.titlesize"] = 14
        matplotlib.rcParams["xtick.labelsize"] = 10
        matplotlib.rcParams["ytick.labelsize"] = 10
        matplotlib.rcParams["legend.fontsize"] = 10
        
        # 定义颜色和线型
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        linestyles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
        
        # print("plot called")
        jct_dict = defaultdict(dict)
        file_names_job_stats = list()
        file_names_cluster_stats = list()

        for scheduler in self.schedulers:
            file_names_job_stats = list()
            file_names_cluster_stats = list()
            for load in self.list_jobs_per_hour:
                stat_fname = f"result/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_{scheduler}_{self.acceptance_policies[0]}_load_{load}_job_stats.json"
                print(stat_fname)
                with open(stat_fname, "r") as fin:
                    data_job = json.load(fin)
                print(scheduler, load)
                jct_dict[scheduler][load] = self._get_avg_jct(data_job)
                # print("Wrote to dict")
        
        # Plot Average JCT
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 6)
        print("Job completion dict {}".format(jct_dict))
        write_folder = f"./plots/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_jct"
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
        
        for idx, scheduler in enumerate(self.schedulers):
            plot_list = list()
            for load in self.list_jobs_per_hour:
                plot_list.append(jct_dict[scheduler][load])
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            marker = markers[idx % len(markers)]
            ax1.plot(
                self.list_jobs_per_hour, 
                plot_list, 
                label=scheduler,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.5,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=1.5
            )
        
        ax1.set_xlabel("Jobs Per Hour", fontweight='bold') 
        ax1.set_ylabel("Average Job Completion Time (seconds)", fontweight='bold') 
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.set_title("Average Job Completion Time vs Load", fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(write_folder, f"load_{load}_jct.pdf"),
            format="pdf",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

        # plot cdf
        for load in self.list_jobs_per_hour:
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(10, 6)
            for idx, scheduler in enumerate(self.schedulers):
                stat_fname = f"result/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_{scheduler}_{self.acceptance_policies[0]}_load_{load}_job_stats.json"
                try:
                    with open(stat_fname, "r") as fin:
                        data_job = json.load(fin)
                    vals = data_job.values()
                    vals = [val[1] - val[0] for val in vals]  # JCT in seconds
                    vals = sorted(vals)
                    # Convert to hours
                    vals_hours = [val / 3600.0 for val in vals]
                    plot_y_val = list()
                    for i, val in enumerate(vals_hours):
                        plot_y_val.append(float(i) / len(vals_hours))
                    
                    color = colors[idx % len(colors)]
                    linestyle = linestyles[idx % len(linestyles)]
                    ax1.plot(
                        vals_hours, 
                        plot_y_val,
                        label=scheduler,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.5,
                        alpha=0.8
                    )
                except FileNotFoundError:
                    print(f"Warning: File not found: {stat_fname}, skipping...")
                    continue
            
            ax1.set_xlabel("Job Completion Time (hours)", fontweight='bold')
            ax1.set_ylabel("Cumulative Distribution Function (CDF)", fontweight='bold')
            ax1.set_title(f"CDF of Job Completion Time (Load: {load} jobs/hour)", fontweight='bold', pad=15)
            ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_xlim(left=0)
            ax1.set_ylim(bottom=0, top=1.0)
            plt.tight_layout()
            write_folder = f"./plots/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_cdf_load_{load}"
            if not os.path.exists(write_folder):
                os.makedirs(write_folder)
            plt.savefig(
                os.path.join(write_folder, f"load_{load}_cdf.pdf"),
                format="pdf",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()            
                # plt.savefig(
                #     os.path.join(write_folder, f"{scheduler}_load_{load}_cdf.pdf"),
                #     format="pdf",
                #     dpi=600,
                #     bbox_inches="tight",
                # )

        # plot GPU demand and free GPUs
        for load in self.list_jobs_per_hour:
            for scheduler in self.schedulers:
                stat_fname = f"result/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_{scheduler}_{self.acceptance_policies[0]}_load_{load}_cluster_stats.json"
                try:
                    with open(stat_fname, "r") as fin:
                        data_job = json.load(fin)
                    gpu_demand = list()
                    free_gpus = list()
                    time_points = list()
                    for d in data_job.keys():
                        time_points.append(float(d))
                        gpu_demand.append(data_job[d]["gpu_demand"])
                        free_gpus.append(data_job[d]["free_gpus"])
                    
                    # Sort by time
                    sorted_data = sorted(zip(time_points, gpu_demand, free_gpus))
                    time_points = [t for t, _, _ in sorted_data]
                    gpu_demand = [g for _, g, _ in sorted_data]
                    free_gpus = [f for _, _, f in sorted_data]
                    
                    # Convert time to hours for better readability
                    time_hours = [t / 3600.0 for t in time_points]
                    
                    # Plot GPU demand
                    fig1, ax1 = plt.subplots(1, 1)
                    fig1.set_size_inches(12, 6)
                    color_idx = self.schedulers.index(scheduler) % len(colors)
                    ax1.plot(
                        time_hours, 
                        gpu_demand,
                        label=scheduler,
                        color=colors[color_idx],
                        linewidth=2,
                        alpha=0.8
                    )
                    ax1.set_xlabel("Time (hours)", fontweight='bold')
                    ax1.set_ylabel("GPU Demand", fontweight='bold')
                    ax1.set_title(f"GPU Demand Over Time (Load: {load} jobs/hour, Scheduler: {scheduler})", 
                                fontweight='bold', pad=15)
                    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    plt.xticks(rotation=45, ha='right')
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
                    plt.tight_layout()
                    write_folder = f"./plots/{self.exp_prefix}_{self.job_ids_to_track[0]}_{self.job_ids_to_track[1]}_{scheduler}_load_{load}"
                    if not os.path.exists(write_folder):
                        os.makedirs(write_folder)
                    plt.savefig(
                        os.path.join(write_folder, f"{scheduler}_load_{load}_gpu_demand.pdf"),
                        format="pdf",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close()
                    
                    # Plot free GPUs
                    fig2, ax2 = plt.subplots(1, 1)
                    fig2.set_size_inches(12, 6)
                    ax2.plot(
                        time_hours, 
                        free_gpus,
                        label=scheduler,
                        color=colors[color_idx],
                        linewidth=2,
                        alpha=0.8
                    )
                    ax2.set_xlabel("Time (hours)", fontweight='bold')
                    ax2.set_ylabel("Free GPUs", fontweight='bold')
                    ax2.set_title(f"Free GPUs Over Time (Load: {load} jobs/hour, Scheduler: {scheduler})", 
                                fontweight='bold', pad=15)
                    ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    plt.xticks(rotation=45, ha='right')
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(write_folder, f"{scheduler}_load_{load}_free_gpu.pdf"),
                        format="pdf",
                        dpi=600,
                        bbox_inches="tight",
                    )
                    plt.close()
                except FileNotFoundError:
                    print(f"Warning: File not found: {stat_fname}, skipping...")
                    continue            


    def _clean_sim_job(self, new_job: dict) -> dict:
        """
        Preprocesses the job for simulations.
        Cleans some fields and non serializable input
        """
        # new_job_time = random.randint(36000, 86400)
        # new_job["job_total_iteration"] = new_job_time
        # new_job["job_duration"] = new_job_time
        new_job["simulation"] = True
        new_job["submit_time"] = new_job["job_arrival_time"]
        # temporary fix not sure why this is happening though
        if "logger" in new_job:
            new_job.pop("logger")
        if "job_task" in new_job:
            new_job.pop("job_task")
        # 保存模型的关键信息，特别是 tput 矩阵，以便后续计算 tput
        if "job_model" in new_job and new_job["job_model"] is not None:
            job_model = new_job["job_model"]
            # 保存 tput 矩阵（转换为列表以便序列化）
            if hasattr(job_model, "tput") and job_model.tput is not None:
                import numpy as np
                if isinstance(job_model.tput, np.ndarray):
                    new_job["tput_matrix"] = job_model.tput.tolist()
            # 移除 job_model 对象（无法序列化）
            new_job.pop("job_model")
        
        # Handle NumPy arrays (convert to list or remove)
        # This is needed because JSON cannot serialize NumPy arrays
        import numpy as np
        keys_to_remove = []
        for key, value in new_job.items():
            if isinstance(value, np.ndarray):
                # Option 1: Convert to list (if you need the data)
                # new_job[key] = value.tolist()
                # Option 2: Remove (if not needed for simulation)
                keys_to_remove.append(key)
        
        # Remove NumPy arrays
        for key in keys_to_remove:
            new_job.pop(key)

        # Set num_GPUs from job_gpu_demand (needed for job_state.add_new_jobs)        
        # Initialize allocated resources to 0 (job hasn't been placed yet)
        if not self.multigpu:
            new_job["job_gpu_demand"] = 1
        new_job["num_GPUs"] = new_job["job_gpu_demand"]

        new_job["num_GPUs_allocated"] = 0
        new_job["cpus_allocated"] = 0
        new_job["mem_allocated"] = 0
        new_job["sspeed_allocated"] = 0

        return new_job

    def _generate_workload(self, workload_config):
        """
        Generate workload for a given config
        """
        # set the random seed before generating the workload
        random.seed(self.random_seed)
        # print("After random seed")
        return Workload(
            self.cluster_job_log,
            scheduler=self.schedulers[0],
            jobs_per_hour=workload_config["load"],
            exponential=self.exponential,
            multigpu=self.multigpu,
            small_trace=self.small_trace,
            series_id_filter=self.job_ids_to_track,
            model_class_split=self.model_class_split,
            # TODO: Fix this
            per_server_size=[
                self.gpus_per_machine,
                self.num_cpu_cores,
                self.memory_per_machine,
                500,
                40,
            ],
            num_jobs_default=self.num_jobs_default,
        )

    def _generate_simulator_configs(self):
        for scheduler in self.schedulers:
            for placement_policy in self.placement_policies:
                for acceptance_policy in self.acceptance_policies:
                    for load in self.list_jobs_per_hour:
                        self.simulator_config.append(
                            {
                                "scheduler": scheduler,
                                "load": load,
                                "start_id_track": self.job_ids_to_track[0],
                                "stop_id_track": self.job_ids_to_track[1],
                                "placement_policy": placement_policy,
                                "acceptance_policy": acceptance_policy,
                            }
                        )

    def setup_cluster(self):
        """
        Cluster setup
        """
        count = 0
        for _ in range(self.number_of_machines):
            count += 1
            request_to_rm = rm_pb2.RegisterRequest()
            request_to_rm.ipaddr = ""
            request_to_rm.numGPUs = self.gpus_per_machine
            request_to_rm.gpuUUIDs = "\n".join(
                [str(x) for x in range(self.gpus_per_machine)]
            )
            request_to_rm.memoryCapacity = self.memory_per_machine
            request_to_rm.numCPUcores = self.num_cpu_cores
            request_to_rm.numaAvailable = self.is_numa_available
            request_to_rm.cpuMaping[0] = 0
            with grpc.insecure_channel(self.ipaddr_rm) as channel:
                stub = rm_pb2_grpc.RMServerStub(channel)
                response = stub.RegisterWorker(request_to_rm)
        # print("Number of machines sent {}".format(count))
        return None

    def NotifyCompletion(self, request, context):
        """
        Call to notify that jobs done
        """
        pass


def parse_args(parser):
    """
    Parse arguments
    """
    parser.add_argument(
        "--sim-type",
        choices=["trace-synthetic", "trace-actual", "synthetic"],
        type=str,
        help="Type of simulation, trace-synthetic:philly trace with specific load, trace-actual : actual replay of philly trace, synthetic:synthetic trace",
    )

    parser.add_argument(
        "--cluster-job-log",
        type=str,
        default="",
        help="Name of the cluster log file to run",
    )
    parser.add_argument("--jobs-per-hour", type=int, default=5, help="Jobs per hour")
    parser.add_argument(
        "--start-job-track", type=int, default=0, help="Start ID of job to track"
    )

    parser.add_argument(
        "--end-job-track", type=int, default=1000, help="End ID of job to track"
    )
    parser.add_argument(
        "--scheduler", type=str, default="Fifo", help="Name of the scheduler"
    )
    parser.add_argument(
        "--placement_preference", type=str, default="consolidated", help="placement preference"
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        help="Unique name for prefix log files, makes sure it is the same as resource manager",
    )
    args = parser.parse_args()
    return args


def launch_server(args) -> grpc.Server:
    """
    Launches the server
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    runner = SimulatorRunner(
            args.cluster_job_log,
            np.arange(args.jobs_per_hour, args.jobs_per_hour+1, 1.0).tolist(),
            (args.start_job_track, args.end_job_track),
            [
                # "Tiresias",
                # "Optimus",
                # "Fifo",
                # "Las",
                "Srtf",
                # "New",
                # "Synergy_fifo"
                "Synergy_srtf"
            ],
            ["Place"],
            ["AcceptAll"],
            exp_prefix=args.exp_prefix,
        )
    
    simulator_pb2_grpc.add_SimServerServicer_to_server(runner, server)
    server.add_insecure_port("[::]:50050")
    server.start()
    print("Print Server started")
    return server,runner


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Arguments for simulation"))
    try:
        server,runner = launch_server(args)
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Exit by ctrl c")
    runner._plot_graphs()

    # simulator = SimulatorRunner(
    # args.cluster_job_log,
    # args.jobs_per_hour,
    # (args.start_job_track, args.end_job_track),
    # args.scheduler,
    # )
    # simulator.run_simulation()
