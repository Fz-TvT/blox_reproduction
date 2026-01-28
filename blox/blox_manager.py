import os
import sys
import time
import copy
import grpc
import json
import logging
import argparse
import pandas as pd
import time
from concurrent import futures

from typing import Tuple, List

# import scheduler
# import placement

# from profile_parsers import pparsers
sys.path.append(os.path.dirname(__file__))
import deployment.grpc_server_rm as rm_serve
import deployment.grpc_client_rm as rm_client

# from cluster_state import ClusterState
# from job_state import JobState


class BloxManager(object):
    """
    Implements the Blox bookkeeping interface, including accepting, scheduling and running jobs
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        self.args = args
        self.scheduler_name = args.scheduler_name
        self.placement_name = args.placement_name
        self.acceptance_policy = args.acceptance_policy
        self.exp_prefix = args.exp_prefix
        self.load = args.load
        self.round_duration = args.round_duration
        self.comm_node_manager = rm_client.ResourceManagerComm(
            node_manager_port=args.node_manager_port
        )
        self.priority_thresh = 3600 * 10  # above this we will have priority thresh
        self.server, self.rmserver = launch_server(
            rm_server_rpc_port=args.central_scheduler_port,
            simulator_rpc_port=args.simulator_rpc_port,
        )
        self.time = 0
        self.terminate = False
        return None

    def reset(self, args: argparse.ArgumentParser) -> None:
        """
        Change some runtime parameters post launch
        """
        self.args = args
        self.scheduler_name = args.scheduler_name
        self.placement_name = args.placement_name
        self.acceptance_policy = args.acceptance_policy
        self.exp_prefix = args.exp_prefix
        self.load = args.load
        self.round_duration = args.round_duration
        # self.comm_node_manager = rm_client.ResourceManagerComm()
        # self.priority_thresh = 3600 * 1000  # above this we will have priority thresh
        # self.server, self.rmserver = launch_server()
        self.time = 0
        self.terminate = False
        return None

    def terminate_server(self):
        """
        Shut down grpc server
        """
        # print("In terminate")
        self.server.stop(0)

    def update_cluster(self, cluster_state):
        """
        Update cluster state

        Args:
            cluster_state - Cluster State Object
        Returns:
            new_nodes - New nodes added in this cluster
        """
        # new_nodes = cluster_state.update()
        new_nodes = self.rmserver.get_new_nodes()
        cluster_state.update(new_nodes)
        return new_nodes

    def _convert_res_map_keys_to_node_id(self, res_map):
        """
        将 res_map 中的 ServerWrapper 键转换为 node_id
        这样 res_map 就可以被 JSON 序列化。
        
        Args:
            res_map: 资源映射字典，键可能是 ServerWrapper 对象或整数
        
        Returns:
            转换后的 res_map,所有键都是整数 node_id
        """
        if not res_map:
            return res_map
        
        converted_map = {}
        for server_key, resources in res_map.items():
            # 提取 node_id
            if hasattr(server_key, 'node_id'):
                node_id = server_key.node_id
            elif hasattr(server_key, 'server_id'):
                node_id = server_key.server_id
            elif isinstance(server_key, int):
                node_id = server_key
            else:
                # 如果无法提取 node_id，跳过这个条目
                continue
            
            # 使用 node_id 作为键
            converted_map[node_id] = resources
        
        return converted_map

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

    def update_metrics(self, cluster_state, job_state):
        """
        Perform metric collection also prunes the jobs.
        Args:
            cluster_state: Cluster State object
            job_state: Job State object

        Return:
            None
        """
        job_id_to_fetch = list()
        ipaddress_to_fetch_from = list()
        if_simulation = list()

        running_jobs = []
        non_running_jobs = []
        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                running_jobs.append(jid)
                job_id_to_fetch.append(jid)
                if_simulation.append(job_state.active_jobs[jid]["simulation"])
                ipaddress_to_fetch_from.append(
                    job_state.active_jobs[jid]["running_ip_address"]
                )
            else:
                non_running_jobs.append(jid)
        print(f"[DEBUG] update_metrics: Running jobs: {running_jobs}")
        print(f"[DEBUG] update_metrics: Non-running jobs: {non_running_jobs}")
        metric_data = self.comm_node_manager.get_metrics(
            job_id_to_fetch,
            ipaddress_to_fetch_from,
            if_simulation,
            self.round_duration,
            job_state.active_jobs,
        )
        # 去除 synergy 相关的无效输出
        # print("Metric Data {}".format(metric_data))

        job_state.update_metrics(metric_data, self.round_duration)
        # prune jobs which have been completed

        jid_to_terminate = list()
        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                if jid in job_state.active_jobs:
                    if "tracked_metrics" in job_state.active_jobs[jid]:
                        if "job_exit" in job_state.active_jobs[jid]["tracked_metrics"]:
                            if (
                                job_state.active_jobs.get(jid)
                                .get("tracked_metrics")
                                .get("job_exit")
                                == True
                            ):
                                # TODO: put a condition to check if need
                                # plotting
                                if (
                                    jid >= job_state.job_ids_to_track[0]
                                    and jid <= job_state.job_ids_to_track[-1]
                                ):
                                    # log the exit
                                    job_state.job_completion_stats[jid] = [
                                        job_state.active_jobs[jid]["submit_time"],
                                        self.time,
                                    ]

                                    # Deep copy job data and convert ServerWrapper keys in res_map to node_id
                                    job_data = copy.deepcopy(job_state.active_jobs[jid])
                                    # Convert ServerWrapper keys in res_map to node_id (int) for JSON serialization
                                    if "res_map" in job_data and job_data["res_map"]:
                                        job_data["res_map"] = self._convert_res_map_keys_to_node_id(
                                            job_data["res_map"]
                                        )
                                    job_state.job_runtime_stats[jid] = job_data
                                    # track completion_time and submission_time as maintained in the Pollux Job object

                                jid_to_terminate.append(jid)
                                # delete GPU utilization
                                _free_gpu_by_jobid(jid, cluster_state.gpu_df)
                                # Free CPU and memory resources for completed job
                                _free_server_resources_by_jobid(jid, job_state.active_jobs, cluster_state)
                                # log the finished jobs
                                job_state.finished_job[jid] = 1

        # additional information for logging responsiveness
        for jid in job_state.active_jobs:
            if job_state.active_jobs[jid]["is_running"] == True:
                if jid in job_state.active_jobs:
                    if "job_launched_first_time" in job_state.active_jobs[jid]:
                        if (
                            job_state.active_jobs.get(jid).get(
                                "job_launched_first_time"
                            )
                            == True
                        ):
                            # TODO: put a condition to check if need
                            # plotting
                            if (
                                jid >= job_state.job_ids_to_track[0]
                                and jid <= job_state.job_ids_to_track[-1]
                            ):
                                # log the exit
                                job_state.job_responsiveness_stats[jid] = [
                                    job_state.active_jobs[jid]["submit_time"],
                                    self.time,
                                ]

        for jid in jid_to_terminate:
            job_state.active_jobs.pop(jid)

        # Update cluster_state.server_resource_usage based on running jobs' res_map
        # This ensures server_resource_usage reflects actual resource usage from running jobs
        # This is especially important for Synergy_FIFO scheduler
        scheduler = os.environ.get("sched_policy", "")
        if scheduler == "Synergy_fifo":
            # Recalculate server_resource_usage from running jobs' res_map
            # This ensures consistency between placement calculations and actual state
            for node_id in cluster_state.server_map:
                # Get capacity from server_map
                cpu_capacity = cluster_state.server_map[node_id].get("numCPUcores", 0)
                mem_capacity = cluster_state.server_map[node_id].get("memoryCapacity", 0)
                gpu_capacity = cluster_state.server_map[node_id].get("numGPUs", 0)
                
                # If capacity is 0, calculate from GPU count (proportional allocation: 3 CPU/GPU, 62.5 GB/GPU)
                if gpu_capacity > 0:
                    if cpu_capacity == 0:
                        cpu_capacity = gpu_capacity * 3
                    if mem_capacity == 0:
                        mem_capacity = gpu_capacity * 62.5
                
                # Initialize if not exists
                if node_id not in cluster_state.server_resource_usage:
                    cluster_state.server_resource_usage[node_id] = {
                        "cpu_used": 0.0,
                        "memory_used": 0.0,
                        "gpu_used": 0
                    }
                
                # Calculate used resources from running jobs' res_map
                cpu_used = 0.0
                memory_used = 0.0
                gpu_used = 0
                
                for jid, job in job_state.active_jobs.items():
                    if job.get("is_running", False):
                        res_map = job.get("res_map", {})
                        if res_map:
                            for server_key, resources in res_map.items():
                                # Extract node_id from ServerWrapper or dict key
                                if hasattr(server_key, 'node_id'):
                                    res_node_id = server_key.node_id
                                elif hasattr(server_key, 'server_id'):
                                    res_node_id = server_key.server_id
                                elif isinstance(server_key, int):
                                    res_node_id = server_key
                                else:
                                    continue
                                
                                if res_node_id == node_id:
                                    # Add resources from this job on this node
                                    cpu_used += resources.get("cpu", 0) if isinstance(resources, dict) else 0
                                    memory_used += resources.get("mem", 0) if isinstance(resources, dict) else 0
                                    gpu_used += resources.get("gpu", 0) if isinstance(resources, dict) else 0
                
                # Update server_resource_usage
                cluster_state.server_resource_usage[node_id]["cpu_used"] = cpu_used
                cluster_state.server_resource_usage[node_id]["memory_used"] = memory_used
                cluster_state.server_resource_usage[node_id]["gpu_used"] = gpu_used
                
                # Ensure non-negative and don't exceed capacity
                cluster_state.server_resource_usage[node_id]["cpu_used"] = max(
                    0, min(cpu_used, cpu_capacity)
                )
                cluster_state.server_resource_usage[node_id]["memory_used"] = max(
                    0, min(memory_used, mem_capacity)
                )
                cluster_state.server_resource_usage[node_id]["gpu_used"] = max(
                    0, min(gpu_used, gpu_capacity)
                )

        # update cluster use
        total_jobs, jobs_in_queue, jobs_running = _get_jobs_status(job_state)

        # gpu utilization

        free_gpus = len(
            cluster_state.gpu_df[cluster_state.gpu_df["IN_USE"] == False][
                "GPU_ID"
            ].tolist()
        )

        gpu_demand = 0
        for jid in job_state.active_jobs:
            gpu_demand += job_state.active_jobs[jid]["num_allocated_gpus"]

        # Get cluster resource usage statistics
        resource_usage = get_cluster_resource_usage(cluster_state)
        
        cluster_state.cluster_stats[cluster_state.time] = {
            "total_jobs": total_jobs,
            "jobs_in_queue": jobs_in_queue,
            "jobs_running": jobs_running,
            "free_gpus": free_gpus,
            "gpu_demand": gpu_demand,
            "cpu_used": resource_usage["total_cpu_used"],
            "cpu_capacity": resource_usage["total_cpu_capacity"],
            "cpu_utilization_percent": resource_usage["cpu_utilization_percent"],
            "memory_used": resource_usage["total_memory_used"],
            "memory_capacity": resource_usage["total_memory_capacity"],
            "memory_utilization_percent": resource_usage["memory_utilization_percent"],
        }

        #

        total_jobs, jobs_in_queue, jobs_running = _get_jobs_status(job_state)

        # gpu utilization

        free_gpus = len(
            cluster_state.gpu_df[cluster_state.gpu_df["IN_USE"] == False][
                "GPU_ID"
            ].tolist()
        )

        gpu_demand = 0
        for jid in job_state.active_jobs:
            gpu_demand += job_state.active_jobs[jid]["num_allocated_gpus"]

        cluster_state.cluster_stats[cluster_state.time] = {
            "total_jobs": total_jobs,
            "jobs_in_queue": jobs_in_queue,
            "jobs_running": jobs_running,
            "free_gpus": free_gpus,
            "gpu_demand": gpu_demand,
        }

        # find the jobs have been finished
        # print(
        # "Not finished job {}".format(
        # list(set(job_state.job_ids_to_track) - set(job_state.finished_job))
        # )
        # )

        if all(jid in job_state.finished_job for jid in job_state.job_ids_to_track):
            os.makedirs("result", exist_ok=True)
            with open(
                f"result/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_load_{self.load}_job_stats.json",
                "w",
            ) as fopen:
                avg_jct = self._get_avg_jct(job_state.job_completion_stats)
                print(
                    f"Scheduler: {self.scheduler_name}, Acceptance Policy: {self.acceptance_policy}, Load: {self.load}, Avg JCT {avg_jct}"
                )
                json.dump(job_state.job_completion_stats, fopen)

            with open(
                f"result/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_load_{self.load}_cluster_stats.json",
                "w",
            ) as fopen:
                json.dump(cluster_state.cluster_stats, fopen)
            # sys.exit(0)
            with open(
                f"result/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_load_{self.load}_run_time_stats.json",
                "w",
            ) as fopen:
                json.dump(job_state.job_runtime_stats, fopen)

            with open(
                f"result/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_load_{self.load}_responsivness.json",
                "w",
            ) as fopen:
                avg_responsiveness = self._get_avg_jct(
                    job_state.job_responsiveness_stats
                )
                print(
                    f"Scheduler: {self.scheduler_name}, Acceptance Policy: {self.acceptance_policy}, Load: {self.load}, Avg responsiveness {avg_responsiveness}"
                )
                json.dump(job_state.job_responsiveness_stats, fopen)
            with open(
                f"result/{self.exp_prefix}_{job_state.job_ids_to_track[0]}_{job_state.job_ids_to_track[-1]}_{self.scheduler_name}_{self.acceptance_policy}_load_{self.load}_custom_metrics.json",
                "w",
            ) as fopen:
                json.dump(job_state.custom_metrics, fopen)

            self.terminate = True
        return None

    def pop_wait_queue(self, is_simulation: bool):
        """
        Get jobs which have arrived during the previous scheduling round
        """

        if is_simulation:
            # get jobs for simulation
            new_jobs = self.rmserver.get_jobs_sim(self.time)

        else:
            # get jobs for real cluster
            new_jobs = self.rmserver.get_new_jobs()
        return new_jobs

    def exec_jobs(
        self,
        jobs_to_launch: dict,
        jobs_to_terminate: list,
        cluster_state,
        active_jobs,
    ) -> None:
        """
        First terminates the jobs. Then marks the jobs to launch.
        Args:
            jobs_to_launch: {Job_ID: [GPUs to launch]}
            jobs_to_terminate : List of Job IDs to Terminate
            cluster_state : ClusterState class
            active_jobs: JobState
        Return:
          None
        """
        terminate_list_id = list()
        terminate_rank_0_ipaddr = list()
        terminate_ipaddr = list()
        terminate_simulation = list()
        for jid in jobs_to_terminate:
            # find ipaddresses for corresponding jobs to terminate
            running_ipddr = list(
                set(_find_ipaddr_by_job_ids(jid, cluster_state.gpu_df))
            )
            rank_0_ipaddr = active_jobs.active_jobs[jid]["rank_0_ip"]
            # terminate_list_id.extend([jid] * len(running_ipddr))
            terminate_list_id.append(jid)
            terminate_rank_0_ipaddr.append(rank_0_ipaddr)
            terminate_ipaddr.append(running_ipddr)
            # terminate_simulation.extend(
            # [active_jobs.active_jobs[jid]["simulation"]] * len(running_ipddr)
            # )
            terminate_simulation.append(active_jobs.active_jobs[jid]["simulation"])
            # mark the job that is running is false
            active_jobs.active_jobs[jid]["is_running"] = False
            active_jobs.active_jobs[jid]["rank_0_ip"] = None
            active_jobs.active_jobs[jid]["running_ip_address"] = None
            # the job was suspended
            active_jobs.active_jobs[jid]["suspended"] = 1
            # mark corresponding GPUs on which the jobs are running as
            # available
            _free_gpu_by_jobid(jid, cluster_state.gpu_df)
            # Free CPU and memory resources for this job
            _free_server_resources_by_jobid(jid, active_jobs.active_jobs, cluster_state)

        self.comm_node_manager.terminate_jobs(
            terminate_list_id,
            terminate_rank_0_ipaddr,
            terminate_ipaddr,
            terminate_simulation,
        )

        # jobs terminated
        def launch_job_func(jid):
            # Handle both old format (list of GPU IDs) and new format (dict with 'nodes' and 'gpus')
            placement_info = jobs_to_launch[jid]
            if isinstance(placement_info, dict) and 'gpus' in placement_info:
                # New format: {job_id: {'nodes': {node_id: [gpu_ids]}, 'gpus': [all_gpu_ids]}}
                gpus_to_launch = placement_info['gpus']
                node_placement = placement_info.get('nodes', {})  # {node_id: [gpu_ids]}
                # Print node-level placement information
            else:
                # Old format: {job_id: [gpu_ids]} (backward compatibility)
                gpus_to_launch = placement_info
                node_placement = {}
            
            ipaddress_to_launch = _find_ipaddr_by_gpu_ids(
                gpus_to_launch, cluster_state.gpu_df
            )
            local_gpu_ids = _find_local_gpu_id(gpus_to_launch, cluster_state.gpu_df)
            self.comm_node_manager.launch_job(
                jid, active_jobs.active_jobs[jid], local_gpu_ids, ipaddress_to_launch
            )
            active_jobs.active_jobs[jid]["is_running"] = True
            active_jobs.active_jobs[jid]["rank_0_ip"] = list(set(ipaddress_to_launch))[
                0
            ]

            active_jobs.active_jobs[jid]["running_ip_address"] = list(
                set(ipaddress_to_launch)
            )
            if "suspended" in active_jobs.active_jobs[jid]:
                active_jobs.active_jobs[jid]["suspended"] = 0
            _mark_gpu_in_use_by_gpu_id(gpus_to_launch, jid, cluster_state.gpu_df)
            res_map = active_jobs.active_jobs[jid].get("res_map", {})
            if res_map:
                _update_server_resource_usage(jid, res_map, cluster_state, operation="add")
            return True

        # for jid in jobs_to_launch:
        with futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_results = [
                executor.submit(launch_job_func, jid) for jid in jobs_to_launch
            ]

            results = [
                future.result() for future in futures.as_completed(future_results)
            ]

        # 检查运行中的作业是否有 GPU 需求大于 1 的
        running_jobs_with_multi_gpu = []
        for jid in active_jobs.active_jobs:
            job = active_jobs.active_jobs[jid]
            if job.get("is_running", False):
                # 获取 GPU 需求（优先使用 job_gpu_demand，否则使用 num_GPUs）
                gpu_demand = job.get("job_gpu_demand")
                if gpu_demand > 1:
                    running_jobs_with_multi_gpu.append({
                        "job_id": jid,
                        "gpu_demand": gpu_demand,
                        "num_GPUs_allocated": job.get("num_GPUs_allocated", 0),
                    })
        
        if running_jobs_with_multi_gpu:
            print(f"[WARNING] Found {len(running_jobs_with_multi_gpu)} running jobs with GPU demand > 1:")
            for job_info in running_jobs_with_multi_gpu:
                print(f"  Job {job_info['job_id']}: GPU demand={job_info['gpu_demand']}, Allocated={job_info['num_GPUs_allocated']}")

        # update the time for training

        for jid in active_jobs.active_jobs:
            if jid in jobs_to_terminate:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            elif jid in jobs_to_launch:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            elif active_jobs.active_jobs[jid]["is_running"]:
                active_jobs.active_jobs[jid]["time_since_scheduled"] = 0
            else:
                active_jobs.active_jobs[jid][
                    "time_since_scheduled"
                ] += self.round_duration
                if (
                    active_jobs.active_jobs[jid]["time_since_scheduled"]
                    >= self.priority_thresh
                ):
                    active_jobs.active_jobs[jid]["job_priority"] = 1


def _get_jobs_status(job_state) -> Tuple[int]:
    """
    Get number of jobs running, jobs in queue and total jobs
    """
    total_jobs = len(job_state.active_jobs.keys())
    jobs_in_queue = 0
    jobs_running = 0
    for jid in job_state.active_jobs:
        if job_state.active_jobs[jid]["is_running"]:
            jobs_running += 1
        if not job_state.active_jobs[jid]["is_running"]:
            jobs_in_queue += 1
    return (total_jobs, jobs_in_queue, jobs_running)


# NOTE: Utilities for querying the GPU DF
def _find_ipaddr_by_job_ids(job_id: str, gpu_df: pd.DataFrame) -> List[str]:
    """
    Given a jobID finds the ip-addresses on which the job runs.
    Args:
        job_id: ID of the job to find corresponding ipaddress
    Returns:
        List of IP addresses on which the job is running
    """
    return gpu_df[gpu_df["JOB_IDS"] == job_id]["IP_addr"].tolist()


def _find_ipaddr_by_gpu_ids(gpu_ids: List[int], gpu_df: pd.DataFrame) -> List[str]:
    """
    Return the IP address for given GPU IDs

    Args:
        gpu_ids: GPU ids to search
    Returns:
        List of IP addresses for corresponding gpu_ids
    """
    ipaddress_to_launch = list()
    for gid in gpu_ids:
        gid_ipaddr = gpu_df[gpu_df["GPU_ID"] == gid]["IP_addr"].tolist()
        assert len(gid_ipaddr) == 1, "Multiple IP addr for same GPU, something wrong"

        ipaddress_to_launch.extend(gid_ipaddr)
    return ipaddress_to_launch


def _free_gpu_by_jobid(job_id: int, gpu_df: pd.DataFrame) -> None:
    """
    Marks the corresponding GPU free for a given job ID
    Args:
        job_id: ID of the job to terminate
    """
    gpu_df.loc[gpu_df["JOB_IDS"] == job_id, ["JOB_IDS", "IN_USE"]] = (
        None,
        False,
    )
    return None


def _mark_gpu_in_use_by_gpu_id(
    gpu_id_list: List[int], job_id: int, gpu_df: pd.DataFrame
) -> None:
    """
    Marks the corresponding GPU in use for a given job ID
    Args:
        gpu_id_list : List of GPU ID's to terminate
        job_id: ID of the job to terminate
    """
    gpu_df.loc[gpu_df["GPU_ID"].isin(gpu_id_list), ["JOB_IDS", "IN_USE"]] = (
        job_id,
        True,
    )
    return None


def _update_server_resource_usage(
    job_id: int,
    res_map: dict,
    cluster_state,
    operation: str = "add"
) -> None:
    """
    Update CPU, memory, and GPU usage on servers based on job's res_map
    Args:
        job_id: Job ID
        res_map: Resource map from placement (format: {ServerWrapper: {'cpu': int, 'mem': float, 'gpu': int, ...}})
        cluster_state: ClusterState object
        operation: "add" to allocate resources, "remove" to free resources
    """
    if not res_map:
        return
    
    multiplier = 1 if operation == "add" else -1
    
    for server_key, resources in res_map.items():
        # Handle both ServerWrapper objects and node_id integers
        if hasattr(server_key, 'node_id'):
            node_id = server_key.node_id
        elif isinstance(server_key, int):
            node_id = server_key
        else:
            # Try to get node_id from server_map_entry if it's a dict-like object
            try:
                node_id = server_key.get('node_id', None) if hasattr(server_key, 'get') else None
            except:
                continue
        
        if node_id is None or node_id not in cluster_state.server_resource_usage:
            continue
        
        # Initialize gpu_used if not exists (for backward compatibility)
        if "gpu_used" not in cluster_state.server_resource_usage[node_id]:
            cluster_state.server_resource_usage[node_id]["gpu_used"] = 0
            
        cpu_alloc = resources.get('cpu', 0) if isinstance(resources, dict) else 0
        mem_alloc = resources.get('mem', 0) if isinstance(resources, dict) else 0
        gpu_alloc = resources.get('gpu', 0) if isinstance(resources, dict) else 0
        
        cluster_state.server_resource_usage[node_id]["cpu_used"] += cpu_alloc * multiplier
        cluster_state.server_resource_usage[node_id]["memory_used"] += mem_alloc * multiplier
        cluster_state.server_resource_usage[node_id]["gpu_used"] += gpu_alloc * multiplier
        
        # Ensure non-negative
        cluster_state.server_resource_usage[node_id]["cpu_used"] = max(
            0, cluster_state.server_resource_usage[node_id]["cpu_used"]
        )
        cluster_state.server_resource_usage[node_id]["memory_used"] = max(
            0, cluster_state.server_resource_usage[node_id]["memory_used"]
        )
        cluster_state.server_resource_usage[node_id]["gpu_used"] = max(
            0, cluster_state.server_resource_usage[node_id]["gpu_used"]
        )


def _free_server_resources_by_jobid(
    job_id: int,
    active_jobs: dict,
    cluster_state
) -> None:
    """
    Free CPU and memory resources for a terminated job
    Args:
        job_id: Job ID to free resources for
        active_jobs: Active jobs dictionary
        cluster_state: ClusterState object
    """
    if job_id not in active_jobs:
        return
    
    job = active_jobs[job_id]
    res_map = job.get("res_map", {})
    
    if res_map:
        _update_server_resource_usage(job_id, res_map, cluster_state, operation="remove")


def get_cluster_resource_usage(cluster_state) -> dict:
    """
    Get cluster-wide CPU and memory usage statistics
    Args:
        cluster_state: ClusterState object
    Returns:
        Dictionary with cluster-wide resource usage statistics
    """
    total_cpu_used = 0.0
    total_memory_used = 0.0
    total_cpu_capacity = 0
    total_memory_capacity = 0
    
    # Iterate over server_map to ensure we include all nodes with capacity info
    # This ensures consistency between capacity and usage statistics
    for node_id, node_info in cluster_state.server_map.items():
        # Get capacity from server_map
        cpu_capacity = node_info.get("numCPUcores", 0)
        memory_capacity = node_info.get("memoryCapacity", 0)
        total_cpu_capacity += cpu_capacity
        total_memory_capacity += memory_capacity
        
        # Get usage from server_resource_usage (use .get() for safety)
        usage = cluster_state.server_resource_usage.get(node_id, {})
        cpu_used = usage.get("cpu_used", 0.0)
        memory_used = usage.get("memory_used", 0.0)
        total_cpu_used += cpu_used
        total_memory_used += memory_used
    
    # Also check for any nodes in server_resource_usage that might not be in server_map
    # (shouldn't happen normally, but handle edge cases)
    for node_id, usage in cluster_state.server_resource_usage.items():
        if node_id not in cluster_state.server_map:
            # Node exists in usage but not in server_map - still count usage
            total_cpu_used += usage.get("cpu_used", 0.0)
            total_memory_used += usage.get("memory_used", 0.0)
    
    cpu_utilization = (total_cpu_used / total_cpu_capacity * 100) if total_cpu_capacity > 0 else 0
    memory_utilization = (total_memory_used / total_memory_capacity * 100) if total_memory_capacity > 0 else 0
    
    return {
        "total_cpu_used": total_cpu_used,
        "total_memory_used": total_memory_used,
        "total_cpu_capacity": total_cpu_capacity,
        "total_memory_capacity": total_memory_capacity,
        "cpu_utilization_percent": cpu_utilization,
        "memory_utilization_percent": memory_utilization,
        "per_server_usage": cluster_state.server_resource_usage.copy()
    }


def _find_local_gpu_id(global_gpu_ids: List[int], gpu_df: pd.DataFrame) -> List[int]:
    """
    Given a list of Global GPU ID's find the corresponding local GPU id's

    Args:
        global_gpu_ids: Global GPU ID's
    Returns:
        local_gpu_ids: Local GPU id's corresponding to that value
    """
    local_gpu_id = list()
    # TODO: Get rid of this for loop using .isin
    for gid in global_gpu_ids:
        lgid = gpu_df[gpu_df["GPU_ID"] == gid]["Local_GPU_ID"].tolist()

        assert (
            len(lgid) == 1
        ), "Multiple Local GPUs for same global GPU ID, something wrong"

        local_gpu_id.extend(lgid)

    return local_gpu_id


# print(metric_data)
# utility functions
def launch_server(
    rm_server_rpc_port: int, simulator_rpc_port: int
) -> Tuple[grpc.Server, rm_serve.RMServer]:
    """
    Launches GRPC server and returns the server object
    Args:
        None
    Returns:
        server : GRPC server object
        rmserver : The class object to work with rmserver
    """
    rmserver = rm_serve.RMServer(simulator_rpc_port=simulator_rpc_port)
    server = rm_serve.start_server(rmserver, rm_server_rpc_port=rm_server_rpc_port)
    print("Server started")
    return server, rmserver
