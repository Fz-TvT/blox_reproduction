import pandas as pd
import copy
from typing import Tuple, List


class JobPlacement(object):
    def __init__(self, args):
        pass

    @staticmethod
    def copy_arguments(function):
        def function_wrapper(
            self, job_state, cluster_state, new_job_schedule, **kwargs
        ):
            return function(
                self,
                job_state.active_jobs,
                copy.deepcopy(new_job_schedule),
                copy.deepcopy(cluster_state.server_map),
                copy.deepcopy(cluster_state.gpu_df),
                **copy.deepcopy(kwargs),
            )

        return function_wrapper

    @copy_arguments.__func__
    def place(
        self,
        active_jobs: dict,
        new_job_schedule: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        **kwargs,
    ) -> dict:
        """
        parses the sorted_jobs dictionary and calls relevant placement policy

        # CAUTION: This function makes in place changes to active jobs and
        # gpu_df

        """
        job_order = new_job_schedule["job_order"]
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        job_to_launch = dict()
        launched_job_ids = list()
        # go over jobs in job order
        if scheduler == "Gavel":
            for idx, job_priority_sorted in enumerate(job_order):
                job_id, gpu_preference = list(job_priority_sorted.keys())[0]
                job = active_jobs[job_id]
                found = False
                if job["is_running"] == True:
                    if job["running_accel"] == gpu_preference:
                        # nothing to do here
                        continue
                    else:
                        # need to terminate this job trying to launch on
                        # different accelerator
                        jobs_to_terminate.append(job_id)
                        job["is_running"] = False
                        delete_job_by_id(gpu_df, job_id)

                if job_id in launched_job_ids:
                    # already launched the same ID in this round
                    continue
                if job["is_running"] == False:
                    # need to find placement only if job is not running
                    place_consolidated = (
                        job.get("placement_preference") == "consolidated"
                    )

                    free_gpus = find_free_GPUs_by_type(gpu_df, gpu_preference)
                    if place_consolidated:
                        placement, found = self._consolidated_placement(job, free_gpus)
                    else:
                        placement, found = self._scattered_placement(job, free_gpus)
                    if not found:
                        # no free GPUs
                        # find the GPU with same GPU preference in the reverse
                        # order of priority
                        for rev_idx in range(1, len(active_jobs) - idx):
                            potential_terminate_job_pair = job_order[-rev_idx]
                            if potential_terminate_job_pair[1] != gpu_preference:
                                # Job doesn't have the same preference
                                continue
                            else:
                                # the job has the same preference

                                # need to check if it is running
                                # and if it is running on the same as current
                                # preference
                                potential_terminate_job_info = active_jobs[
                                    potential_terminate_job_pair[0]
                                ]
                                if (
                                    potential_terminate_job_info["is_running"] == True
                                ) and (
                                    potential_terminate_job_info["running_accel"]
                                    == gpu_preference
                                ):
                                    # only terminate in case the training is
                                    # also happening on the same GPU as the
                                    # preference
                                    jobs_to_terminate.append(
                                        potential_terminate_job_pair[0]
                                    )
                                    potential_terminate_job_info["is_running"] = False
                                    # freeing up GPUs
                                    delete_job_by_id(
                                        gpu_df, potential_terminate_job_pair[0]
                                    )
                                    free_gpus = find_free_GPUs_by_type(
                                        gpu_df, gpu_preference
                                    )

                                    if place_consolidated:
                                        placement, found = self._consolidated_placement(
                                            job, free_gpus
                                        )
                                    else:
                                        placement, found = self._scattered_placement(
                                            job, free_gpus
                                        )

                                    if found:
                                        # we found the placement
                                        break

                                    # terminate this job
                                else:
                                    # job matching not found
                                    continue
                if found:
                    launched_job_ids.append(job_id)
                    job_to_launch[job_id] = placement
                    active_jobs[jid]["running_accel"] = gpu_preference
                    mark_gpu_in_use(gpu_df, placement, job_id)
                else:
                    break
    
            return jobs_to_terminate, jobs_to_launch

        elif scheduler=="Synergy_fifo":
            jobs_to_terminate = list()
            job_to_launch = dict()
            jobs_this_round = new_job_schedule["jobs_this_round"]
            
            # Calculate total GPU demand and capacity for proportional allocation
            total_gpu_demand = sum(
                job[1].get("job_gpu_demand", 0) for job in jobs_this_round
            )
            total_gpu_capacity = sum(
                node_info[node_id].get("numGPUs", 0) for node_id in node_info
            )
            
            # Track server resource usage (simplified - in practice need per-server tracking)
            server_resource_usage = {}
            for node_id in node_info:
                server_resource_usage[node_id] = get_server_available_resources(
                    node_id, node_info[node_id], gpu_df, active_jobs
                )
            
            # Process each job in order
            for job_tuple in jobs_this_round:
                job_id, job_info = job_tuple
                job = active_jobs[job_id]
                
                if job.get("is_running", False):
                    continue  # Skip already running jobs
                
                # Get job demand vector
                gpu_demand = job_info.get("job_gpu_demand", 0)
                cpu_demand = job_info.get("job_cpu_demand", job_info.get("job_cpu_demand_orig", 0))
                mem_demand = job_info.get("job_mem_demand", job_info.get("job_mem_demand_orig", 0))
                
                # Try to find placement with original demands
                placement, found = self._synergy_find_placement(
                    job_info, node_info, gpu_df, server_resource_usage, active_jobs,
                    gpu_demand, cpu_demand, mem_demand
                )
                
                # Step 1: If not found, check if demand > GPU proportional share
                if not found:
                    prop_alloc = calculate_gpu_proportional_allocation(
                        gpu_demand, total_gpu_demand, total_gpu_capacity
                    )
                    
                    # Check if current demand is greater than proportional
                    if cpu_demand > prop_alloc["cpu"] or mem_demand > prop_alloc["memory"]:
                        # Switch to GPU-proportional share and retry
                        cpu_demand = prop_alloc["cpu"]
                        mem_demand = prop_alloc["memory"]
                        placement, found = self._synergy_find_placement(
                            job_info, node_info, gpu_df, server_resource_usage, active_jobs,
                            gpu_demand, cpu_demand, mem_demand
                        )
                
                # Step 2: If still not found, ignore CPU/memory, only satisfy GPU
                if not found:
                    # Find server with sufficient GPU
                    free_gpus_dict = find_free_GPUs(gpu_df)
                    placement, found = self._synergy_gpu_only_placement(
                        job_info, free_gpus_dict, gpu_demand, node_info, 
                        server_resource_usage, active_jobs, jobs_to_terminate, gpu_df
                    )
                
                if found:
                    job_to_launch[job_id] = placement
                    # Update server resource usage
                    for gpu_id in placement:
                        node_id = gpu_df.loc[gpu_df["GPU_ID"] == gpu_id, "Node_ID"].iloc[0]
                        if node_id in server_resource_usage:
                            server_resource_usage[node_id]["gpu"] -= 1
                            server_resource_usage[node_id]["cpu"] -= cpu_demand / gpu_demand if gpu_demand > 0 else 0
                            server_resource_usage[node_id]["memory"] -= mem_demand / gpu_demand if gpu_demand > 0 else 0
                    mark_gpu_in_use(gpu_df, placement, job_id)
                else:
                    # Cannot place this job, skip remaining jobs
                    break
            
            return (jobs_to_terminate, job_to_launch)
            
        else:
            running_jobs = 0
            new_scheduled_jobs = 0
            jobs_to_schedule = 0
            for idx, job_id in enumerate(job_order):
                job_id, _ = job_id
                job = active_jobs[job_id]
                found = False
                if job["is_running"] == True:
                    # move to lower priority jobs
                    running_jobs += 1
                    continue
                if job["is_running"] == False:
                    # need to find placement only if job is not running
                    place_consolidated = (
                        job.get("placement_preference") == "consolidated"
                    )
                    # first checking if there are free GPUs
                    free_gpus = find_free_GPUs(gpu_df)
                    if place_consolidated:
                        placement, found = self._consolidated_placement(job, free_gpus)
                    else:
                        placement, found = self._scattered_placement(job, free_gpus)
                    # next checking if there are lower priority jobs which have
                    if not found:
                        # no free GPUs
                        # need to see if there are lower priority jobs which can be
                        # terminated and placement can be found then

                        for rev_idx in range(1, len(active_jobs) - idx):
                            potential_job_to_terminate = active_jobs[
                                job_order[-rev_idx][0]
                            ]
                            if potential_job_to_terminate["is_running"] == True:
                                # terminate this job
                                jobs_to_terminate.append(job_order[-rev_idx][0])
                                potential_job_to_terminate["is_running"] = False
                                # freeing up GPUs
                                delete_job_by_id(gpu_df, job_order[-rev_idx][0])
                                free_gpus = find_free_GPUs(gpu_df)
                                if place_consolidated:
                                    placement, found = self._consolidated_placement(
                                        job, free_gpus
                                    )
                                else:
                                    placement, found = self._scattered_placement(
                                        job, free_gpus
                                    )
                                if found:
                                    # we found an assignment
                                    # print(
                                    # f"Placed {job_id} by determining to terminate{job_order[-rev_idx][0]}"
                                    # )
                                    break
                if found:
                    new_scheduled_jobs += 1
                    job_to_launch[job_id] = placement
                    # update manual-pipeline-list for bert and gpt
                    mark_gpu_in_use(gpu_df, placement, job_id)
                else:
                    # print(f"New Jobs scheduled {new_scheduled_jobs}")
                    # print(f"Jobs previously running {running_jobs}")
                    # print(f"Jobs terminated {len(jobs_to_terminate)}")
                    # print(f"Jobs in queue {len(job_order)-idx}")
                    break
            return (jobs_to_terminate, job_to_launch)

    def _consolidated_placement(
        self, job_param: dict, free_gpus: dict
    ) -> Tuple[list, bool]:
        """
        Find a consolidated placement
        Args:
        job_param: Job Param configuration
        free_gpus: Dict of free GPUs {node_id: [list of GPU IDs']}
        Returns:
        list of GPU IDs on which to place the job
        boolean indicating if we found placement
        """
        # if there is a machine with exact required GPUs
        numGPUs_needed = job_param["num_GPUs"]
        for node in free_gpus:
            if len(free_gpus[node]) == numGPUs_needed:
                # found a perfect match
                return (free_gpus[node], True)
        # if we don't find an exact match find a node more GPUs
        # find the mode with min more GPUs then needed
        min_more_GPUs = 256  # random large enough number
        node_with_min_more_GPUs = None
        for node in free_gpus:
            if len(free_gpus[node]) >= numGPUs_needed:
                # found a node with more GPUs then needed
                if min_more_GPUs > len(free_gpus[node]):
                    min_more_GPUs = len(free_gpus[node])
                    node_with_min_more_GPUs = node
        if node_with_min_more_GPUs is not None:
            # only extracting the GPUs we need
            return (free_gpus[node_with_min_more_GPUs][:numGPUs_needed], True)
        # didn't find the requested number of GPUs
        return ([], False)

    def _scattered_placement(
        self, job_param: dict, free_gpus: dict
    ) -> Tuple[list, bool]:
        """
        Find placement without worrying about consolidation.
        Args:
        job_param: Job Param configuration
        free_gpus: Dict of free GPUs {node_id: [list of GPU IDs']}
        Returns:
        list of GPU IDs on which to place the job
        boolean indicating if we found placement
        """
        numGPUs_needed = job_param["num_GPUs"]
        gpus_for_job = list()
        found = False
        for node in free_gpus:
            gpus_for_job.extend(free_gpus[node][:numGPUs_needed])
            if len(gpus_for_job) == numGPUs_needed:
                found = True
                break
        if found:
            return (gpus_for_job, found)
        else:
            return ([], False)

    def _synergy_find_placement(
        self, job_info: dict, node_info: dict, gpu_df: pd.DataFrame,
        server_resource_usage: dict, active_jobs: dict,
        gpu_demand: int, cpu_demand: float, mem_demand: float
    ) -> Tuple[list, bool]:
        """
        Find placement with minimum fragmentation (Synergy-TUNE algorithm)
        Args:
            job_info: Job information dictionary
            node_info: Server information dictionary
            gpu_df: GPU dataframe
            server_resource_usage: Current server resource usage
            active_jobs: Active jobs dictionary
            gpu_demand: Job's GPU demand
            cpu_demand: Job's CPU demand
            mem_demand: Job's memory demand
        Returns:
            (list of GPU IDs, bool indicating if placement found)
        """
        free_gpus_dict = find_free_GPUs(gpu_df)
        
        # If single GPU job, find server with minimum free resources that can fit
        if gpu_demand == 1:
            best_server = None
            min_fragmentation = float('inf')
            
            for node_id in free_gpus_dict:
                if len(free_gpus_dict[node_id]) >= gpu_demand:
                    available = server_resource_usage.get(node_id, {})
                    if (available.get("gpu", 0) >= gpu_demand and
                        available.get("cpu", 0) >= cpu_demand and
                        available.get("memory", 0) >= mem_demand):
                        # Calculate fragmentation (waste)
                        fragmentation = (
                            (available["gpu"] - gpu_demand) +
                            (available["cpu"] - cpu_demand) / 100.0 +  # Normalize
                            (available["memory"] - mem_demand) / 100.0
                        )
                        if fragmentation < min_fragmentation:
                            min_fragmentation = fragmentation
                            best_server = node_id
            
            if best_server is not None:
                return ([free_gpus_dict[best_server][0]], True)
        
        # Multi-GPU job: find minimum set of servers
        else:
            # Try consolidated placement first
            for node_id in free_gpus_dict:
                if len(free_gpus_dict[node_id]) >= gpu_demand:
                    available = server_resource_usage.get(node_id, {})
                    if (available.get("gpu", 0) >= gpu_demand and
                        available.get("cpu", 0) >= cpu_demand and
                        available.get("memory", 0) >= mem_demand):
                        return (free_gpus_dict[node_id][:gpu_demand], True)
            
            # If consolidated placement not found, try scattered placement
            gpus_for_job = []
            remaining_gpus = gpu_demand
            servers_used = []
            
            # Sort servers by fragmentation (ascending)
            sorted_servers = sorted(
                free_gpus_dict.items(),
                key=lambda x: (
                    server_resource_usage.get(x[0], {}).get("gpu", 0),
                    server_resource_usage.get(x[0], {}).get("cpu", 0),
                    server_resource_usage.get(x[0], {}).get("memory", 0)
                )
            )
            
            for node_id, gpu_list in sorted_servers:
                if remaining_gpus <= 0:
                    break
                
                available = server_resource_usage.get(node_id, {})
                gpus_to_take = min(remaining_gpus, len(gpu_list))
                
                # Check if this server can contribute resources
                cpu_per_gpu = cpu_demand / gpu_demand if gpu_demand > 0 else 0
                mem_per_gpu = mem_demand / gpu_demand if gpu_demand > 0 else 0
                
                if (available.get("gpu", 0) >= gpus_to_take and
                    available.get("cpu", 0) >= cpu_per_gpu * gpus_to_take and
                    available.get("memory", 0) >= mem_per_gpu * gpus_to_take):
                    gpus_for_job.extend(gpu_list[:gpus_to_take])
                    remaining_gpus -= gpus_to_take
                    servers_used.append(node_id)
            
            if remaining_gpus == 0:
                return (gpus_for_job, True)
        
        return ([], False)

    def _synergy_gpu_only_placement(
        self, job_info: dict, free_gpus_dict: dict, gpu_demand: int,
        node_info: dict, server_resource_usage: dict, active_jobs: dict,
        jobs_to_terminate: list, gpu_df: pd.DataFrame
    ) -> Tuple[list, bool]:
        """
        Find placement ignoring CPU/memory, only satisfy GPU requirements.
        Then identify jobs to switch to GPU-proportional share.
        Args:
            job_info: Job information dictionary
            free_gpus_dict: Dictionary of free GPUs by node
            gpu_demand: Job's GPU demand
            node_info: Server information dictionary
            server_resource_usage: Current server resource usage
            active_jobs: Active jobs dictionary
            jobs_to_terminate: List of jobs to terminate (modified in place)
            gpu_df: GPU dataframe
        Returns:
            (list of GPU IDs, bool indicating if placement found)
        """
        # Find server with sufficient free GPUs
        for node_id in free_gpus_dict:
            if len(free_gpus_dict[node_id]) >= gpu_demand:
                # Found a server with enough GPUs
                placement = free_gpus_dict[node_id][:gpu_demand]
                
                # Identify jobs on this server that need to be switched to GPU-proportional
                # Get all running jobs on this server
                node_gpus = gpu_df.loc[gpu_df["Node_ID"] == node_id]
                running_job_ids = node_gpus.loc[
                    node_gpus["IN_USE"] == True
                ]["JOB_IDS"].unique()
                
                jobs_to_switch = []
                for running_job_id in running_job_ids:
                    if running_job_id is not None and running_job_id in active_jobs:
                        running_job = active_jobs[running_job_id]
                        # Check if this job has more than GPU-proportional share
                        # For simplicity, we'll terminate lower priority jobs
                        # In practice, should calculate actual proportional share
                        if running_job.get("job_priority", 0) < job_info.get("job_priority", 0):
                            jobs_to_switch.append(running_job_id)
                
                # If we found jobs to switch, terminate them
                if jobs_to_switch:
                    for switch_job_id in jobs_to_switch:
                        if switch_job_id not in jobs_to_terminate:
                            jobs_to_terminate.append(switch_job_id)
                            active_jobs[switch_job_id]["is_running"] = False
                            delete_job_by_id(gpu_df, switch_job_id)
                    
                    # Update free GPUs after termination
                    free_gpus_dict = find_free_GPUs(gpu_df)
                    # Retry placement
                    if node_id in free_gpus_dict and len(free_gpus_dict[node_id]) >= gpu_demand:
                        return (free_gpus_dict[node_id][:gpu_demand], True)
                else:
                    # No jobs to switch, but we have enough GPUs
                    return (placement, True)
        
        # Try scattered placement across multiple servers
        gpus_for_job = []
        remaining_gpus = gpu_demand
        
        for node_id in free_gpus_dict:
            if remaining_gpus <= 0:
                break
            gpus_to_take = min(remaining_gpus, len(free_gpus_dict[node_id]))
            gpus_for_job.extend(free_gpus_dict[node_id][:gpus_to_take])
            remaining_gpus -= gpus_to_take
        
        if remaining_gpus == 0:
            return (gpus_for_job, True)
        
        return ([], False)


# Gavel get job ids sorted by vals


def get_ids_sorted_by_priorities(priority_vals: dict) -> list:
    """
    Sorts the dict by value and return a sorted list in descending order of
    their priorities
    Args:
    priority_vals: key- job_id, vals- priority vals
    Returns:
    list of job ids sorted by their values
    """
    sorted_pairs = sorted(priority_vals.items(), key=lambda x: x[1], reverse=True)

    sorted_ids = [x for x, _ in sorted_pairs]
    return sorted_ids


# Pandas Utilities
def find_gpus_matching_JobID(job_id: int, gpu_df: pd.DataFrame) -> list:
    """
    Finds the GPU IDs which are running the given job id
    """
    return gpu_df.loc[gpu_df["JOB_IDS"] == job_id]["GPU_ID"].tolist()


# Find free GPUs


def find_free_GPUs(gpu_df: pd.DataFrame) -> dict:
    """
    Find the nodeID's which have free GPUs
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    Returns:
    dict: {Node_ID: [list of free GPUs]}
    """
    return (
        gpu_df.loc[gpu_df["IN_USE"] == False]
        .groupby("Node_ID")["GPU_ID"]
        .apply(list)
        .to_dict()
    )


def find_free_GPUs_by_type(gpu_df: pd.DataFrame, gpu_type: str) -> dict:
    """
    Find free nodeID's which have free GPUs of specific type

    Args:
    gpu_df : DataFrame consiting the information about GPUs
    Returns:
    dict : {Node_ID : [list of free GPUs]}
    """
    return (
        gpu_df.loc[(gpu_df["IN_USE"] == False) & (gpu_df["GPU_type"] == gpu_type)]
        .groupby("Node_ID")["GPU_ID"]
        .apply(list)
        .to_dict()
    )


# Mark a GPU in use


def mark_gpu_in_use(gpu_df: pd.DataFrame, gpu_id: List[int], job_id: int) -> None:
    """
    Find the GPU ID and mark it in use. After deciding to schedule something on
    it.
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    gpu_id : GPU to mark busy
    job_id: Job being scheduled on GPU with id=gpu_id

    Returns:
    None
    In place modifies the gpu_df
    """
    gpu_df.loc[gpu_df["GPU_ID"].isin(gpu_id), ["JOB_IDS", "IN_USE"]] = job_id, True
    return None


# Delete Job from data frame


def delete_job_by_id(gpu_df: pd.DataFrame, job_id: int) -> None:
    """
    Finds the job ID provided. Marks those jobs free and marks the GPU free to
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    job_id : Job to delete

    Returns:
    None
    In place modifies the gpu_df
    """
    gpu_df.loc[gpu_df["JOB_IDS"] == job_id, ["JOB_IDS", "IN_USE"]] = None, False
    return None


def find_num_free_GPUs(gpu_df: pd.DataFrame) -> int:
    """
    Find the number of free GPU's
    Args:
    gpu_df : DataFrame consisting of information about GPUs
    Returns:
    int : Number of free GPUs
    """
    return len(gpu_df.loc[gpu_df["IN_USE"] == False])


def get_server_available_resources(
    node_id: int, node_info: dict, gpu_df: pd.DataFrame, active_jobs: dict
) -> dict:
    """
    Calculate available resources on a server
    Args:
        node_id: Server node ID
        node_info: Server information dict
        gpu_df: GPU dataframe
        active_jobs: Active jobs dictionary
    Returns:
        dict with keys: 'gpu', 'cpu', 'memory'
    """
    # Get server capacity
    server_capacity = {
        "gpu": node_info.get("numGPUs", 0),
        "cpu": node_info.get("numCPUcores", 0),
        "memory": node_info.get("memoryCapacity", 0),
    }
    
    # Calculate used resources
    used_resources = {"gpu": 0, "cpu": 0, "memory": 0}
    
    # Count used GPUs on this node
    node_gpus = gpu_df.loc[gpu_df["Node_ID"] == node_id]
    used_resources["gpu"] = len(node_gpus.loc[node_gpus["IN_USE"] == True])
    
    # Calculate used CPU and memory from running jobs on this node
    running_job_ids = node_gpus.loc[node_gpus["IN_USE"] == True]["JOB_IDS"].unique()
    for job_id in running_job_ids:
        if job_id is not None and job_id in active_jobs:
            job = active_jobs[job_id]
            # Get allocated resources for this job on this node
            # Note: This is a simplified calculation
            # In reality, we need to track per-node resource allocation
            if job.get("is_running", False):
                # Estimate based on job's GPU allocation on this node
                job_gpus_on_node = len(node_gpus.loc[
                    (node_gpus["IN_USE"] == True) & (node_gpus["JOB_IDS"] == job_id)
                ])
                if job_gpus_on_node > 0:
                    # Proportional allocation
                    total_job_gpus = job.get("num_GPUs", 1)
                    if total_job_gpus > 0:
                        ratio = job_gpus_on_node / total_job_gpus
                        used_resources["cpu"] += job.get("job_cpu_demand", 0) * ratio
                        used_resources["memory"] += job.get("job_mem_demand", 0) * ratio
    
    # Calculate available resources
    available = {
        "gpu": max(0, server_capacity["gpu"] - used_resources["gpu"]),
        "cpu": max(0, server_capacity["cpu"] - used_resources["cpu"]),
        "memory": max(0, server_capacity["memory"] - used_resources["memory"]),
    }
    
    return available


def calculate_gpu_proportional_allocation(
    job_gpu_demand: int, total_gpu_demand: int, total_gpu_capacity: int
) -> dict:
    """
    Calculate GPU-proportional resource allocation for a job
    Args:
        job_gpu_demand: Job's GPU demand
        total_gpu_demand: Total GPU demand of all jobs
        total_gpu_capacity: Total GPU capacity in cluster
    Returns:
        dict with proportional CPU and memory allocation
    """
    if total_gpu_demand == 0 or total_gpu_capacity == 0:
        return {"cpu": 0, "memory": 0}
    
    # GPU proportional share
    gpu_share = min(job_gpu_demand / total_gpu_demand, job_gpu_demand / total_gpu_capacity)
    
    # For simplicity, assume CPU and memory are proportional to GPU
    # In practice, this should be calculated based on actual resource ratios
    return {
        "cpu": gpu_share * 100,  # Simplified: assume 100 CPU per GPU share
        "memory": gpu_share * 100,  # Simplified: assume 100 memory per GPU share
    }
