import pandas as pd
import copy
import heapq
from typing import Tuple, List, Dict, Any
import os
from blox.deployment.grpc_client_rm import get_tput_from_job_dict
from .All_placement import(
        Srtf_placement,
        Synergy_fifo_placement,
        Fifo_placement
)
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
                cluster_state=cluster_state,  # Pass cluster_state for server_resource_usage
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
        cluster_state=None,
        **kwargs,
    ) -> dict:
        """
        parses the sorted_jobs dictionary and calls relevant placement policy

        # CAUTION: This function makes in place changes to active jobs and
        # gpu_df

        """
        job_order = new_job_schedule["job_order"]
        scheduler = os.environ["sched_policy"]
        jobs_to_terminate = list()
        job_to_launch = dict()  # Initialize job_to_launch dictionary
        launched_job_ids = list()
        # go over jobs in job order
        print("scheduler",scheduler)
        #Borrowed from https://github.com/msr-fiddle/synergy/blob/master/simulator/resources/cluster.py#L581
        if scheduler == "Synergy_fifo":
            return Synergy_fifo_placement(new_job_schedule, cluster_state, node_info, gpu_df, active_jobs)
        elif scheduler == "Srtf":
           return Srtf_placement(new_job_schedule, cluster_state, node_info, gpu_df, active_jobs)
        ##todo: add Synergy_srtf placement
        elif scheduler == "Fifo" :
            return Fifo_placement(new_job_schedule, cluster_state, node_info, gpu_df, active_jobs)
    def _gpu_normalized_vector(self, vector: list) -> list:
        """
        Normalize demand vector by GPU (first element).
        Args:
            vector: Demand vector [gpu, cpu, mem, sspeed, ...]
        Returns:
            Normalized vector (per-GPU)
        """
        return [item / vector[0] for item in vector]



    def _vector_to_map(self, demand_vec: list) -> dict:
        """
        Convert demand vector to resource map format.
        Args:
            demand_vec: Demand vector [gpu, cpu, mem, sspeed, 0]
        Returns:
            Dictionary with resource allocations
        """
        return {
            "gpu": int(demand_vec[0]),
            "cpu": int(demand_vec[1]),
            "mem": float(demand_vec[2]),
            "sspeed": float(demand_vec[3]) if len(demand_vec) > 3 else 0.0
        }

   

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
        numGPUs_needed = job_param.get("job_gpu_demand", job_param.get("num_GPUs", 0))
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
        numGPUs_needed = job_param.get("job_gpu_demand", job_param.get("num_GPUs", 0))
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

# Pandas Utilities

def find_gpus_matching_JobID(job_id: int, gpu_df: pd.DataFrame) -> list:
    """
    Finds the GPU IDs which are running the given job id
    """
    return gpu_df.loc[gpu_df["JOB_IDS"] == job_id]["GPU_ID"].tolist()


# Find free GPUs

# Mark a GPU in use
def mark_gpu_in_use(
    gpu_df: pd.DataFrame, 
    gpu_id: List[int], 
    job_id: int,
    res_map: dict = None,
    server_resource_usage: dict = None
) -> None:
    """
    Find the GPU ID and mark it in use. After deciding to schedule something on it.
    Optionally update server resource usage if res_map is provided (similar to _server.allocate(res_map[_server])).
    
    Args:
        gpu_df: DataFrame consisting of information about GPUs
        gpu_id: GPU IDs to mark busy
    job_id: Job being scheduled on GPU with id=gpu_id
        res_map: Optional resource map {ServerWrapper: {'cpu': int, 'mem': float, 'gpu': int, 'sspeed': float}}
                 If provided, will update server_resource_usage similar to _server.allocate(res_map[_server])
        server_resource_usage: Optional dictionary to update with resource usage
                              Format: {node_id: {"gpu": int, "cpu": float, "memory": float}}

    Returns:
    None
    In place modifies the gpu_df and optionally server_resource_usage
    """
    # Mark GPUs as in use
    gpu_df.loc[gpu_df["GPU_ID"].isin(gpu_id), ["JOB_IDS", "IN_USE"]] = job_id, True
    
    # Update server resource usage if res_map is provided (similar to _server.allocate(res_map[_server]))
    if res_map is not None and server_resource_usage is not None:
        for server_key, resources in res_map.items():
            # Extract node_id from ServerWrapper or dict key
            if hasattr(server_key, 'node_id'):
                node_id = server_key.node_id
            elif hasattr(server_key, 'server_id'):
                node_id = server_key.server_id
            elif isinstance(server_key, int):
                node_id = server_key
            else:
                continue
            
            if node_id in server_resource_usage:
                # Update server resource usage (similar to _server.allocate(res_map[_server]))
                # Subtract allocated resources from available resources
                cpu_allocated = resources.get("cpu_allocated", 0)
                mem_allocated = resources.get("mem_allocated", 0)
                gpu_allocated = resources.get("gpu_allocated", 0)
                server_resource_usage[node_id]["gpu"] = server_resource_usage[node_id].get("gpu", 0) - gpu_allocated
                server_resource_usage[node_id]["cpu"] =  server_resource_usage[node_id].get("cpu", 0) - cpu_allocated
                server_resource_usage[node_id]["memory"] =  server_resource_usage[node_id].get("memory", 0) - mem_allocated
    
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



def calculate_gpu_proportional_allocation(
    job_gpu_demand_in_server: int
) -> dict:
    """
    Calculate GPU-proportional resource allocation for a job
    Args:
        job_gpu_demand_in_server: Job's GPU demand
    Returns:
        dict with proportional CPU and memory allocation
    """
    # GPU proportional share
    return {
        "cpu": job_gpu_demand_in_server * 3,  
        "memory": job_gpu_demand_in_server * 62.5,  
    }
