 # Initialize server resource usage for SRTF scheduler
import pandas as pd
from typing import Tuple, List, Dict, Any
import heapq
import copy


def Fifo_placement(new_job_schedule: dict, cluster_state: dict, node_info: dict, gpu_df: pd.DataFrame, active_jobs: dict) -> Tuple[List[int], Dict]:
    # Initialize server resource usage for other schedulers
    server_resource_usage = {}
    jobs_to_terminate = list()
    job_to_launch = dict()
    jobs_this_round = new_job_schedule["job_order"]
    if cluster_state and hasattr(cluster_state, 'server_resource_usage'):
        # Use cluster_state.server_resource_usage as base (updated in update_metrics)
        for node_id in node_info:
            # Get capacity from node_info, fallback to cluster_state.server_map if not available
            cpu_capacity = node_info[node_id].get("numCPUcores", 0)
            mem_capacity = node_info[node_id].get("memoryCapacity", 0)
            gpu_capacity = node_info[node_id].get("numGPUs", 0)
            # If capacity is missing, try to get from cluster_state.server_map
            # Get used resources from cluster_state (updated in update_metrics)
            cpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("cpu_used", 0)
            mem_used = cluster_state.server_resource_usage.get(node_id, {}).get("memory_used", 0)
            gpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("gpu_used", 0)
            # print("cpu_used",cpu_used)
            # Calculate available resources (consistent with CPU and memory calculation)
            available_cpu = max(0, cpu_capacity - cpu_used)
            available_mem = max(0, mem_capacity - mem_used)
            available_gpu = max(0, gpu_capacity - gpu_used)
            server_resource_usage[node_id] = {
                "gpu": available_gpu,
                "cpu": available_cpu,
                "memory": available_mem
            }
    else:
        # Fallback: calculate from scratch if cluster_state not available
        for node_id in node_info:
            server_resource_usage[node_id] = get_server_available_resources(
                node_id, node_info[node_id], gpu_df, active_jobs
            )
    for idx, job_tuple in enumerate(jobs_this_round):
        job_id, job_info = job_tuple
        job = active_jobs[job_id]
        
        
        if job.get("is_running", False):
            continue  # Skip already running jobs

        # Use actual allocated GPUs from gpu_df, not just the num_GPUs field
        # This ensures we check the real state, not just a cached value
        # Get GPUs allocated to this job and calculate total across all nodes
        allocated_gpus_in_df = find_gpus_matching_JobID(job_id, gpu_df)
        # Count GPUs per node and sum them up to get total allocated GPUs
        if allocated_gpus_in_df:
            # Group GPUs by node and count per node, then sum
            job_gpus_df = gpu_df.loc[gpu_df["GPU_ID"].isin(allocated_gpus_in_df)]
            gpu_count_per_node = job_gpus_df.groupby("Node_ID").size()
            actual_allocated_gpus = gpu_count_per_node.sum()
        else:
            actual_allocated_gpus = 0
        
        job_gpu_deficit = job_info.get("job_gpu_demand") - job.get("num_GPUs_allocated", 0)
        
        
        # If job already has all needed GPUs actually allocated, skip placement
        if job_gpu_deficit <= 0 and actual_allocated_gpus > 0:
            continue
        
        # If job_gpu_deficit is 0 or negative, cannot place
        if job_gpu_deficit <= 0:
            continue
        
        # Calculate CPU and memory deficit
        # Calculate total demand based on GPU deficit
        total_cpu_demand = job_gpu_deficit * 3
        total_mem_demand = job_gpu_deficit * 62.5
        total_sspeed_demand = 0
        
        # Calculate deficit (what's still needed)
        cpu_deficit = max(0, total_cpu_demand - job.get("cpus_allocated", 0))
        mem_deficit = max(0, total_mem_demand - job.get("mem_allocated", 0))
        sspeed_deficit = max(0, total_sspeed_demand - job.get("sspeed_allocated", 0))
        
        demand_vec = [
            job_gpu_deficit,
            cpu_deficit,
            mem_deficit,
            sspeed_deficit
        ]
        
        total_available_gpu = sum(s.get('gpu', 0) for s in server_resource_usage.values())
        total_available_cpu = sum(s.get('cpu', 0) for s in server_resource_usage.values())
        total_available_mem = sum(s.get('memory', 0) for s in server_resource_usage.values())
        free_gpus_dict = find_free_GPUs(gpu_df)
        total_free_gpus = sum(len(gpus) for gpus in free_gpus_dict.values())
        
        placement, found, res_map, updated_server_resource_usage = _synergy_find_placement(
            job_info, node_info, gpu_df, server_resource_usage, active_jobs,
            demand_vec[0], demand_vec[1], demand_vec[2]
        )
        

        # Update server_resource_usage if provided
        if found and updated_server_resource_usage:
            server_resource_usage = updated_server_resource_usage
        
        if found:
            # Validate placement: must have at least one GPU
            
            # Group GPUs by node for better organization
            node_gpu_map = group_gpus_by_node(placement, gpu_df)
            # Store placement with node information
            # Format: {job_id: {'nodes': {node_id: [gpu_ids]}, 'gpus': [all_gpu_ids]}}
            job_to_launch[job_id] = {
                'nodes': node_gpu_map,  # Node-level placement: {node_id: [gpu_ids]}
                'gpus': placement       # All GPU IDs (for backward compatibility)
            }
            
            # Store res_map in active_jobs for later use by job.allocate()
            active_jobs[job_id]["res_map"] = res_map
            
            # Update job state with allocated resources (critical for metrics calculation)
            actual_gpu_count = len(placement) if placement else 0
            active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
            active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
            
            # Update tput based on allocated CPU and memory
            allocated_cpus = active_jobs[job_id]["cpus_allocated"]
            allocated_mem = active_jobs[job_id]["mem_allocated"]
            # Note: Do NOT update cluster_state.server_resource_usage here
            # Resource usage is updated in blox_manager.exec_jobs() via _update_server_resource_usage()
            # Updating here would cause duplicate counting since exec_jobs() also updates it
        elif not found:
            # for rev_idx in range(1, len(active_jobs) - idx):
            #     potential_job_to_terminate = active_jobs[
            #         jobs_this_round[-rev_idx][0]
            #     ]
            #     if potential_job_to_terminate["is_running"] == True:
            #         # terminate this job
            #         terminated_job_id = jobs_this_round[-rev_idx][0]
            #         jobs_to_terminate.append(terminated_job_id)
            #         potential_job_to_terminate["is_running"] = False
            #         # freeing up GPUs
            #         delete_job_by_id(gpu_df, terminated_job_id)
            #         free_gpus = find_free_GPUs(gpu_df)
                
            #         placement, found, res_map, updated_server_resource_usage = _synergy_find_placement(
            #             job_info, node_info, gpu_df, server_resource_usage, active_jobs,
            #             demand_vec[0], demand_vec[1], demand_vec[2]
            #         )
                    
            #         # Update server_resource_usage if provided
            #         if found and updated_server_resource_usage:
            #             server_resource_usage = updated_server_resource_usage
            #         if found:
            #             # Validate placement: must have at least one GPU
            #             if not placement or len(placement) == 0:
            #                 continue  # Try next lower priority job
                        
            #             # Group GPUs by node for better organization
            #             node_gpu_map = group_gpus_by_node(placement, gpu_df)
                        
            #             # Store placement with node information
            #             # Format: {job_id: {'nodes': {node_id: [gpu_ids]}, 'gpus': [all_gpu_ids]}}
            #             job_to_launch[job_id] = {
            #                 'nodes': node_gpu_map,  # Node-level placement: {node_id: [gpu_ids]}
            #                 'gpus': placement       # All GPU IDs (for backward compatibility)
            #             }
                        
            #             # Store res_map in active_jobs for later use by job.allocate()
            #             active_jobs[job_id]["res_map"] = res_map
                        
            #             # Update job state with allocated resources (critical for metrics calculation)
            #             actual_gpu_count = len(placement) if placement else 0
            #             active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
            #             active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
            #             active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
            #             active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
                        
            #             break  # Successfully placed, exit the termination loop
            #         else:
                        continue
    return jobs_to_terminate, job_to_launch


def Synergy_fifo_placement(new_job_schedule: dict, cluster_state: dict, node_info: dict, gpu_df: pd.DataFrame, active_jobs: dict) -> Tuple[List[int], Dict]:    
    jobs_to_terminate = list()
    job_to_launch = dict()  # Initialize job_to_launch dictionary
    jobs_this_round = new_job_schedule["jobs_this_round"]
    # Calculate total GPU demand and capacity for proportional allocation
    # Initialize server resource usage from cluster_state
    # cluster_state.server_resource_usage is updated in update_metrics() based on running jobs' res_map
    server_resource_usage = {}
    if cluster_state and hasattr(cluster_state, 'server_resource_usage'):
        # Use cluster_state.server_resource_usage as base (updated in update_metrics)
        for node_id in node_info:
            # Get capacity from node_info
            cpu_capacity = node_info[node_id].get("numCPUcores", 0)
            mem_capacity = node_info[node_id].get("memoryCapacity", 0)
            gpu_capacity = node_info[node_id].get("numGPUs", 0)
            # Get used resources from cluster_state (updated in update_metrics)
            cpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("cpu_used", 0)
            mem_used = cluster_state.server_resource_usage.get(node_id, {}).get("memory_used", 0)
            gpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("gpu_used", 0)
            # print("gpu_used",gpu_used)
            # Calculate available resources (consistent with CPU and memory calculation)
            server_resource_usage[node_id] = {
                "gpu": max(0, gpu_capacity - gpu_used),
                "cpu": max(0, cpu_capacity - cpu_used),
                "memory": max(0, mem_capacity - mem_used)
            }
    else:
        # Fallback: calculate from scratch if cluster_state not available
        for node_id in node_info:
            server_resource_usage[node_id] = get_server_available_resources(
                node_id, node_info[node_id], gpu_df, active_jobs
            )
    # Process each job in order using Synergy-TUNE algorithm
    # print(f"[DEBUG] Synergy_fifo: jobs_this_round has {len(jobs_this_round)} jobs")
    for idx, job_tuple in enumerate(jobs_this_round):
        job_id, job_info = job_tuple
        job = active_jobs[job_id]
        
        # print(f"[DEBUG] Synergy_fifo: Processing job {job_id}, is_running={job.get('is_running', False)}")
        
        if job.get("is_running", False):
            # print(f"[DEBUG] Synergy_fifo: Job {job_id} is already running, skipping")
            continue  # Skip already running jobs
        
        # Get job demand vector [gpu_deficit, cpu_deficit, mem_deficit, sspeed_deficit]
        gpu_demand = job_info.get("job_gpu_demand", 0)
        cpu_demand = job_info.get("job_cpu_demand", job_info.get("job_cpu_demand_orig", 0))
        mem_demand = job_info.get("job_mem_demand", job_info.get("job_mem_demand_orig", 0))
        sspeed_demand = job_info.get("job_sspeed_demand", job_info.get("job_sspeed_demand_orig", 0))
        
        # Calculate GPU deficit (for jobs not yet running, deficit equals demand)
        job_gpu_deficit = gpu_demand - job.get("num_GPUs_allocated", 0)
        
        # print(f"[DEBUG] Synergy_fifo: Job {job_id} - gpu_demand={gpu_demand}, num_GPUs_allocated={job.get('num_GPUs_allocated', 0)}, job_gpu_deficit={job_gpu_deficit}")
        
        # If job already has all needed GPUs and resources, skip placement
        if job_gpu_deficit == 0:
            # print(f"[DEBUG] Synergy_fifo: Job {job_id} has no GPU deficit, skipping")
            continue
        
        demand_vec = [
            job_gpu_deficit,
            cpu_demand - job.get("cpus_allocated", 0),
            mem_demand - job.get("mem_allocated", 0),
            sspeed_demand - job.get("sspeed_allocated", 0)
        ]

        # Store demand values for potential retry after timeout
        gpu_d = int(demand_vec[0])
        cpu_d = float(demand_vec[1])
        mem_d = float(demand_vec[2])
        
        # Create a partial function wrapper for _call_allocate with job-specific parameters
        def _call_allocate_wrapper(available_gpus_dict, gpu_deficit, job_info_dict, time=None, fair=False, demand_vec=None):
            """Wrapper function that calls the standalone _call_allocate with job-specific parameters"""
            return _call_allocate(
                available_gpus_dict, gpu_deficit, job_info_dict, job_id,
                node_info, gpu_df, server_resource_usage, active_jobs,
                job_to_launch, time, fair, demand_vec
            )
        
        # Call _tune with initial=True to start the tuning process
        # print(f"[DEBUG] Synergy_fifo: Calling _tune for job {job_id}")
        placement, found, res_map = _tune(
            job_info, demand_vec, job_gpu_deficit,
            peer_adjust=False, initial=True, final=False,
            _call_allocate=_call_allocate_wrapper,
            available_gpus=find_free_GPUs(gpu_df),
            time=0.0,  # Time should be passed from caller
            fair=True,
            node_info=node_info,
            gpu_df=gpu_df,
            server_resource_usage=server_resource_usage,
            active_jobs=active_jobs,
            jobs_to_terminate=jobs_to_terminate,
            job_id=job_id,
            job_to_launch=job_to_launch
        )
        # print(f"[DEBUG] Synergy_fifo: Job {job_id} - found={found}, placement length={len(placement) if placement else 0}")

        if found:
            # Validate placement: must have at least one GPU
            # Group GPUs by node for better organization
            node_gpu_map = group_gpus_by_node(placement, gpu_df)
            
            # Store placement with node information
            # Format: {job_id: {'nodes': {node_id: [gpu_ids]}, 'gpus': [all_gpu_ids]}}
            # This provides both node-level and GPU-level information
            job_to_launch[job_id] = {
                'nodes': node_gpu_map,  # Node-level placement: {node_id: [gpu_ids]}
                'gpus': placement       # All GPU IDs (for backward compatibility)
            }
            # Store res_map in active_jobs for later use by job.allocate()
            active_jobs[job_id]["res_map"] = res_map
            # print("placement",placement)
            # Update job state with allocated resources (critical for metrics calculation)
            actual_gpu_count = len(placement) if placement else 0
            active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
            active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
            # Output CPU and memory allocation information
            total_cpu = active_jobs[job_id]["cpus_allocated"]
            total_mem = active_jobs[job_id]["mem_allocated"]
            # print(f"Synergy_FIFO: Job {job_id} allocated - Total: GPU={actual_gpu_count}, CPU={total_cpu}, Memory={total_mem:.2f}GB")
            if res_map:
                for server, resources in res_map.items():
                    node_id = server.server_id if hasattr(server, 'server_id') else server
                    # print(f"  Node {node_id}: GPU={resources.get('gpu', 0)}, CPU={resources.get('cpu', 0)}, Memory={resources.get('mem', 0):.2f}GB")
        else:
            # Placement not found - log why
            # print(f"[PLACEMENT_FAILED] Job {job_id}: Cannot find placement - GPU demand: {gpu_d}, CPU demand: {cpu_d:.2f}, Memory demand: {mem_d:.2f}")
            # Print available resources for debugging
            total_available_gpu = sum(server_resource_usage[node_id].get('gpu', 0) for node_id in server_resource_usage)
            total_available_cpu = sum(server_resource_usage[node_id].get('cpu', 0) for node_id in server_resource_usage)
            total_available_mem = sum(server_resource_usage[node_id].get('memory', 0) for node_id in server_resource_usage)
            # print(f"  Total available resources: GPU={total_available_gpu}, CPU={total_available_cpu:.2f}, Memory={total_available_mem:.2f}")
            for node_id in server_resource_usage:
                available = server_resource_usage[node_id]
                # print(f"    Node {node_id}: GPU={available.get('gpu', 0)}, CPU={available.get('cpu', 0):.2f}, Memory={available.get('memory', 0):.2f}")
        continue
            
    return (jobs_to_terminate, job_to_launch)

def Srtf_placement(new_job_schedule: dict, cluster_state: dict, node_info: dict, gpu_df: pd.DataFrame, active_jobs: dict) -> Tuple[List[int], Dict]:
    """
    SRTF 调度器的放置方法
    
    Args:
        new_job_schedule: 新的作业调度计划
        cluster_state: 集群状态对象
        node_info: 节点信息字典
        gpu_df: GPU 数据框
        active_jobs: 活跃作业字典
    
    Returns:
        (jobs_to_terminate, job_to_launch) 元组
    """
    # 延迟导入以避免循环导入

    
    server_resource_usage = {}
    jobs_to_terminate = list()
    job_to_launch = dict()
    jobs_this_round = new_job_schedule["job_order"]
    
    if cluster_state and hasattr(cluster_state, 'server_resource_usage'):
        # Use cluster_state.server_resource_usage as base (updated in update_metrics)
        for node_id in node_info:
            cpu_capacity = node_info[node_id].get("numCPUcores", 0)
            mem_capacity = node_info[node_id].get("memoryCapacity", 0)
            gpu_capacity = node_info[node_id].get("numGPUs", 0)
            cpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("cpu_used", 0)
            mem_used = cluster_state.server_resource_usage.get(node_id, {}).get("memory_used", 0)
            gpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("gpu_used", 0)
            available_cpu = max(0, cpu_capacity - cpu_used)
            available_mem = max(0, mem_capacity - mem_used)
            available_gpu = max(0, gpu_capacity - gpu_used)
            server_resource_usage[node_id] = {
                "gpu": available_gpu,
                "cpu": available_cpu,
                "memory": available_mem
            }
    else:
        # Fallback: calculate from scratch if cluster_state not available
        for node_id in node_info:
            server_resource_usage[node_id] = get_server_available_resources(
                node_id, node_info[node_id], gpu_df, active_jobs
            )
    
    # Get free GPUs count for checking availability
    free_gpus_dict = find_free_GPUs(gpu_df)
    total_free_gpus = sum(len(gpus) for gpus in free_gpus_dict.values())
    
    # Process jobs in sorted order (SRTF: shortest remaining time first)
    for idx, job_tuple in enumerate(jobs_this_round):
                job_id, job_info = job_tuple
                job = active_jobs[job_id]
                # For SRTF: Skip jobs that are already running and have sufficient resources
                # Only skip if the job is running AND has all needed GPUs allocated
                # if job.get("is_running") == True:
                #     allocated_gpus_in_df = find_gpus_matching_JobID(job_id, gpu_df)
                #     if allocated_gpus_in_df:
                #         job_gpus_df = gpu_df.loc[gpu_df["GPU_ID"].isin(allocated_gpus_in_df)]
                #         gpu_count_per_node = job_gpus_df.groupby("Node_ID").size()
                #         actual_allocated_gpus = gpu_count_per_node.sum()
                #     else:
                #         actual_allocated_gpus = 0
                    
                #     job_gpu_demand = job_info.get("job_gpu_demand")
                #     # Only skip if job has all needed GPUs and is running
                #     if actual_allocated_gpus >= job_gpu_demand and job_gpu_demand > 0:
                #         continue
                
                # # Use actual allocated GPUs from gpu_df
                # allocated_gpus_in_df = find_gpus_matching_JobID(job_id, gpu_df)
                # if allocated_gpus_in_df:
                #     job_gpus_df = gpu_df.loc[gpu_df["GPU_ID"].isin(allocated_gpus_in_df)]
                #     gpu_count_per_node = job_gpus_df.groupby("Node_ID").size()
                #     actual_allocated_gpus = gpu_count_per_node.sum()
                # else:
                #     actual_allocated_gpus = 0
                
                job_gpu_deficit = job_info.get("job_gpu_demand", 0) - job.get("num_GPUs_allocated", 0)
                
                # Only try to place if there's a GPU deficit and enough free GPUs available
                if job_gpu_deficit > 0 and total_free_gpus >= job_gpu_deficit and job.get("is_running") == False:
                    # Calculate CPU and memory deficit (similar to FIFO)
                    total_cpu_demand = job_gpu_deficit * 3
                    total_mem_demand = job_gpu_deficit * 62.5
                    total_sspeed_demand = 0
                    
                    cpu_deficit = max(0, total_cpu_demand - job.get("cpus_allocated", 0))
                    mem_deficit = max(0, total_mem_demand - job.get("mem_allocated", 0))
                    sspeed_deficit = max(0, total_sspeed_demand - job.get("sspeed_allocated", 0))
                    demand_vec = [
                        job_gpu_deficit,
                        cpu_deficit,
                        mem_deficit,
                        sspeed_deficit
                    ]
                    
                    # Try to find placement
                    placement, found, res_map, updated_server_resource_usage = _synergy_find_placement(
                        job_info, node_info, gpu_df, server_resource_usage, active_jobs,
                        demand_vec[0], demand_vec[1], demand_vec[2]
                    )
                    
                    # Update server_resource_usage if provided
                    if found and updated_server_resource_usage:
                        server_resource_usage = updated_server_resource_usage
                    
                    if found:
                        # Group GPUs by node for better organization
                        node_gpu_map = group_gpus_by_node(placement, gpu_df)
                        # Store placement with node information
                        job_to_launch[job_id] = {
                            'nodes': node_gpu_map,
                            'gpus': placement
                        }
                        
                        # Store res_map in active_jobs for later use
                        active_jobs[job_id]["res_map"] = res_map
                        
                        # Update job state with allocated resources
                        actual_gpu_count = len(placement) if placement else 0
                        active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
                        active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
                        active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
                        active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
                        
                        # Update free GPUs count after successful allocation
                        free_gpus_dict = find_free_GPUs(gpu_df)
                        total_free_gpus = sum(len(gpus) for gpus in free_gpus_dict.values())
                    elif not found:
                        # If placement not found, try to preempt running jobs from the end of job_order
                        # Start from the end of jobs_this_round and work backwards (skip current job)
                        # for rev_idx in range(1, len(active_jobs) - idx):
                        #     potential_job_tuple = jobs_this_round[-rev_idx]
                        #     potential_job_id = potential_job_tuple[0]
                        #     potential_job_to_terminate = active_jobs[potential_job_id]
                            
                        #     # Skip if this job is already marked for termination
                        #     if potential_job_id in jobs_to_terminate:
                        #         continue
                            
                        #     # Only terminate running jobs
                        #     if potential_job_to_terminate.get("is_running"):
                        #         # Terminate this job
                        #         jobs_to_terminate.append(potential_job_id)
                        #         potential_job_to_terminate["is_running"] = False
                        #         # Free up GPUs
                        #         delete_job_by_id(gpu_df, potential_job_id)
                                
                        #         # Update server_resource_usage
                        #         if potential_job_to_terminate.get("res_map"):
                        #             free_node_resource_usage(potential_job_to_terminate["res_map"], server_resource_usage)
                                
                        #         # Recalculate server_resource_usage
                        #         for node_id in node_info:
                        #             server_resource_usage[node_id] = get_server_available_resources(
                        #                 node_id, node_info[node_id], gpu_df, active_jobs
                        #             )
                                
                        #         # Try to place after preemption
                        #         placement, found, res_map, updated_server_resource_usage = _synergy_find_placement(
                        #             job_info, node_info, gpu_df, server_resource_usage, active_jobs,
                        #             demand_vec[0], demand_vec[1], demand_vec[2]
                        #         )
                                
                        #         # Update server_resource_usage if provided
                        #         if found and updated_server_resource_usage:
                        #             server_resource_usage = updated_server_resource_usage
                                
                        #         if found:
                        #             # We found an assignment
                        #             # Group GPUs by node
                        #             node_gpu_map = group_gpus_by_node(placement, gpu_df)
                        #             job_to_launch[job_id] = {
                        #                 'nodes': node_gpu_map,
                        #                 'gpus': placement
                        #             }
                                    
                        #             # Store res_map
                        #             active_jobs[job_id]["res_map"] = res_map
                                    
                        #             # Update job state with allocated resources
                        #             actual_gpu_count = len(placement) if placement else 0
                        #             active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
                        #             active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
                        #             active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
                        #             active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
                                    
                        #             # Update free GPUs count after successful allocation
                        #             free_gpus_dict = find_free_GPUs(gpu_df)
                        #             total_free_gpus = sum(len(gpus) for gpus in free_gpus_dict.values())
                                    
                        #             break  # Successfully placed, exit preemption loop
                        #         else:
                                    continue
                else:
                    # If job_gpu_deficit <= 0 or not enough free GPUs, skip this job
                    continue
    return (jobs_to_terminate, job_to_launch)


def Synergy_srtf_placement(new_job_schedule: dict, cluster_state: dict, node_info: dict, gpu_df: pd.DataFrame, active_jobs: dict) -> Tuple[List[int], Dict]:
    """
    Synergy-SRTF 放置方法：使用 Synergy 的资源分配逻辑，但基于 SRTF 的任务列表
    
    这个函数结合了：
    - SRTF 调度器的任务排序（job_order）
    - Synergy_fifo 的资源分配算法（_tune）
    
    Args:
        new_job_schedule: 新的作业调度计划（包含 job_order，由 SRTF 调度器生成）
        cluster_state: 集群状态对象
        node_info: 节点信息字典
        gpu_df: GPU 数据框
        active_jobs: 活跃作业字典
    
    Returns:
        (jobs_to_terminate, job_to_launch) 元组
    """
    jobs_to_terminate = list()
    job_to_launch = dict()  # Initialize job_to_launch dictionary
    # 使用 SRTF 的任务列表（job_order），而不是 jobs_this_round
    jobs_this_round = new_job_schedule["job_order"]
    
    # Initialize server resource usage from cluster_state
    # cluster_state.server_resource_usage is updated in update_metrics() based on running jobs' res_map
    server_resource_usage = {}
    if cluster_state and hasattr(cluster_state, 'server_resource_usage'):
        # Use cluster_state.server_resource_usage as base (updated in update_metrics)
        for node_id in node_info:
            # Get capacity from node_info
            cpu_capacity = node_info[node_id].get("numCPUcores", 0)
            mem_capacity = node_info[node_id].get("memoryCapacity", 0)
            gpu_capacity = node_info[node_id].get("numGPUs", 0)
            # Get used resources from cluster_state (updated in update_metrics)
            cpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("cpu_used", 0)
            mem_used = cluster_state.server_resource_usage.get(node_id, {}).get("memory_used", 0)
            gpu_used = cluster_state.server_resource_usage.get(node_id, {}).get("gpu_used", 0)
            # Calculate available resources (consistent with CPU and memory calculation)
            server_resource_usage[node_id] = {
                "gpu": max(0, gpu_capacity - gpu_used),
                "cpu": max(0, cpu_capacity - cpu_used),
                "memory": max(0, mem_capacity - mem_used)
            }
    else:
        # Fallback: calculate from scratch if cluster_state not available
        for node_id in node_info:
            server_resource_usage[node_id] = get_server_available_resources(
                node_id, node_info[node_id], gpu_df, active_jobs
            )
    
    # Process each job in SRTF order using Synergy-TUNE algorithm
    for idx, job_tuple in enumerate(jobs_this_round):
        job_id, job_info = job_tuple
        job = active_jobs[job_id]
        
        if job.get("is_running", False):
            continue  # Skip already running jobs
        
        # Get job demand vector [gpu_deficit, cpu_deficit, mem_deficit, sspeed_deficit]
        gpu_demand = job_info.get("job_gpu_demand", 0)
        cpu_demand = job_info.get("job_cpu_demand", job_info.get("job_cpu_demand_orig", 0))
        mem_demand = job_info.get("job_mem_demand", job_info.get("job_mem_demand_orig", 0))
        sspeed_demand = job_info.get("job_sspeed_demand", job_info.get("job_sspeed_demand_orig", 0))
        
        # Calculate GPU deficit (for jobs not yet running, deficit equals demand)
        job_gpu_deficit = gpu_demand - job.get("num_GPUs_allocated", 0)
        
        # If job already has all needed GPUs and resources, skip placement
        if job_gpu_deficit == 0:
            continue
        
        demand_vec = [
            job_gpu_deficit,
            cpu_demand - job.get("cpus_allocated", 0),
            mem_demand - job.get("mem_allocated", 0),
            sspeed_demand - job.get("sspeed_allocated", 0)
        ]

        # Store demand values for potential retry after timeout
        gpu_d = int(demand_vec[0])
        cpu_d = float(demand_vec[1])
        mem_d = float(demand_vec[2])
        
        # Create a partial function wrapper for _call_allocate with job-specific parameters
        def _call_allocate_wrapper(available_gpus_dict, gpu_deficit, job_info_dict, time=None, fair=False, demand_vec=None):
            """Wrapper function that calls the standalone _call_allocate with job-specific parameters"""
            return _call_allocate(
                available_gpus_dict, gpu_deficit, job_info_dict, job_id,
                node_info, gpu_df, server_resource_usage, active_jobs,
                job_to_launch, time, fair, demand_vec
            )
        
        # Call _tune with initial=True to start the tuning process
        placement, found, res_map = _tune(
            job_info, demand_vec, job_gpu_deficit,
            peer_adjust=False, initial=True, final=False,
            _call_allocate=_call_allocate_wrapper,
            available_gpus=find_free_GPUs(gpu_df),
            time=0.0,  # Time should be passed from caller
            fair=True,
            node_info=node_info,
            gpu_df=gpu_df,
            server_resource_usage=server_resource_usage,
            active_jobs=active_jobs,
            jobs_to_terminate=jobs_to_terminate,
            job_id=job_id,
            job_to_launch=job_to_launch
        )

        if found:
            # Validate placement: must have at least one GPU
            # Group GPUs by node for better organization
            node_gpu_map = group_gpus_by_node(placement, gpu_df)
            
            # Store placement with node information
            # Format: {job_id: {'nodes': {node_id: [gpu_ids]}, 'gpus': [all_gpu_ids]}}
            # This provides both node-level and GPU-level information
            job_to_launch[job_id] = {
                'nodes': node_gpu_map,  # Node-level placement: {node_id: [gpu_ids]}
                'gpus': placement       # All GPU IDs (for backward compatibility)
            }
            # Store res_map in active_jobs for later use by job.allocate()
            active_jobs[job_id]["res_map"] = res_map
            # Update job state with allocated resources (critical for metrics calculation)
            actual_gpu_count = len(placement) if placement else 0
            active_jobs[job_id]["num_GPUs_allocated"] = actual_gpu_count
            active_jobs[job_id]["cpus_allocated"] = sum(r.get("cpu", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["mem_allocated"] = sum(r.get("mem", 0) for r in res_map.values()) if res_map else 0
            active_jobs[job_id]["sspeed_allocated"] = sum(r.get("sspeed", 0) for r in res_map.values()) if res_map else 0
            # Output CPU and memory allocation information
            total_cpu = active_jobs[job_id]["cpus_allocated"]
            total_mem = active_jobs[job_id]["mem_allocated"]
            # print(f"Synergy_SRTF: Job {job_id} allocated - Total: GPU={actual_gpu_count}, CPU={total_cpu}, Memory={total_mem:.2f}GB")
            if res_map:
                for server, resources in res_map.items():
                    node_id = server.server_id if hasattr(server, 'server_id') else server
                    # print(f"  Node {node_id}: GPU={resources.get('gpu', 0)}, CPU={resources.get('cpu', 0)}, Memory={resources.get('mem', 0):.2f}GB")
        else:
            # Placement not found - log why
            # print(f"[PLACEMENT_FAILED] Job {job_id}: Cannot find placement - GPU demand: {gpu_d}, CPU demand: {cpu_d:.2f}, Memory demand: {mem_d:.2f}")
            # Print available resources for debugging
            total_available_gpu = sum(server_resource_usage[node_id].get('gpu', 0) for node_id in server_resource_usage)
            total_available_cpu = sum(server_resource_usage[node_id].get('cpu', 0) for node_id in server_resource_usage)
            total_available_mem = sum(server_resource_usage[node_id].get('memory', 0) for node_id in server_resource_usage)
            # print(f"  Total available resources: GPU={total_available_gpu}, CPU={total_available_cpu:.2f}, Memory={total_available_mem:.2f}")
            for node_id in server_resource_usage:
                available = server_resource_usage[node_id]
                # print(f"    Node {node_id}: GPU={available.get('gpu', 0)}, CPU={available.get('cpu', 0):.2f}, Memory={available.get('memory', 0):.2f}")
        continue
            
    return (jobs_to_terminate, job_to_launch)


def _call_allocate(
    available_gpus_dict: dict,
    gpu_deficit: int,
    job_info_dict: dict,
    job_id: int,
    node_info: dict,
    gpu_df: pd.DataFrame,
    server_resource_usage: dict,
    active_jobs: dict,
    job_to_launch: dict,
    time: float = None,
    fair: bool = False,
    demand_vec: list = None
) -> Tuple[bool, list]:
    """
    Callback function for _tune to attempt allocation.
    
    Args:
        available_gpus_dict: Dictionary of available GPUs
        gpu_deficit: GPU deficit for the job
        job_info_dict: Job information dictionary
        job_id: Job ID
        node_info: Server information dictionary
        gpu_df: GPU dataframe
        server_resource_usage: Server resource usage dictionary (modified in place)
        active_jobs: Active jobs dictionary
        job_to_launch: Job to launch dictionary (modified in place)
        time: Current time (optional)
        fair: Whether to use fair-share allocation (optional)
        demand_vec: Demand vector [gpu_deficit, cpu_deficit, mem_deficit, sspeed_deficit] (optional)
    
    Returns:
        Tuple of (success: bool, placement: list)
    """
    if demand_vec is None:
        demand_vec = [gpu_deficit, 0, 0, 0]
    
    gpu_d_local = int(demand_vec[0])
    cpu_d_local = float(demand_vec[1])
    mem_d_local = float(demand_vec[2])
    
    # Try to find placement
    placement, found, res_map, updated_server_resource_usage = _synergy_find_placement(
        job_info_dict, node_info, gpu_df, server_resource_usage, active_jobs,
        gpu_d_local, cpu_d_local, mem_d_local
    )
    
    # Update server_resource_usage from returned value
    if found and updated_server_resource_usage:
        server_resource_usage.update(updated_server_resource_usage)
    
    # If placement is empty, it's not a valid placement
    if found and (not placement or len(placement) == 0):
        return (False, [])
    
    if found:
        node_gpu_map = group_gpus_by_node(placement, gpu_df)
        job_to_launch[job_id] = {
            'nodes': node_gpu_map,  # Node-level placement: {node_id: [gpu_ids]}
            'gpus': placement       # All GPU IDs (for backward compatibility)
        }
        return (True, placement)
    return (False, [])


def _tune(
        job_info: dict, demand_vec: list, job_gpu_deficit: int,
        peer_adjust: bool, initial: bool, final: bool,
        _call_allocate, available_gpus: dict, time: float, fair: bool,
        node_info: dict, gpu_df: pd.DataFrame, server_resource_usage: dict,
        active_jobs: dict, jobs_to_terminate: list,
        job_id: int = None, job_to_launch: dict = None
    ) -> Tuple[list, bool, dict]:
        """
        Tune job allocation using Synergy-TUNE algorithm.
        Args:
            job_info: Job information dictionary
            demand_vec: Demand vector [gpu_deficit, cpu_deficit, mem_deficit, sspeed_deficit]
            job_gpu_deficit: GPU deficit
            peer_adjust: Whether to adjust peer jobs
            initial: Whether this is initial call
            final: Whether this is final attempt
            _call_allocate: Allocation callback function
            available_gpus: Dictionary of available GPUs
            time: Current time
            fair: Whether to use fair-share allocation
            node_info: Server information dictionary
            gpu_df: GPU dataframe
            server_resource_usage: Server resource usage dictionary
            active_jobs: Active jobs dictionary
            jobs_to_terminate: List of jobs to terminate (modified in place)
        Returns:
            (placement: list, success: bool, res_map: dict)
        """
        # Final attempt: log failure and return
        if final:
            # Log failure information
            # In practice, should use logger here
            return ([], False, {})
        
        # Try to allocate with current demand vector
        gpu_demand = demand_vec[0]
        cpu_demand = demand_vec[1]
        mem_demand = demand_vec[2]
        
        # Call allocation function
        success, allocated_gpus = _call_allocate(
            available_gpus, job_gpu_deficit, job_info, time=time,
            fair=fair, demand_vec=demand_vec
        )
        
        if success:
            if final:
                # Log success
                pass
            # Create res_map from allocation
            res_map = {}
            if allocated_gpus:
                res_map = create_res_map_from_placement(
                    allocated_gpus, gpu_df, node_info,
                    gpu_demand, cpu_demand, mem_demand,
                    demand_vec[3] if len(demand_vec) > 3 else 0.0
                )
            return (allocated_gpus, True, res_map)
        
        # Initial: try to switch job to fair-share
        can_adjust = False
        if initial:
            can_adjust, new_demand_vec = _make_fair_share(job_info, demand_vec)
            if can_adjust:
                # Recursively call _tune with adjusted demand
                return _tune(
                    job_info, new_demand_vec, job_gpu_deficit,
                    False, False, False, _call_allocate, available_gpus,
                    time, fair, node_info, gpu_df, server_resource_usage,
                    active_jobs, jobs_to_terminate, job_id, job_to_launch
                )
        
        # Cannot adjust and peer not adjusted yet
        if not initial and not peer_adjust:
            return _tune(
                job_info, demand_vec, job_gpu_deficit,
                True, False, False, _call_allocate, available_gpus,
                time, fair, node_info, gpu_df, server_resource_usage,
                active_jobs, jobs_to_terminate, job_id, job_to_launch
            )
    
        # Peer adjust: reallocate peer jobs
        elif peer_adjust and not final:
            # Get underutilized servers
            server_handle_map = _get_underutilized_servers(
                job_gpu_deficit, available_gpus,
                consolidate=job_info.get("prefers_consolidation", False)
            )
            for serv, num_gpus_from_serv in server_handle_map.items():
                # Get actual GPU list for this server
                gpus = available_gpus.get(serv, [])[:num_gpus_from_serv]
                free_vec = [
                    num_gpus_from_serv,
                    server_resource_usage.get(serv, {}).get("cpu", 0),
                    server_resource_usage.get(serv, {}).get("memory", 0),
                    0, 0
                ]
                ratio = num_gpus_from_serv / job_gpu_deficit if job_gpu_deficit > 0 else 0
                demand_vec_share = [res * ratio for res in demand_vec]
                # Reallocate peer jobs
                jobs_to_realloc = _reallocate_peer(
                    demand_vec_share, free_vec, serv, active_jobs, gpu_df
                )
                
                # Save original state of peer jobs for rollback if allocation fails
                peer_jobs_original_state = {}
                for j_id in jobs_to_realloc:
                    if j_id in active_jobs:
                        j = active_jobs[j_id]
                        peer_jobs_original_state[j_id] = {
                            "res_map": copy.deepcopy(j.get("res_map")),
                            "num_GPUs_allocated": j.get("num_GPUs_allocated"),
                            "cpus_allocated": j.get("cpus_allocated"),
                            "mem_allocated": j.get("mem_allocated"),
                            "sspeed_allocated": j.get("sspeed_allocated", ),
                            "is_running": j.get("is_running", False),
                            "gpus": find_gpus_matching_JobID(j_id, gpu_df)
                        }
                
                # Process each peer job: deallocate and reallocate with fair-share
                for j_id in jobs_to_realloc:
                    if j_id not in active_jobs:
                        continue
                    j = active_jobs[j_id]
                    
                    # Save peer job's res_map (deep copy) - preserve server keys
                    peer_res_map = {}
                    j_res_map = j.get("res_map", {})
                    if j_res_map:
                        for server_key, resources in j_res_map.items():
                            # Preserve the original server key (ServerWrapper or int)
                            peer_res_map[server_key] = copy.deepcopy(resources)
                    
                    # Deallocate peer job (revert_iter=True to rollback progress)
                    # Free resources in server_resource_usage
                    
                    # Get peer job's demand vector (after deallocation, deficit = demand)
                    peer_demand_vec = [
                        j.get("job_gpu_demand"),
                        j.get("job_cpu_demand"),
                        j.get("job_mem_demand"),
                        j.get("job_sspeed_demand")
                    ]
                    # Try to adjust peer job to fair-share
                    can_adjust_peer, new_peer_demand_vec = _make_fair_share(j, peer_demand_vec)
                    if can_adjust_peer:
                        peer_demand_vec = new_peer_demand_vec
                    
                    # Get current GPU allocation for this peer job (needed for calculating GPU share)
                    gpus_realloc = find_gpus_matching_JobID(j_id, gpu_df)
                    
                    # Recalculate peer_res_map with fair-share allocation per server
                    # Based on GPU share per server (from original res_map)
                    for server_key in peer_res_map.keys():
                        # Get GPU count for this server from original res_map
                        original_gpu_count = peer_res_map[server_key].get('gpu_allocated', 0)
                        if original_gpu_count == 0:
                            continue
                        
                        # Calculate GPU share: GPUs on this server / total GPUs
                        gpu_share = original_gpu_count / len(gpus_realloc) if len(gpus_realloc) > 0 else 0
                        peer_demand_vec_share = [res * gpu_share for res in peer_demand_vec]
                        peer_alloc_map = _vector_to_map(peer_demand_vec_share)
                        peer_res_map[server_key] = peer_alloc_map
                    
                    # Reallocate peer job with updated res_map
                    if peer_res_map:
                        # Update server resource usage (treating entire node as a whole)
                        # This function only updates server_resource_usage, not gpu_df
                        update_node_resource_usage(peer_res_map, server_resource_usage)
                        
                        # Update job state
                        j["res_map"] = peer_res_map
                        j["num_GPUs_allocated"] = len(gpus_realloc)
                        j["cpus_allocated"] = sum(r.get("cpu_allocated", 0) for r in peer_res_map.values())
                        j["mem_allocated"] = sum(r.get("mem_allocated", 0) for r in peer_res_map.values())
                        j["sspeed_allocated"] = sum(r.get("sspeed_allocated", 0) for r in peer_res_map.values())
                        j["is_running"] = True
                
                # Try to allocate current job after reallocating peers
                updated_available_gpus = find_free_GPUs(gpu_df)
                success, allocated_gpus = _call_allocate(
                    updated_available_gpus, job_gpu_deficit, job_info,
                    time=time, fair=fair, demand_vec=demand_vec
                )
                
                if success:
                    res_map = {}
                    if allocated_gpus:
                        sspeed_d = demand_vec[3]
                        res_map = create_res_map_from_placement(
                            allocated_gpus, gpu_df, node_info,
                            gpu_demand, cpu_demand, mem_demand, sspeed_d
                        )
                    return (allocated_gpus, True, res_map)
                # If allocation failed for this server, continue to next server
                # (rollback will happen after all servers are tried)
                else:
                    # Allocation failed: rollback peer jobs to original state
                    _rollback_peer_jobs(
                        peer_jobs_original_state, active_jobs, gpu_df, 
                        server_resource_usage, jobs_to_terminate
                    )
            
        
        return ([], False, {})

def _rollback_peer_jobs(
        peer_jobs_original_state: dict,
        active_jobs: dict,
        gpu_df: pd.DataFrame,
        server_resource_usage: dict,
        jobs_to_terminate: list
    ) -> None:
        """
        回退 peer jobs 到原始状态
        
        Args:
            peer_jobs_original_state: 保存的 peer jobs 原始状态字典
            active_jobs: 活跃作业字典
            gpu_df: GPU 数据框
            server_resource_usage: 服务器资源使用情况字典
            jobs_to_terminate: 待终止作业列表
        """
        for j_id, original_state in peer_jobs_original_state.items():
            if j_id not in active_jobs:
                continue
            
            j = active_jobs[j_id]
            original_gpus = original_state["gpus"]
            original_res_map = original_state["res_map"]
            
            # Free current allocation (fair-share allocation)
            current_gpus = find_gpus_matching_JobID(j_id, gpu_df)
            if current_gpus:
                delete_job_by_id(gpu_df, j_id)
                # Free resources in server_resource_usage
                current_res_map = j.get("res_map", {})
                if current_res_map:
                    for server_key, resources in current_res_map.items():
                        if hasattr(server_key, 'node_id'):
                            serv_id = server_key.node_id
                        elif isinstance(server_key, int):
                            serv_id = server_key
                        else:
                            continue
                        
                        if serv_id in server_resource_usage:
                            cpu_freed = resources.get('cpu', 0) if isinstance(resources, dict) else 0
                            mem_freed = resources.get('mem', 0) if isinstance(resources, dict) else 0
                            gpu_freed = resources.get('gpu', 0) if isinstance(resources, dict) else 0
                            
                            server_resource_usage[serv_id]["gpu"] += gpu_freed
                            server_resource_usage[serv_id]["cpu"] += cpu_freed
                            server_resource_usage[serv_id]["memory"] += mem_freed
            
            # Restore original allocation
            if original_gpus and original_res_map:
                # Note: When restoring, we need to add resources back (opposite of allocation)
                # So we manually restore resources first, then mark GPUs
                
                # Mark GPUs as in use (but don't update server_resource_usage again since we already restored)
                update_node_resource_usage(original_res_map, server_resource_usage)
                
                # Restore job state
                j["res_map"] = copy.deepcopy(original_res_map)
                j["num_GPUs_allocated"] = original_state["num_GPUs_allocated"]
                j["cpus_allocated"] = original_state["cpus_allocated"]
                j["mem_allocated"] = original_state["mem_allocated"]
                j["sspeed_allocated"] = original_state["sspeed_allocated"]
                j["is_running"] = original_state["is_running"]
                
                # Remove from jobs_to_terminate if it was added
                if j_id in jobs_to_terminate:
                    jobs_to_terminate.remove(j_id)
def _synergy_find_placement(
        job_info: dict, node_info: dict, gpu_df: pd.DataFrame,
        server_resource_usage: dict, active_jobs: dict,
        gpu_demand: int, cpu_demand: float, mem_demand: float
    ) -> Tuple[list, bool, dict]:
        """
        Find placement with minimum fragmentation (Synergy-TUNE algorithm).
        Uses priority queue to allocate from servers with least free GPUs first.
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
            (list of GPU IDs, bool indicating if placement found, res_map dictionary, updated server_resource_usage)
        """
        # If GPU demand is 0, cannot place (job already has all needed GPUs or invalid demand)
        if gpu_demand <= 0:
            return ([], False, {}, server_resource_usage)
        
        free_gpus_dict = find_free_GPUs(gpu_df)
        total_free_gpus = sum(len(gpus) for gpus in free_gpus_dict.values())
        sspeed_demand = job_info.get("job_sspeed_demand", job_info.get("job_sspeed_demand_orig", 0))
        
        # Build demand vector [gpu, cpu, mem, sspeed]
        demand_vector = [gpu_demand, cpu_demand, mem_demand, sspeed_demand]
        
        # Check if job prefers consolidation
        prefers_consolidation = job_info.get("placement_preference") == "consolidated" or \
                               job_info.get("prefers_consolidation", False)
        
        # If job prefers consolidation, try to find a single server that fits
        if prefers_consolidation:
            candidates = []
            for node_id, gpus in free_gpus_dict.items():
                if len(gpus) >= gpu_demand:
                    # Check if server can fit the entire job
                    total_cpu_needed = cpu_demand
                    total_mem_needed = mem_demand
                    available = server_resource_usage.get(node_id, {})
                    fits = (available.get("gpu", 0) >= gpu_demand and
                            available.get("cpu", 0) >= total_cpu_needed and
                            available.get("memory", 0) >= total_mem_needed)
                    if fits:
                        candidates.append((len(gpus), node_id))
            if len(candidates) > 0:
                # Choose server with minimum free GPUs (to reduce fragmentation)
                _, target_node = min(candidates, key=lambda x: x[0])
                placement = free_gpus_dict[target_node][:gpu_demand]
                res_map = create_res_map_from_placement(
                    placement, gpu_df, node_info, gpu_demand, cpu_demand, mem_demand, sspeed_demand
                )
                # Update GPU state and server resource usage immediately
                # mark_gpu_in_use will update both gpu_df and server_resource_usage if res_map is provided
                update_node_resource_usage(res_map, server_resource_usage)
                return (placement, True, res_map, server_resource_usage)
        
        # If cannot be consolidated or does not prefer one, use priority queue placement
        # Normalize demand vector by GPU (per-GPU demand)        
        gpu_demand_norm = _gpu_normalized_vector(demand_vector)
        gpus_to_allocate, res_map = _top_synergy_gpus_placement(
            gpu_demand_norm, gpu_demand, free_gpus_dict,
            server_resource_usage, node_info
        )
        
        if gpus_to_allocate is None:
            return ([], False, {}, server_resource_usage)
        
        # Update GPU state immediately
        # Note: server_resource_usage is already updated in _top_synergy_gpus_placement during allocation
        # So we only need to mark GPUs as in use here
        update_node_resource_usage(res_map, server_resource_usage)
        
        return (gpus_to_allocate, True, res_map, server_resource_usage)

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

def free_node_resource_usage(
    res_map: dict,
    server_resource_usage: dict
) -> None:
    """
    释放节点资源使用情况，将整个节点上的资源看作一个整体。
    这是 update_node_resource_usage 的反向操作，用于释放资源。
    
    Args:
        res_map: 资源映射字典 {ServerWrapper/int: {'cpu'/'cpu_allocated': int, 
                 'mem'/'mem_allocated': float, 'gpu'/'gpu_allocated': int, 
                 'sspeed'/'sspeed_allocated': float}}
                支持两种格式：带或不带 _allocated 后缀
        server_resource_usage: 服务器资源使用情况字典，格式: {node_id: {"gpu": int, "cpu": float, "memory": float}}
    
    Returns:
        None
        原地修改 server_resource_usage,将资源加回到可用资源中
    """
    if res_map is None or server_resource_usage is None:
        return
    
    # 按节点汇总资源分配
    node_resources = {}
    for server_key, resources in res_map.items():
        # 提取 node_id
        if hasattr(server_key, 'node_id'):
            node_id = server_key.node_id
        elif hasattr(server_key, 'server_id'):
            node_id = server_key.server_id
        elif isinstance(server_key, int):
            node_id = server_key
        else:
            continue
        
        # 汇总该节点上的所有资源（支持两种键名格式）
        if node_id not in node_resources:
            node_resources[node_id] = {
                "cpu": 0,
                "mem": 0,
                "gpu": 0,
                "sspeed": 0
            }
        
        # 支持两种格式：带或不带 _allocated 后缀
        node_resources[node_id]["cpu"] += resources.get("cpu_allocated", resources.get("cpu", 0))
        node_resources[node_id]["mem"] += resources.get("mem_allocated", resources.get("mem", 0))
        node_resources[node_id]["gpu"] += resources.get("gpu_allocated", resources.get("gpu", 0))
        node_resources[node_id]["sspeed"] += resources.get("sspeed_allocated", resources.get("sspeed", 0))
    
    # 一次性释放每个节点的资源使用情况（将资源加回到可用资源中）
    for node_id, total_resources in node_resources.items():
        if node_id in server_resource_usage:
            # 将已分配的资源加回到可用资源中
            cpu_freed = total_resources.get("cpu", 0)
            mem_freed = total_resources.get("mem", 0)
            gpu_freed = total_resources.get("gpu", 0)
            
            server_resource_usage[node_id]["gpu"] = server_resource_usage[node_id].get("gpu", 0) + gpu_freed
            server_resource_usage[node_id]["cpu"] = server_resource_usage[node_id].get("cpu", 0) + cpu_freed
            server_resource_usage[node_id]["memory"] = server_resource_usage[node_id].get("memory", 0) + mem_freed
    
    return None


def group_gpus_by_node(gpu_ids: List[int], gpu_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Group GPU IDs by their node IDs
    
    Args:
        gpu_ids: List of GPU IDs
        gpu_df: GPU dataframe
    Returns:
        Dictionary mapping node_id to list of GPU IDs on that node
        Format: {node_id: [gpu_id1, gpu_id2, ...]}
    """
    if not gpu_ids:
        return {}
    
    node_gpu_map = {}
    for gpu_id in gpu_ids:
        node_row = gpu_df.loc[gpu_df["GPU_ID"] == gpu_id]
        if not node_row.empty:
            node_id = node_row["Node_ID"].iloc[0]
            if node_id not in node_gpu_map:
                node_gpu_map[node_id] = []
            node_gpu_map[node_id].append(gpu_id)
    
    return node_gpu_map
def find_gpus_matching_JobID(job_id: int, gpu_df: pd.DataFrame) -> list:
    """
    Finds the GPU IDs which are running the given job id
    """
    return gpu_df.loc[gpu_df["JOB_IDS"] == job_id]["GPU_ID"].tolist()


def create_res_map_from_placement(
    placement: List[int], 
    gpu_df: pd.DataFrame, 
    node_info: Dict[int, Dict],
    gpu_demand: int,
    cpu_demand: float,
    mem_demand: float,
    sspeed_demand: float = 0.0
) -> Dict[Any, Dict[str, Any]]:
    """
    Create res_map from GPU placement list
    Args:
        placement: List of GPU IDs allocated to the job
        gpu_df: GPU dataframe
        node_info: Server information dictionary
        gpu_demand: Total GPU demand for the job
        cpu_demand: Total CPU demand for the job
        mem_demand: Total memory demand for the job
        sspeed_demand: Total storage speed demand for the job
    Returns:
        res_map: Dictionary mapping server objects to resource allocations
    """
    res_map = {}
    
    if not placement:
        return res_map
    
    # Group GPUs by node_id
    node_gpu_count = {}
    for gpu_id in placement:
        node_id = gpu_df.loc[gpu_df["GPU_ID"] == gpu_id, "Node_ID"].iloc[0]
        node_gpu_count[node_id] = node_gpu_count.get(node_id, 0) + 1
    
    # Calculate resources per server
    # Use a list to track allocations and ensure total doesn't exceed demand
    server_list = []
    total_cpu_allocated = 0
    
    for node_id, gpu_count in node_gpu_count.items():
        # Create server wrapper
        server = ServerWrapper(node_id)
        
        # Calculate proportional resources for this server
        gpu_ratio = gpu_count / gpu_demand if gpu_demand > 0 else 0
        # Use floor to ensure we don't exceed demand, then adjust last server
        cpu_alloc = int(cpu_demand * gpu_ratio)  # Floor division
        mem_alloc = mem_demand * gpu_ratio
        sspeed_alloc = sspeed_demand * gpu_ratio
        
        server_list.append({
            'server': server,
            'gpu_count': gpu_count,
            'cpu_alloc': cpu_alloc,
            'mem_alloc': mem_alloc,
            'sspeed_alloc': sspeed_alloc
        })
        
        total_cpu_allocated += cpu_alloc
    
    # Adjust for rounding errors: add remainder to last server
    # This ensures total equals demand without exceeding it
    cpu_diff = int(cpu_demand) - total_cpu_allocated
    if cpu_diff > 0 and server_list:
        # Add remainder to last server (ensures total = demand, not exceeding)
        server_list[-1]['cpu_alloc'] += cpu_diff
    
    # Build res_map from allocations
    for item in server_list:
        res_map[item['server']] = {
            'gpu': item['gpu_count'],
            'cpu': item['cpu_alloc'],
            'mem': item['mem_alloc'],
            'sspeed': item['sspeed_alloc']
        }
    
    return res_map

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
    
    # If CPU or memory capacity is 0 but GPU capacity > 0, calculate from GPU count
    # This ensures CPU and memory are available if GPUs are available (proportional allocation)
    if server_capacity["gpu"] > 0:
        if server_capacity["cpu"] == 0:
            # Default: at least 3 CPU cores per GPU
            server_capacity["cpu"] = server_capacity["gpu"] * 3
        if server_capacity["memory"] == 0:
            # Default: at least 62.5 GB per GPU
            server_capacity["memory"] = server_capacity["gpu"] * 62.5
    
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
            if job.get("is_running", False):  # Fixed: default should be False, not True
                # Estimate based on job's GPU allocation on this node
                job_gpus_on_node = len(node_gpus.loc[
                    (node_gpus["IN_USE"] == True) & (node_gpus["JOB_IDS"] == job_id)
                ])
                if job_gpus_on_node > 0:
                    # Use res_map if available (actual allocated resources), otherwise use proportional allocation
                    res_map = job.get("res_map", {})
                    if res_map:
                        # Calculate actual allocated resources on this node from res_map
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
                                # Use actual allocated resources from res_map
                                used_resources["cpu"] += resources.get("cpu", 0)
                                used_resources["memory"] += resources.get("mem", 0)
                                break
                        else:
                            # res_map exists but this node not found, use proportional allocation
                            total_job_gpus = job.get("num_GPUs_allocated", 1)
                            if total_job_gpus > 0:
                                ratio = job_gpus_on_node / total_job_gpus
                                used_resources["cpu"] += job.get("job_cpu_demand", 0) * ratio
                                used_resources["memory"] += job.get("job_mem_demand", 0) * ratio
                    else:
                        # No res_map, use proportional allocation based on GPU count
                        total_job_gpus = job.get("num_GPUs_allocated", 1)
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
def _top_synergy_gpus_placement(
        demand_vector_norm: list, num_gpus: int, 
        free_gpus_dict: dict, server_resource_usage: dict, 
        node_info: dict
    ) -> Tuple[List[int], dict]:
        """
        Returns GPUs and allocation map using priority queue (least free GPUs first).
        Args:
            norm_demand_vector: Per-GPU normalized demand vector
            num_gpus: Number of GPUs needed
            free_gpus_dict: Dictionary of free GPUs by node
            server_resource_usage: Current server resource usage
            node_info: Server information dictionary
        Returns:
            (list of GPU IDs, res_map dictionary) or (None, None) if not enough GPUs
        """
        gpus_to_allocate = []
        res_map = {}
        
        # Track remaining GPUs per server (dictionary mapping node_id to list of GPU IDs)
        # Only include nodes that exist in server_resource_usage
        server_gpus_remaining = {
            node_id: list(gpus) 
            for node_id, gpus in free_gpus_dict.items() 
            if len(gpus) > 0  # Has remaining GPUs
            and node_id in server_resource_usage  # Node exists in resource usage
        }
        
        # Create priority queue: (num_free_gpus, node_id)
        # Use min-heap behavior (we want least free GPUs first)
        pq = [(len(gpus), node_id) for node_id, gpus in server_gpus_remaining.items()]
        heapq.heapify(pq)
        
        while len(gpus_to_allocate) < num_gpus and len(pq) > 0:
            _, node_id = heapq.heappop(pq)
            gpus = server_gpus_remaining.get(node_id, [])
            
            # Check if this server can fit the demand (at least one GPU worth of resources)
            fits = _fits_in_server(demand_vector_norm, node_id, server_resource_usage, node_info)
            if len(gpus) > 0 and fits:
                # Take one GPU from this server
                gpu_id = gpus.pop()
                gpus_to_allocate.append(gpu_id)
                
                # Temporarily hold resources (update server_resource_usage)
                server_alloc_map = _build_alloc_map(demand_vector_norm)
                if node_id in server_resource_usage:
                    server_resource_usage[node_id]["gpu"] -= server_alloc_map.get("gpu", 0)
                    server_resource_usage[node_id]["cpu"] -= server_alloc_map.get("cpu", 0)
                    server_resource_usage[node_id]["memory"] -= server_alloc_map.get("mem", 0)
                
                # Update res_map (cumulative)
                server_wrapper = ServerWrapper(node_id)
                if server_wrapper not in res_map:
                    res_map[server_wrapper] = server_alloc_map
                else:
                    res_map[server_wrapper] =_cumulative_map(
                        res_map[server_wrapper], server_alloc_map
                    )
                
                # Update server_gpus_remaining
                server_gpus_remaining[node_id] = gpus
                
                # If server still has GPUs, push back to queue
                if len(gpus) > 0:
                    heapq.heappush(pq, (len(gpus), node_id))
        
        # Release held resources if allocation failed (rollback)
        if len(gpus_to_allocate) < num_gpus:
            # for node_id, gpus in server_gpus_remaining.items():
            #     available = server_resource_usage.get(node_id, {})
            #     fits = self._fits_in_server(demand_vector_norm, node_id, server_resource_usage, node_info)
            #     print(f"    Node {node_id}: {len(gpus)} free GPUs, Available resources - GPU={available.get('gpu', 0)}, CPU={available.get('cpu', 0):.2f}, Memory={available.get('memory', 0):.2f}, Fits={fits}")
            # Rollback: restore resources for allocated GPUs
            for server_wrapper, resources in res_map.items():
                node_id = server_wrapper.node_id if hasattr(server_wrapper, 'node_id') else server_wrapper.server_id
                if node_id in server_resource_usage:
                    server_resource_usage[node_id]["gpu"] += resources.get("gpu", 0)
                    server_resource_usage[node_id]["cpu"] += resources.get("cpu", 0)
                    server_resource_usage[node_id]["memory"] += resources.get("mem", 0)
            return None, None
        
        return gpus_to_allocate, res_map
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
def _gpu_normalized_vector(vector: list) -> list:
        """
        Normalize demand vector by GPU (first element).
        Args:
            vector: Demand vector [gpu, cpu, mem, sspeed, ...]
        Returns:
            Normalized vector (per-GPU)
        """
        return [item / vector[0] for item in vector]

def _fits_in_server(
        norm_demand_vector: list, node_id: int, 
        server_resource_usage: dict, node_info: dict
    ) -> bool:
        """
        Check if normalized demand vector fits in a server.
        Args:
            norm_demand_vector: Normalized demand vector [gpu, cpu, mem, sspeed, 0]
            node_id: Server node ID
            server_resource_usage: Current server resource usage
            node_info: Server information dictionary
        Returns:
            bool indicating if the demand fits
        """
        
        available = server_resource_usage.get(node_id, {})
        if not available:
            return False
        
        # Check if server has enough resources for at least one GPU
        # norm_demand_vector is per-GPU normalized, so we check for 1 GPU worth
        gpu_req = norm_demand_vector[0] 
        cpu_req = norm_demand_vector[1]
        mem_req = norm_demand_vector[2] 
        
        available_gpu = available.get("gpu", 0)
        available_cpu = available.get("cpu", 0)
        available_mem = available.get("memory", 0)
        gpu_fits = available_gpu >= gpu_req
        cpu_fits = available_cpu >= cpu_req
        mem_fits = available_mem >= mem_req
        fits = gpu_fits and cpu_fits and mem_fits
        
        return fits

def _get_underutilized_servers(
     job_gpu_deficit: int, available_gpus: dict, consolidate: bool = False
) -> dict:
    """
    Get servers with underutilized GPUs using priority queue.
    Returns servers mapped to number of GPUs to allocate from each.
    Args:
        job_gpu_deficit: Number of GPUs needed (num_gpus)
        available_gpus: Dictionary of available GPUs by node {node_id: [gpu_ids]}
        consolidate: Whether to prefer consolidated placement
    Returns:
        Dictionary mapping node_id to number of GPUs to allocate from that server
    """
    num_gpus = job_gpu_deficit
    
    # If num_gpus > 1, try to find a single server with enough GPUs first
    if num_gpus > 1:
        # Try to find a server with enough GPUs for the entire job
        for node_id, gpu_list in available_gpus.items():
            if len(gpu_list) >= num_gpus:
                return {node_id: num_gpus}
    
    # Use priority queue (heap) to allocate from servers with least free GPUs
    # Create priority queue: (num_available_gpus, node_id)
    # Min-heap: servers with least GPUs first
    pq = [(len(gpu_list), node_id) for node_id, gpu_list in available_gpus.items()]
    heapq.heapify(pq)
    
    server_map = {}
    remaining_gpus = num_gpus
    
    # Allocate GPUs from servers with least free GPUs first
    while remaining_gpus > 0 and len(pq) > 0:
        gpus_available, node_id = heapq.heappop(pq)
        
        if gpus_available >= remaining_gpus:
            # This server has enough GPUs for remaining demand
            server_map[node_id] = remaining_gpus
            return server_map
        else:
            # Take all available GPUs from this server
            server_map[node_id] = gpus_available
            remaining_gpus -= gpus_available
    
    return server_map
def _make_fair_share(job_info: dict, demand_vec: list) -> Tuple[bool, list]:
        """
        Make fair-share adjustment for a job's demand vector.
        Args:
            job_info: Job information dictionary
            demand_vec: Demand vector [gpu_deficit, cpu_deficit, mem_deficit, sspeed_deficit, 0]
        Returns:
            (can_adjust: bool, new_demand_vec: list)
        """
        gpu_deficit = demand_vec[0]
        cpu_deficit = demand_vec[1]
        mem_deficit = demand_vec[2]
        
        # Calculate GPU-proportional allocation
        # 延迟导入以避免循环导入
        from .placement import calculate_gpu_proportional_allocation
        prop_alloc = calculate_gpu_proportional_allocation(gpu_deficit)
        
        # Check if current demand exceeds proportional share
        if cpu_deficit > prop_alloc["cpu"] or mem_deficit > prop_alloc["memory"]:
            # Can adjust to fair-share
            new_demand_vec = [
                gpu_deficit,
                prop_alloc["cpu"],
                prop_alloc["memory"],
                demand_vec[3] if len(demand_vec) > 3 else 0  # sspeed_deficit unchanged
            ]
            return (True, new_demand_vec)
        
        return (False, demand_vec)


def _build_alloc_map(norm_demand_vector: list) -> dict:
        """
        Build allocation map from normalized demand vector (per-GPU).
        Args:
            norm_demand_vector: Normalized demand vector [gpu, cpu, mem, sspeed, 0]
        Returns:
            Dictionary with resource allocations for one GPU
        """
        return {
            "gpu": int(norm_demand_vector[0]),
            "cpu": int(norm_demand_vector[1]),
            "mem": float(norm_demand_vector[2]),
            "sspeed": float(norm_demand_vector[3]) 
        }
def _cumulative_map(map1: dict, map2: dict) -> dict:
        """
        Cumulatively add two allocation maps.
        Args:
            map1: First allocation map
            map2: Second allocation map
        Returns:
            Combined allocation map
        """
        return {
            "gpu": map1.get("gpu", 0) + map2.get("gpu", 0),
            "cpu": map1.get("cpu", 0) + map2.get("cpu", 0),
            "mem": map1.get("mem", 0) + map2.get("mem", 0),
            "sspeed": map1.get("sspeed", 0) + map2.get("sspeed", 0)
        }

class ServerWrapper:
    """
    Simple wrapper class to represent a server for res_map
    """
    def __init__(self, node_id: int):
        self.server_id = node_id
        self.node_id = node_id
    
    def __hash__(self):
        return hash(self.server_id)
    
    def __eq__(self, other):
        if isinstance(other, ServerWrapper):
            return self.server_id == other.server_id
        return False

def _reallocate_peer(
    demand_vec_share: list, free_vec: list, serv: int,
    active_jobs: dict, gpu_df: pd.DataFrame
    ) -> list:
    """
    Reallocate peer jobs on a server to make room for new job.
    Args:
        demand_vec_share: Demand vector scaled by GPU share ratio [gpu, cpu, mem, sspeed]
        free_vec: Free resources vector on the server [gpu, cpu, mem, sspeed, ...]
        serv: Server node ID
        active_jobs: Active jobs dictionary
        gpu_df: GPU dataframe
    Returns:
        List of job IDs to reallocate
        """
    # Calculate spare resources needed: max(0, demand - available)
    spare_res_need = [max(0, x1 - x2) for (x1, x2) in zip(demand_vec_share, free_vec[:len(demand_vec_share)])]

    # If no spare resources needed, no jobs to reallocate
    if all(v == 0 for v in spare_res_need):
        return []

    # Get all running jobs on this server
    node_gpus = gpu_df.loc[gpu_df["Node_ID"] == serv]
    running_job_ids = node_gpus.loc[
        node_gpus["IN_USE"] == True
    ]["JOB_IDS"].unique()

    # Build job list with (job_id, job_dict) tuples
    job_list = []
    for job_id in running_job_ids:
        if job_id is not None and job_id in active_jobs:
            job = active_jobs[job_id]
            if job.get("is_running", False):
                job_list.append((job_id, job))

    # Sort jobs by:
    # 1. Number of servers (single-server jobs first)
    # 2. CPU/GPU ratio (descending, higher ratio first)
    # 3. Memory/GPU ratio (descending, higher ratio first)
    def get_sort_key(job_tuple):
        job_id, j = job_tuple
        # Get number of servers this job runs on
        job_gpus = find_gpus_matching_JobID(job_id, gpu_df)
        if not job_gpus:
            return (999, 0, 0)  # No GPUs, put at end
        
        # Count unique servers
        job_servers = set()
        for gpu_id in job_gpus:
            node_id = gpu_df.loc[gpu_df["GPU_ID"] == gpu_id, "Node_ID"].iloc[0]
            job_servers.add(node_id)
        num_servers = len(job_servers)
        
        # Get allocation vector [gpu, cpu, mem, sspeed, ...]
        alloc_vec = [
            j.get("num_GPUs_allocated", 0),
            j.get("cpus_allocated", 0),
            j.get("mem_allocated", 0),
            j.get("sspeed_allocated", 0)
        ]
        
        # Calculate ratios
        cpu_per_gpu = alloc_vec[1] / alloc_vec[0] if alloc_vec[0] > 0 else 0
        mem_per_gpu = alloc_vec[2] / alloc_vec[0] if alloc_vec[0] > 0 else 0
        
        return (num_servers, -cpu_per_gpu, -mem_per_gpu)  # Negative for descending

    job_list.sort(key=get_sort_key)

    jobs_to_realloc = []

    # Per-GPU fair share: [1 GPU, 3 CPU, 62.5 GB memory, 0 sspeed]
    per_server_size_fair = [1, 3, 62.5, 0]

    # Process each job to determine if it should be reallocated
    for job_id, j in job_list:
        # Get job's GPUs on this server
        job_gpus_this_server = len(node_gpus.loc[
            (node_gpus["IN_USE"] == True) & (node_gpus["JOB_IDS"] == job_id)
        ])
        
        if job_gpus_this_server == 0:
            continue
        
        # Calculate fair share for this job on this server
        job_fair = [x * job_gpus_this_server for x in per_server_size_fair]
        
        # Get job's total allocation vector
        job_alloc_vector = [
            j.get("num_GPUs_allocated", 0),
            j.get("cpus_allocated", 0),
            j.get("mem_allocated", 0),
            j.get("sspeed_allocated", 0)
        ]
        
        # Calculate GPU share: GPUs on this server / total GPUs
        total_job_gpus = job_alloc_vector[0] if job_alloc_vector[0] > 0 else 1
        job_gpu_share = job_gpus_this_server / total_job_gpus
        
        # Calculate allocation share on this server
        job_alloc_share = [res * job_gpu_share for res in job_alloc_vector]
        
        # Calculate excess: max(0, allocated - fair)
        job_excess_vec = [max(0, x1 - x2) for (x1, x2) in zip(job_alloc_share, job_fair)]
        
        # Calculate diff: max(0, excess - spare_res_need)
        diff = [max(0, x2 - x1) for (x1, x2) in zip(job_excess_vec, spare_res_need)]
        
        # Decision logic
        if all(v == 0 for v in diff):
            # This job's excess exactly matches or exceeds spare_res_need
            jobs_to_realloc.append(job_id)
            return jobs_to_realloc
        elif all(v > 0 for v in diff):
            # This job's excess is less than spare_res_need, skip it
            continue
        else:
            # Partial match: add this job and update spare_res_need
            jobs_to_realloc.append(job_id)
            spare_res_need = diff

    return jobs_to_realloc

def update_node_resource_usage(
    res_map: dict,
    server_resource_usage: dict
) -> None:
    """
    更新节点资源使用情况，将整个节点上的资源看作一个整体。
    只更新 server_resource_usage,不更新 gpu_df。
    
    Args:
        res_map: 资源映射字典 {ServerWrapper/int: {'cpu_allocated': int, 'mem_allocated': float, 
                 'gpu_allocated': int, 'sspeed_allocated': float}}
        server_resource_usage: 服务器资源使用情况字典，格式: {node_id: {"gpu": int, "cpu": float, "memory": float}}
    
    Returns:
        None
        原地修改 server_resource_usage
    """
    if res_map is None or server_resource_usage is None:
        return
    
    # 按节点汇总资源分配
    node_resources = {}
    for server_key, resources in res_map.items():
        # 提取 node_id
        if hasattr(server_key, 'node_id'):
            node_id = server_key.node_id
        elif hasattr(server_key, 'server_id'):
            node_id = server_key.server_id
        elif isinstance(server_key, int):
            node_id = server_key
        else:
            continue
        
        # 汇总该节点上的所有资源
        if node_id not in node_resources:
            node_resources[node_id] = {
                "cpu_allocated": 0,
                "mem_allocated": 0,
                "gpu_allocated": 0,
                "sspeed_allocated": 0
            }
        
        node_resources[node_id]["cpu_allocated"] += resources.get("cpu_allocated", 0)
        node_resources[node_id]["mem_allocated"] += resources.get("mem_allocated", 0)
        node_resources[node_id]["gpu_allocated"] += resources.get("gpu_allocated", 0)
        node_resources[node_id]["sspeed_allocated"] += resources.get("sspeed_allocated", 0)
    
    # 一次性更新每个节点的资源使用情况
    for node_id, total_resources in node_resources.items():
        if node_id in server_resource_usage:
            # 从可用资源中减去已分配的资源
            cpu_allocated = total_resources.get("cpu_allocated", 0)
            mem_allocated = total_resources.get("mem_allocated", 0)
            gpu_allocated = total_resources.get("gpu_allocated", 0)
            
            server_resource_usage[node_id]["gpu"] = server_resource_usage[node_id].get("gpu", 0) - gpu_allocated
            server_resource_usage[node_id]["cpu"] = server_resource_usage[node_id].get("cpu", 0) - cpu_allocated
            server_resource_usage[node_id]["memory"] = server_resource_usage[node_id].get("memory", 0) - mem_allocated
    
    return None