import pandas as pd
import copy
import heapq
from typing import Tuple, List, Dict, Any


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
        scheduler = new_job_schedule.get("scheduler")
        jobs_to_terminate = list()
        job_to_launch = dict()
        launched_job_ids = list()
        # go over jobs in job order
        
        #Borrowed from https://github.com/msr-fiddle/synergy/blob/master/simulator/resources/cluster.py#L581
        if scheduler=="Synergy_fifo":
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
            
            # Initialize server resource usage from cluster_state if available
            # Otherwise, calculate from scratch
            server_resource_usage = {}
            if cluster_state and hasattr(cluster_state, 'server_resource_usage'):
                # Use cluster_state.server_resource_usage as base
                for node_id in node_info:
                    if node_id in cluster_state.server_resource_usage:
                        # Get capacity from node_info
                        cpu_capacity = node_info[node_id].get("numCPUcores", 0)
                        mem_capacity = node_info[node_id].get("memoryCapacity", 0)
                        gpu_capacity = node_info[node_id].get("numGPUs", 0)
                        
                        # Get used resources from cluster_state
                        cpu_used = cluster_state.server_resource_usage[node_id].get("cpu_used", 0)
                        mem_used = cluster_state.server_resource_usage[node_id].get("memory_used", 0)
                        
                        # Calculate available resources
                        server_resource_usage[node_id] = {
                            "gpu": gpu_capacity - len(gpu_df.loc[(gpu_df["Node_ID"] == node_id) & (gpu_df["IN_USE"] == True)]),
                            "cpu": max(0, cpu_capacity - cpu_used),
                            "memory": max(0, mem_capacity - mem_used)
                        }
                    else:
                        # Fallback to calculation if node not in cluster_state
                        server_resource_usage[node_id] = get_server_available_resources(
                            node_id, node_info[node_id], gpu_df, active_jobs
                        )
            else:
                # Fallback: calculate from scratch if cluster_state not available
                for node_id in node_info:
                    server_resource_usage[node_id] = get_server_available_resources(
                        node_id, node_info[node_id], gpu_df, active_jobs
                    )
            
            # Process each job in order using Synergy-TUNE algorithm
            for job_tuple in jobs_this_round:
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
                job_gpu_deficit = gpu_demand - job.get("num_GPUs", 0)
                demand_vec = [
                    job_gpu_deficit,
                    cpu_demand - job.get("cpus", 0),
                    mem_demand - job.get("mem", 0),
                    sspeed_demand - job.get("sspeed", 0)
                ]
                
                # Define allocation callback function
                def _call_allocate(available_gpus_dict, gpu_deficit, job_info_dict, time=None, fair=False, demand_vec=None):
                    """Callback function for _tune to attempt allocation"""
                    if demand_vec is None:
                        demand_vec = [gpu_deficit, 0, 0, 0]
                    
                    gpu_d = int(demand_vec[0])
                    cpu_d = float(demand_vec[1])
                    mem_d = float(demand_vec[2])
                    
                    # Try to find placement
                    placement, found, res_map, updated_server_resource_usage = self._synergy_find_placement(
                        job_info_dict, node_info, gpu_df, server_resource_usage, active_jobs,
                        gpu_d, cpu_d, mem_d
                    )
                    
                    # Update server_resource_usage from returned value
                    nonlocal server_resource_usage
                    if found and updated_server_resource_usage:
                        server_resource_usage = updated_server_resource_usage
                    
                    if found:
                        return (True, placement)
                    return (False, [])
                
                # Call _tune with initial=True to start the tuning process
                placement, found, res_map = self._tune(
                    job_info, demand_vec, job_gpu_deficit,
                    peer_adjust=False, initial=True, final=False,
                    _call_allocate=_call_allocate,
                    available_gpus=find_free_GPUs(gpu_df),
                    time=0.0,  # Time should be passed from caller
                    fair=True,
                    node_info=node_info,
                    gpu_df=gpu_df,
                    server_resource_usage=server_resource_usage,
                    active_jobs=active_jobs,
                    jobs_to_terminate=jobs_to_terminate
                )
                
                if found:
                    # Store GPU placement (keep compatibility with existing code)
                    job_to_launch[job_id] = placement
                    # Store res_map in active_jobs for later use by job.allocate()
                    active_jobs[job_id]["res_map"] = res_map
                    # Update job dictionary with adjusted CPU and memory demands if needed
                    # (demand_vec may have been adjusted by _make_fair_share)
                    if demand_vec[1] != job_info.get("job_cpu_demand_orig", 0):
                        active_jobs[job_id]["job_cpu_demand"] = demand_vec[1]
                    if demand_vec[2] != job_info.get("job_mem_demand_orig", 0):
                        active_jobs[job_id]["job_mem_demand"] = demand_vec[2]
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
                    
                    # Create res_map for CPU and memory proportional allocation
                    # Get job resource demands (use original demands if available)
                    gpu_demand = job.get("job_gpu_demand", job.get("num_GPUs", len(placement)))
                    cpu_demand = job.get("job_cpu_demand_orig", job.get("job_cpu_demand", 0))
                    mem_demand = job.get("job_mem_demand_orig", job.get("job_mem_demand", 0))
                    sspeed_demand = job.get("job_sspeed_demand_orig", job.get("job_sspeed_demand", 0))
                    
                    # If CPU/memory demands are not set, use default proportional allocation
                    # Default: assume 3 CPU cores and 62.5 GB memory per GPU (common defaults)
                    if cpu_demand <= 0:
                        cpu_demand = gpu_demand * 3  # Default: 3 CPU cores per GPU
                    if mem_demand <= 0:
                        mem_demand = gpu_demand * 62.5  # Default: 62.5 GB memory per GPU 
                    # Create res_map from GPU placement
                    # This will proportionally allocate CPU and memory based on GPU allocation
                    res_map = create_res_map_from_placement(
                        placement, gpu_df, node_info,
                        gpu_demand, cpu_demand, mem_demand, sspeed_demand
                    )
                    
                    # Store res_map in active_jobs for later use by job.allocate()
                    active_jobs[job_id]["res_map"] = res_map
                    
                    # update manual-pipeline-list for bert and gpt
                    mark_gpu_in_use(gpu_df, placement, job_id)
                else:
                    break
            return (jobs_to_terminate, job_to_launch)

    def _gpu_normalized_vector(self, vector: list) -> list:
        """
        Normalize demand vector by GPU (first element).
        Args:
            vector: Demand vector [gpu, cpu, mem, sspeed, ...]
        Returns:
            Normalized vector (per-GPU)
        """
        if not vector or len(vector) == 0 or vector[0] == 0:
            return [1.0] + [0.0] * (len(vector) - 1) if len(vector) > 1 else [1.0]
        return [item / vector[0] for item in vector]

    def _fits_in_server(
        self, norm_demand_vector: list, node_id: int, 
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
        if not norm_demand_vector or len(norm_demand_vector) < 3:
            return False
        
        available = server_resource_usage.get(node_id, {})
        # Check if server has enough resources for at least one GPU
        # norm_demand_vector is per-GPU normalized, so we check for 1 GPU worth
        gpu_req = norm_demand_vector[0] if norm_demand_vector[0] > 0 else 1
        cpu_req = norm_demand_vector[1] if len(norm_demand_vector) > 1 else 0
        mem_req = norm_demand_vector[2] if len(norm_demand_vector) > 2 else 0
        
        return (
            available.get("gpu", 0) >= gpu_req and
            available.get("cpu", 0) >= cpu_req and
            available.get("memory", 0) >= mem_req
        )

    def _build_alloc_map(self, norm_demand_vector: list) -> dict:
        """
        Build allocation map from normalized demand vector (per-GPU).
        Args:
            norm_demand_vector: Normalized demand vector [gpu, cpu, mem, sspeed, 0]
        Returns:
            Dictionary with resource allocations for one GPU
        """
        if not norm_demand_vector or len(norm_demand_vector) < 3:
            return {"gpu": 1, "cpu": 0, "mem": 0, "sspeed": 0}
        
        return {
            "gpu": int(norm_demand_vector[0]) if norm_demand_vector[0] > 0 else 1,
            "cpu": int(norm_demand_vector[1]) if len(norm_demand_vector) > 1 else 0,
            "mem": float(norm_demand_vector[2]) if len(norm_demand_vector) > 2 else 0,
            "sspeed": float(norm_demand_vector[3]) if len(norm_demand_vector) > 3 else 0
        }

    def _cumulative_map(self, map1: dict, map2: dict) -> dict:
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

    def _top_synergy_gpus_placement(
        self, norm_demand_vector: list, num_gpus: int, 
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
        
        # Create priority queue: (num_free_gpus, node_id)
        # Use negative for min-heap behavior (we want least free GPUs first)
        pq = [(len(gpus), node_id) for node_id, gpus in free_gpus_dict.items() if len(gpus) > 0]
        heapq.heapify(pq)
        
        # Track remaining GPUs per server
        server_gpus_remaining = {node_id: list(gpus) for node_id, gpus in free_gpus_dict.items()}
        
        while len(gpus_to_allocate) < num_gpus and len(pq) > 0:
            _, node_id = heapq.heappop(pq)
            
            # Check if this server can fit the demand
            if self._fits_in_server(norm_demand_vector, node_id, server_resource_usage, node_info):
                gpus = server_gpus_remaining.get(node_id, [])
                if len(gpus) > 0:
                    # Take one GPU from this server
                    gpu_id = gpus.pop(0)
                    gpus_to_allocate.append(gpu_id)
                    
                    # Build allocation map for this GPU
                    server_alloc_map = self._build_alloc_map(norm_demand_vector)
                    
                    # Update res_map (cumulative)
                    server_wrapper = ServerWrapper(node_id)
                    if server_wrapper not in res_map:
                        res_map[server_wrapper] = server_alloc_map
                    else:
                        res_map[server_wrapper] = self._cumulative_map(
                            res_map[server_wrapper], server_alloc_map
                        )
                    
                    # Update server resource usage (temporarily hold resources)
                    if node_id in server_resource_usage:
                        server_resource_usage[node_id]["gpu"] -= server_alloc_map.get("gpu", 1)
                        server_resource_usage[node_id]["cpu"] -= server_alloc_map.get("cpu", 0)
                        server_resource_usage[node_id]["memory"] -= server_alloc_map.get("mem", 0)
                    
                    # If server still has GPUs, push back to queue
                    if len(gpus) > 0:
                        heapq.heappush(pq, (len(gpus), node_id))
        
        if len(gpus_to_allocate) < num_gpus:
            return None, None
        
        return gpus_to_allocate, res_map

    def _synergy_find_placement(
        self, job_info: dict, node_info: dict, gpu_df: pd.DataFrame,
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
        free_gpus_dict = find_free_GPUs(gpu_df)
        sspeed_demand = job_info.get("job_sspeed_demand", job_info.get("job_sspeed_demand_orig", 0))
        
        # Build demand vector [gpu, cpu, mem, sspeed]
        demand_vector = [gpu_demand, cpu_demand, mem_demand, sspeed_demand]
        
        # Normalize demand vector by GPU (per-GPU demand)
        job_demand_vector_gpu_norm = self._gpu_normalized_vector(demand_vector)
        
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
                    if (available.get("gpu", 0) >= gpu_demand and
                        available.get("cpu", 0) >= total_cpu_needed and
                        available.get("memory", 0) >= total_mem_needed):
                        candidates.append((len(gpus), node_id))
            
            if len(candidates) > 0:
                # Choose server with minimum free GPUs (to reduce fragmentation)
                _, target_node = min(candidates, key=lambda x: x[0])
                placement = free_gpus_dict[target_node][:gpu_demand]
                res_map = create_res_map_from_placement(
                    placement, gpu_df, node_info, gpu_demand, cpu_demand, mem_demand, sspeed_demand
                )
                # Update GPU state and server resource usage immediately
                mark_gpu_in_use(gpu_df, placement, job_info.get("job_id"))
                # Update server_resource_usage
                if target_node in server_resource_usage:
                    server_resource_usage[target_node]["gpu"] -= gpu_demand
                    server_resource_usage[target_node]["cpu"] -= cpu_demand
                    server_resource_usage[target_node]["memory"] -= mem_demand
                return (placement, True, res_map, server_resource_usage)
        
        # If cannot be consolidated or does not prefer one, use priority queue placement
        gpus_to_allocate, res_map = self._top_synergy_gpus_placement(
            job_demand_vector_gpu_norm, gpu_demand, free_gpus_dict,
            server_resource_usage, node_info
        )
        
        if gpus_to_allocate is None:
            return ([], False, {}, server_resource_usage)
        
        # Update GPU state immediately
        mark_gpu_in_use(gpu_df, gpus_to_allocate, job_info.get("job_id"))
        
        # Convert ServerWrapper keys in res_map to match create_res_map_from_placement format
        # The res_map from _top_synergy_gpus_placement already has ServerWrapper keys
        # which is compatible with the expected format
        
        return (gpus_to_allocate, True, res_map, server_resource_usage)


    def _make_fair_share(self, job_info: dict, demand_vec: list) -> Tuple[bool, list]:
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

    def _get_underutilized_servers(
        self, job_gpu_deficit: int, available_gpus: dict, consolidate: bool = False
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

    def _reallocate_peer(
        self, demand_vec_share: list, free_vec: list, serv: int,
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
                j.get("num_GPUs", 0),
                j.get("cpus", 0),
                j.get("mem", 0),
                j.get("sspeed", 0)
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
                j.get("num_GPUs", 0),
                j.get("cpus", 0),
                j.get("mem", 0),
                j.get("sspeed", 0)
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

    def _tune(
        self, job_info: dict, demand_vec: list, job_gpu_deficit: int,
        peer_adjust: bool, initial: bool, final: bool,
        _call_allocate, available_gpus: dict, time: float, fair: bool,
        node_info: dict, gpu_df: pd.DataFrame, server_resource_usage: dict,
        active_jobs: dict, jobs_to_terminate: list
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
            can_adjust, new_demand_vec = self._make_fair_share(job_info, demand_vec)
            if can_adjust:
                # Recursively call _tune with adjusted demand
                return self._tune(
                    job_info, new_demand_vec, job_gpu_deficit,
                    False, False, False, _call_allocate, available_gpus,
                    time, fair, node_info, gpu_df, server_resource_usage,
                    active_jobs, jobs_to_terminate
                )
        
        # Cannot adjust and peer not adjusted yet
        if not can_adjust and not peer_adjust:
            return self._tune(
                job_info, demand_vec, job_gpu_deficit,
                False, False, False, _call_allocate, available_gpus,
                time, fair, node_info, gpu_df, server_resource_usage,
                active_jobs, jobs_to_terminate
            )
        
        # Cannot adjust but peer already adjusted
        if not can_adjust and peer_adjust:
            return self._tune(
                job_info, demand_vec, job_gpu_deficit,
                True, False, True, _call_allocate, available_gpus,
                time, fair, node_info, gpu_df, server_resource_usage,
                active_jobs, jobs_to_terminate
            )
        
        # Peer adjust: reallocate peer jobs
        if peer_adjust and not final:
            # Get underutilized servers
            server_handle_map = self._get_underutilized_servers(
                job_gpu_deficit, available_gpus,
                consolidate=job_info.get("prefers_consolidation", False)
            )
            
            peer_res_map = {}
            gpus_realloc = []
            
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
                jobs_to_realloc = self._reallocate_peer(
                    demand_vec_share, free_vec, serv, active_jobs, gpu_df
                )
                
                for j_id in jobs_to_realloc:
                    if j_id not in active_jobs:
                        continue
                    
                    j = active_jobs[j_id]
                    # Deallocate peer job
                    j_gpus = find_gpus_matching_JobID(j_id, gpu_df)
                    if j_gpus:
                        delete_job_by_id(gpu_df, j_id)
                        # Free resources
                        j_res_map = j.get("res_map", {})
                        if j_res_map:
                            for server_key, resources in j_res_map.items():
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
                        
                        if j_id not in jobs_to_terminate:
                            jobs_to_terminate.append(j_id)
                        j["is_running"] = False
                    
                    # Get peer job's demand vector
                    peer_demand_vec = [
                        j.get("job_gpu_demand", 0) - j.get("num_GPUs", 0),
                        j.get("job_cpu_demand", 0) - j.get("cpus", 0),
                        j.get("job_mem_demand", 0) - j.get("mem", 0),
                        j.get("job_sspeed_demand", 0) - j.get("sspeed", 0)
                    ]
                    
                    # Calculate GPU share for peer job
                    gpus_realloc = j_gpus if j_gpus else []
                    if gpus_realloc:
                        gpu_share = len(gpus_realloc) / len(gpus_realloc) if gpus_realloc else 0
                        peer_demand_vec_share = [res * gpu_share for res in peer_demand_vec]
                        peer_alloc_map = self._vector_to_map(peer_demand_vec_share)
                        
                        # Create server wrapper
                        serv_wrapper = ServerWrapper(serv)
                        peer_res_map[serv_wrapper] = peer_alloc_map
                
                # Try to allocate current job after reallocating peers
                updated_available_gpus = find_free_GPUs(gpu_df)
                success, allocated_gpus = _call_allocate(
                    updated_available_gpus, job_gpu_deficit, job_info,
                    time=time, fair=fair, demand_vec=demand_vec
                )
                
                if success:
                    res_map = {}
                    if allocated_gpus:
                            sspeed_d = demand_vec[3] if len(demand_vec) > 3 else 0.0
                        res_map = create_res_map_from_placement(
                            allocated_gpus, gpu_df, node_info,
                            gpu_demand, cpu_demand, mem_demand, sspeed_d
                        )
                    return (allocated_gpus, True, res_map)
        
        return ([], False, {})

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
