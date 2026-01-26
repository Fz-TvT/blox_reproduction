import os
import sys
import json
import grpc
import logging
from typing import List
from concurrent import futures
from collections import defaultdict
import numpy as np

sys.path.append(os.path.join((__file__), "./grpc_stubs"))
import nm_pb2
import nm_pb2_grpc

import rm_pb2

import simulator_pb2
import simulator_pb2_grpc


def get_tput_from_job_dict(job_dict, cpu_allocated=None, mem_allocated=None):
    """
    从字典形式的 job 中获取 tput
    
    Args:
        job_dict: 作业字典
        cpu_allocated: 分配的 CPU 数量（如果为 None,使用 job_dict["cpus"])
        mem_allocated: 分配的内存数量（如果为 None,使用 job_dict["mem"])
    
    Returns:
        tput 值(float),如果无法获取则返回 synergy_speedup 或 1.0(默认值）
    """

    
    # 优先从保存的 tput_matrix 列表中获取（job_model 可能在 _clean_sim_job 中被移除）
    tput_matrix = None
    if "tput_matrix" in job_dict and job_dict["tput_matrix"] is not None:
        # 将列表转换回 numpy 数组
        tput_matrix = np.array(job_dict["tput_matrix"])
    elif "job_model" in job_dict and job_dict["job_model"] is not None:
        # 如果 job_model 还存在，直接从对象获取
        job_model = job_dict["job_model"]
        if hasattr(job_model, "tput") and job_model.tput is not None:
            tput_matrix = job_model.tput if isinstance(job_model.tput, np.ndarray) else np.array(job_model.tput)
    
    if tput_matrix is None:
        logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: tput_matrix not found")
        return 1.0
    
    # 使用传入的值或字典中的值（优先使用新字段名 *_allocated）
    cpu = cpu_allocated if cpu_allocated is not None else job_dict.get("cpus_allocated", job_dict.get("cpus", 0))
    mem = mem_allocated if mem_allocated is not None else job_dict.get("mem_allocated", job_dict.get("mem", 0))
    
    # 如果 CPU 或内存为 0，无法查找 tput
    if cpu == 0 or mem == 0:
        logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: cpu={cpu}, mem={mem}, cannot get tput (resources not allocated yet)")
        return 1.0
    
    # CPU 和内存值映射（与 Job 类中的定义一致）
    cpu_val = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9, 7: 12, 8: 24}
    mem_val = {0: 62.5, 1: 125, 2: 187.5, 3: 250, 4: 312.5, 5: 375, 6: 437.5, 7: 500}
    
    # 找到对应的索引（使用最接近的值）
    def get_idx(id_map, value):
        # 首先尝试精确匹配
        for k, v in id_map.items():
            if value == v:
                return k
        # 如果找不到精确匹配，找最接近的值
        closest_key = None
        min_diff = float('inf')
        for k, v in id_map.items():
            diff = abs(value - v)
            if diff < min_diff:
                min_diff = diff
                closest_key = k
        return closest_key
    
    cpu_idx = get_idx(cpu_val, cpu)
    mem_idx = get_idx(mem_val, mem)
    
    if cpu_idx is None or mem_idx is None:
        logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: Cannot find index for cpu={cpu}, mem={mem}")
        return 1.0
    
    # 从 tput 矩阵中获取值
    try:
        if isinstance(tput_matrix, np.ndarray):
            # 确保索引在有效范围内
            if cpu_idx < tput_matrix.shape[0] and mem_idx < tput_matrix.shape[1]:
                tput_value = tput_matrix[cpu_idx, mem_idx]
                if tput_value > 0:
                    return tput_value
                else:
                    logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: tput value is 0 at [{cpu_idx}, {mem_idx}]")
            else:
                logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: Index out of range: cpu_idx={cpu_idx}, mem_idx={mem_idx}, shape={tput_matrix.shape}")
        else:
            logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: tput_matrix is not a numpy array")
    except Exception as e:
        logging.warning(f"Job {job_dict.get('job_id', 'unknown')}: Error getting tput from matrix: {e}")
    
    return 1.0


class ResourceManagerComm(object):
    """
    Resource Manager communication class
    """

    def __init__(self, node_manager_port) -> None:
        self.rpc_port = node_manager_port
        return None

    def launch_job(
        self,
        job_id: int,
        job_description: dict,
        local_gpu_ids: List[int],
        ipaddr_list: List[str],
    ) -> None:
        """
        Notify respesctive node managers to launch jobs.
        For each job this is called once.
        Args:
            job_description: Job description from the job ID dictionary
            gpu_ids: Number of GPUS to launch
            ipaddr : list of IP address to contact node manager
        Returns:
            None
        """
        dist_rank = 0
        world_size = len(local_gpu_ids)
        if job_description["simulation"] == False:
            for ipaddr, lgid in zip(ipaddr_list, local_gpu_ids):
                if dist_rank == 0:
                    master_ip_address = ipaddr
                ipaddr = f"{ipaddr}:{self.rpc_port}"
                launch_dict = dict()
                launch_dict["job_id"] = job_id
                # if job_id == 2:
                # import ipdb

                # ipdb.set_trace()
                launch_dict["local_GPU_ID"] = lgid
                if "launch_command" not in job_description:
                    raise Exception("Missing Launch Command")
                launch_dict["launch_command"] = job_description["launch_command"]
                if "suspended" in job_description:
                    launch_dict["should_resume"] = job_description["suspended"]
                else:
                    launch_dict["should_resume"] = "0"

                # we have simplified this

                launch_params = list()
                launch_params.append(lgid)
                launch_params.append(master_ip_address)
                launch_params.append(world_size)
                launch_params.append(dist_rank)
                launch_params.extend(job_description["launch_params"])
                launch_params.append(str(launch_dict["job_id"]))
                # launch_params_string = ",".join(launch_params)

                # sending parameters for launch params
                environment_variable_pairs = dict()
                environment_variable_pairs["local_gpu_id"] = str(lgid)
                environment_variable_pairs["master_ip_address"] = master_ip_address
                environment_variable_pairs["world_size"] = str(world_size)
                environment_variable_pairs["dist_rank"] = str(dist_rank)
                environment_variable_pairs["job_id"] = str(launch_dict["job_id"])
                environment_variable_pairs["local_accessible_gpus"] = ",".join(
                    [str(x) for x in local_gpu_ids]
                )
                launch_dict["env_variables"] = environment_variable_pairs
                launch_dict["launch_params"] = launch_params
                print("Launch Params {}".format(launch_params))
                # ["0,", "6001", "1", "resnet50", "64" ]
                launch_request = rm_pb2.JsonResponse()
                launch_request.response = json.dumps(launch_dict)
                with grpc.insecure_channel(ipaddr) as channel:
                    stub = nm_pb2_grpc.NMServerStub(channel)
                    response = stub.LaunchJob(launch_request)
                print(
                    f"Launched Job {job_id}, response {response}, request {launch_dict}"
                )
                dist_rank += 1

            return None
        elif job_description["simulation"] == True:
            # TODO: Add time for checkpoint and restore
            return None

    def terminate_jobs(
        self,
        job_id_list: List[int],
        terminate_rank_0_ipaddr: List[int],
        all_ipaddr_list: List[List[str]],
        terminate_simulation: List[bool],
    ) -> None:
        """
        Given a list of Job_ID's and their corresponding ip addresses.
        Terminate these jobs.
        Args:
            job_id: list of job ids to terminate
            ipaddr: list of corresponding ip addresses
            terminate_simulation : whether job is simulation or not
        Returns:
            None
        """
        # TODO: Multithread this
        assert len(job_id_list) == len(terminate_simulation)
        assert len(job_id_list) == len(terminate_rank_0_ipaddr)
        assert len(job_id_list) == len(all_ipaddr_list)

        send_request_dict = defaultdict(list)
        other_ip_address_to_send = defaultdict(list)
        for job_id, rank_0_ipaddr, all_ip_addr, simulation in zip(
            job_id_list, terminate_rank_0_ipaddr, all_ipaddr_list, terminate_simulation
        ):

            if not simulation:
                all_ip_addr = [f"{all_ip}:{self.rpc_port}" for all_ip in all_ip_addr]

                for send_ip_address in all_ip_addr:
                    # ipaddr = f"{send_ip_address}:{self.rpc_port}"
                    send_request_dict[send_ip_address].append(job_id)
                    other_ip_address_to_send[send_ip_address].append(all_ip_addr)

        for send_ip_address in send_request_dict:
            terminate_request = rm_pb2.JsonResponse()

            terminate_request.response = json.dumps(
                {
                    "Job_ID_list": send_request_dict[send_ip_address],
                    "IP_addr_terminate": other_ip_address_to_send[send_ip_address],
                }
            )
            # TODO: Add simulator


            with grpc.insecure_channel(send_ip_address) as channel:
                stub = nm_pb2_grpc.NMServerStub(channel)
                response = stub.TerminateJob(terminate_request)
        return None

    def get_metrics(
        self,
        job_id_list: List[int],
        ipaddr_list: List[str],
        if_simulation: List[bool],
        round_duration: int,
        active_job_dict: dict,
    ) -> dict:
        """
        Given a job ID list fetch metrics from all the node managers
        job_id_list : List of Job ID's
        ipaddr_list : List of corresponding Job ID's
        if_simulation: List of boolean telling if the job is simulation or not
        round_duration: Represents the round duration
        active_job_dict: Active jobs dictionary
        #CAUTION: In case simulation we modify some of the parameters in place.
        """
        # TODO: Multi-thread this

        # acclerate = lambda x: x/1.5
        
        metric_data_dict = dict()
        for idx, job_id in enumerate(job_id_list):
            ipaddr_to_query = ipaddr_list[idx]
            if_sim = if_simulation[idx]
            job_exit = False
            if not if_sim:
                # added tracking
                previous_metric = active_job_dict[job_id]["tracked_metrics"]
                metric_data_dict[job_id] = previous_metric
                for ipaddr in ipaddr_to_query:
                    ipaddr = f"{ipaddr}:{self.rpc_port}"
                    metric_request = rm_pb2.JsonResponse()
                    metric_request.response = json.dumps({"Job_ID": job_id})
                    with grpc.insecure_channel(ipaddr) as channel:
                        stub = nm_pb2_grpc.NMServerStub(channel)
                        response = stub.GetMetrics(metric_request)
                    metric_data = json.loads(response.response)
                    # make sure we update and not overwrite
                    if job_id in metric_data_dict:
                        for key in metric_data:
                            if key == "attained_service":
                                metric_data_dict[job_id][key] += metric_data[key]
                            if key == "per_iter_time":
                                # average key
                                if key in metric_data_dict[job_id]:
                                    metric_data_dict[job_id][key] = (
                                        metric_data_dict[job_id][key] + metric_data[key]
                                    ) / 2
                                else:
                                    pass
                            if key == "iter_num":
                                metric_data_dict[job_id][key] += metric_data[key]

                    else:
                        metric_data_dict[job_id] = metric_data

                    # 去除 synergy 相关的无效输出
                    # Same job ids can be running at multiple ip addr
            else:
                # # this is a simulation
                # # profile scaling by number of GPUs
                # # 确保只初始化一次
            
                if active_job_dict[job_id]["previously_launched"] == False:
                    active_job_dict[job_id]["job_launched_first_time"] = True
                if active_job_dict[job_id]["previously_launched"] == True:
                    active_job_dict[job_id]["job_launched_first_time"] = False

                active_job_dict[job_id]["previously_launched"] = True
                job = active_job_dict[job_id]
                
                # 获取 GPU 数量（数值），确保是整数类型
                num_gpus = active_job_dict[job_id]["num_GPUs_allocated"]
                
                # 如果 num_GPUs 为 0，作业没有运行，不应该有迭代进度
                if num_gpus == 0:
                    # 作业没有 GPU，不更新迭代进度，只更新 attained_service（为0）
                    attained_service = (
                        active_job_dict[job_id]["tracked_metrics"]["attained_service"]
                        + 0  # 没有 GPU，不增加 attained_service
                    )
                    metric_data_dict[job_id] = {
                        "attained_service": attained_service,
                        "per_iter_time": active_job_dict[job_id]["job_iteration_time"],
                    }
                    continue
                
                try:
                    # 获取 tput（从字典形式的 job 中）
                    tput = get_tput_from_job_dict(job, job.get("cpu_allocated"), job.get("mem_allocated"))
                    if tput is None or tput <= 0:
                        # 如果无法获取 tput，使用默认值 1.0
                        tput = 1.0
                    # 获取基准迭代时间（从模型获取的迭代时间，在 add_synergy_profile 中设置）
                    # 如果 iter_is_duration 为 True，job_total_iteration 已经在 add_synergy_profile 中
                    # 从持续时间转换为迭代次数：job_total_iteration = int(duration / job_iteration_time)
                    base_iteration_time = job.get("iter_time_base")
                    
                    # 实际迭代时间 = 基准迭代时间 / (tput * synergy_speedup)
                    # synergy_speedup 已经在 add_synergy_profile 中根据资源分配调整过了
                    actual_iteration_time = base_iteration_time / (tput)
                    # 计算本轮每个 GPU 能完成的迭代数
                    # 每轮完成的迭代数 = round_duration / actual_iteration_time
                    iterations_per_gpu_in_round = round_duration / actual_iteration_time 
                    # 对于多 GPU：每个 GPU 并行处理，所以总迭代数 = 单 GPU 迭代数 * GPU 数量
                    total_iterations_in_round = iterations_per_gpu_in_round * num_gpus
                    
                    # attained_service 应该考虑 GPU 数量：如果有 N 个 GPU，每轮累加 round_duration * N
                    # 这与 attained_service_scheduler 的计算方式一致
                    attained_service = (
                        active_job_dict[job_id]["tracked_metrics"]["attained_service"]
                        + round_duration * num_gpus
                    )

                    # 实际迭代时间（考虑 tput 和 synergy_speedup 影响）
                    per_iteration_time = actual_iteration_time

                    # 新的总迭代次数 = 本轮完成的迭代次数 + 历史已经完成的迭代次数
                    job_executed_iteration = active_job_dict[job_id].get("job_executed_iteration")
                    total_iteration_achieved = (
                        total_iterations_in_round
                        + job_executed_iteration
                    )
                    
                    # CAUTION: In place update
                    # TODO: Clean this part of update
                    active_job_dict[job_id][
                        "job_executed_iteration"
                    ] = total_iteration_achieved
                    active_job_dict[job_id][
                        "job_time_remaining"
                    ] = (active_job_dict[job_id]["job_total_iteration"] - total_iteration_achieved)*base_iteration_time
                    if (total_iteration_achieved >= active_job_dict[job_id]["job_total_iteration"]):
                        job_exit = True

                    if job_exit == True:
                        metric_data_dict[job_id] = {
                            "attained_service": attained_service,
                            "per_iter_time": per_iteration_time,
                            "job_exit": True,
                        }
                    else:
                        metric_data_dict[job_id] = {
                            "attained_service": attained_service,
                            "per_iter_time": per_iteration_time,
                        }
                except Exception as e:
                    # 如果计算过程中出现异常，使用默认值，确保至少返回基本的 metric 数据
                    import logging
                    logging.warning(f"Error calculating metrics for job {job_id}: {e}")
                    # 使用默认值返回 metric，避免返回空字典
                    metric_data_dict[job_id] = {
                        "attained_service": active_job_dict[job_id]["tracked_metrics"].get("attained_service", 0),
                        "per_iter_time": job.get("job_iteration_time", 1.0),
                    }

        return metric_data_dict
