import os
import sys
import json
import grpc
import logging
from typing import List
from concurrent import futures
from collections import defaultdict

sys.path.append(os.path.join((__file__), "./grpc_stubs"))
import nm_pb2
import nm_pb2_grpc

import rm_pb2

import simulator_pb2
import simulator_pb2_grpc


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
        print("In terminate simulation")
        print("job id list {}".format(job_id_list))
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

            print("Called Terminate for ip address {}".format(send_ip_address))
            print("Terminating job ids {}".format(send_request_dict[send_ip_address]))
            print(
                "IP_addr_terminate {}".format(other_ip_address_to_send[send_ip_address])
            )

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

                    # Add previous data
                    print(
                        "Job id {} Aggregated Metric Data {}".format(
                            job_id, metric_data
                        )
                    )
                    # Same job ids can be running at multiple ip addr
            else:
                # this is a simulation
                # profile scaling by number of GPUs
                # 确保只初始化一次
                if not hasattr(self, 'optimus_scale_by_gpus'):
                    self.optimus_scale_by_gpus = {
                        "0": 0.0,    # 0 GPU，无加速
                        "1": 1.0,    # 基准
                        "1.0": 1.0,    # 基准
                        "2": 1.95,  # ~98% 效率
                        "2.0": 1.95,  # ~98% 效率
                        "3": 2.85,  # ~95% 效率
                        "3.0": 2.85,  # ~95% 效率
                        "4": 3.75,  # ~94% 效率
                        "4.0": 3.75,  # ~94% 效率
                        "5": 4.60,  # ~92% 效率
                        "5.0": 4.60,  # ~92% 效率
                        "6": 5.40,  # ~90% 效率
                        "6.0": 5.40,  # ~90% 效率
                        "7": 6.15,  # ~88% 效率
                        "7.0": 6.15,  # ~88% 效率
                        "8": 6.85,  # ~86% 效率
                        "8.0": 6.85,  # ~86% 效率
                        "9": 7.50,  # ~83% 效率
                        "9.0": 7.50,  # ~83% 效率
                        "10.0": 8.10,  # ~80% 效率
                    }
                if active_job_dict[job_id]["previously_launched"] == False:
                    active_job_dict[job_id]["job_launched_first_time"] = True
                if active_job_dict[job_id]["previously_launched"] == True:
                    active_job_dict[job_id]["job_launched_first_time"] = False

                active_job_dict[job_id]["previously_launched"] = True

                # 获取 GPU 数量（数值）
                num_gpus = active_job_dict[job_id]["num_GPUs"]
                # print("job_id",job_id)
                # print("num_gpus",num_gpus)
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

                total_iterations_in_round = (
                    round_duration / active_job_dict[job_id]["job_iteration_time"]
                ) ##计算本轮能完成多少迭代
                
                # attained_service 应该考虑 GPU 数量：如果有 N 个 GPU，每轮累加 round_duration * N
                # 这与 attained_service_scheduler 的计算方式一致
                attained_service = (
                    active_job_dict[job_id]["tracked_metrics"]["attained_service"]
                    + round_duration * num_gpus
                )

                per_iteration_time = active_job_dict[job_id]["job_iteration_time"]

                # total_iteration_achieved = (
                #     total_iterations_in_round
                #     + active_job_dict[job_id]["job_executed_iteration"])
                #新的总迭代次数=本轮完成的迭代次数+历史已经完成的迭代次数
                num_gpus = active_job_dict[job_id]["num_GPUs"]
                # print("num_gpus",num_gpus)
                if num_gpus != "0":
                    scale_factor = self.optimus_scale_by_gpus.get(num_gpus)
                    total_iteration_achieved = (
                        total_iterations_in_round * num_gpus
                        + active_job_dict[job_id]["job_executed_iteration"]
                    )
                else:
                    total_iteration_achieved = active_job_dict[job_id]["job_executed_iteration"]
                
                # print("gpu",self.optimus_scale_by_gpus[str(active_job_dict[job_id]["job_gpu_demand"])])
                ##added for Pollux
                if os.environ["sched_policy"] == "Pollux":
                    if active_job_dict[job_id]["tracked_metrics"]["pollux_metrics"].completion_time is not None:
                        job_exit = True
                elif (
                    total_iteration_achieved
                    >= active_job_dict[job_id]["job_total_iteration"]
                ):
                    job_exit = True

                # CAUTION: In place update
                # TODO: Clean this part of update
                active_job_dict[job_id][
                    "job_executed_iteration"
                ] = total_iteration_achieved

                if job_exit == True:
                    metric_data_dict[job_id] = {
                        "attained_service": attained_service,
                        "per_iter_time": per_iteration_time,
                        "job_exit": True,
                    }
                if not job_exit:
                    metric_data_dict[job_id] = {
                        "attained_service": attained_service,
                        "per_iter_time": per_iteration_time,
                    }

        return metric_data_dict
