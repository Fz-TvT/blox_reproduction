from .scheduler_policy import SchedulingPolicy
import pandas as pd
from operator import getitem
import os
from typing import Optional
import heapq  


class Optimus(SchedulingPolicy):
    """
    Implements Shortest remaining time first
    """

    def __init__(self, args,job_state):
        """
        Use this to hold any extra state the scheduler wants to hold
        """
        self.metric_to_track = ["per_iter_time", "attained_service","num_allocated_gpus"]
        self.default_metric_value = [0, 0]
        self.job_state = job_state
        # 设置单个作业的最大 GPU 数量上限（默认 8，可通过环境变量配置）
        self.max_gpus_per_job = int(os.environ.get("OPTIMUS_MAX_GPUS_PER_JOB", 8))
        # self.marginal_utility = {'resnet50':
        pass
    def GPU_speed(self,x):
        return x/(3+4*x)
    def compute_marginal_gain(self, job_dict,job):
            time_before_add_gpu = job_dict[job]["time_remaining"]/(self.GPU_speed(job_dict[job]["num_GPUs"]))
            time_after_add_gpu = job_dict[job]["time_remaining"]/self.GPU_speed(job_dict[job]["num_GPUs"] + 1)
            marginal_gain = time_before_add_gpu-time_after_add_gpu
            # print("marginal gain",marginal_gain)
            return marginal_gain
    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        """
        Optimus Scheduler. Assuming the job closest to convergence is known. Similar to Pollux's, Optimus Oracle
        """
        heap=[]
        for job in job_dict:
            job_dict[job]["time_remaining"] = (
                job_dict[job]["job_iteration_time"]
                * job_dict[job]["job_total_iteration"]
                - job_dict[job]["tracked_metrics"]["attained_service"]
            )

        schedule_info = dict()
        schedule_info["run_all_jobs"] = False
        # NOTE: Borrowed from https://github.com/kzhang28/Optimus/blob/cea8c8bb39da493b5a45cdce625fe3c225c1793c/measurement/training-speed/measure-speed-placements.py#L170
        jobs_by_submit = sorted(job_dict.items(), key=lambda x: x[1]["submit_time"])
        schedule_info["job_order"] = jobs_by_submit
        
        # 获取当前可用 GPU 数量
        free_gpus = self._get_free_gpus(gpu_df)
        
        # 3. 按 FIFO 顺序，每个作业至少分配 1 GPU（如果还有资源）
        for job_id, job in jobs_by_submit:
            if free_gpus > 0:
                job["num_GPUs"] = 1  # 至少 1 个
                free_gpus -= 1  # 更新可用 GPU 数量
            elif free_gpus<0:
                break
        
        # 在分配 GPU 后，更新 job_gpu_demand 和 total_gpus
        for job in job_dict:
            job_dict[job]["job_gpu_demand"] = job_dict[job]["num_GPUs"]
            job_dict[job]["total_gpus"] = os.environ["sched_load"]
            if job_dict[job]["time_since_scheduled"] > 10 * 3600:
                job_dict[job]["job_priority"] = 1
            
        # 计算边际收益并构建堆
        for job in job_dict:
            if job_dict[job]["num_GPUs"] > 0:  # 只为已分配 GPU 的作业计算边际收益
                marginal_gain = self.compute_marginal_gain(job_dict,job)
                heapq.heappush(heap,(-marginal_gain,job))
        
        # allocate more GPUs if there are GPUs left
        # the jobs are already stored in the order they will converge
        # 使用当前实际可用 GPU 数量（已减去初始分配的 GPU）
        while free_gpus > 0 and heap:
            # keep adding the number of GPUs needed by each job
            _,job = heapq.heappop(heap)
            
            # 检查是否超过单个作业的最大 GPU 数量上限
            # 上限可以是：1) 作业的原始需求 job_gpu_demand，2) 配置的最大值 max_gpus_per_job
            # max_allowed_gpus = max(
            #     job_dict[job].get("job_gpu_demand", 0),  # 作业的原始 GPU 需求
            #     self.max_gpus_per_job  # 配置的最大值
            # )
            # # 取两者中的较小值作为实际上限（更保守）
            # job_original_demand = job_dict[job].get("job_gpu_demand", 0)
            # if job_original_demand > 0:
            #     # 如果作业有原始需求，使用原始需求和配置上限的较小值
            #     max_allowed_gpus = min(job_original_demand, self.max_gpus_per_job)
            # else:
            #     # 如果没有原始需求，使用配置的上限
            #     max_allowed_gpus = self.max_gpus_per_job
            
            # # 如果当前 GPU 数量已达到上限，不再分配更多 GPU
            # if job_dict[job]["num_GPUs"] >= max_allowed_gpus:
            #     continue  # 跳过这个作业，不再分配 GPU
            
            job_dict[job]["num_GPUs"] += 1
            free_gpus = free_gpus - 1
            # print(free_gpus)
            marginal_gain = self.compute_marginal_gain(job_dict,job)
            if marginal_gain>0:
                heapq.heappush(heap,(-marginal_gain,job))
        # schedule_info["gpu_allocation"] = {
        # job_id: job_dict[job_id]["num_GPUs"] for job_id in job_dict
        # }
        # print("scheduled",schedule_info)
        return schedule_info

    def _total_gpu_demand(self, sorted_job_order):
        """
        Sum up the GPU demand
        """
        gpu_demand = 0
        for jobs in sorted_job_order:
            gpu_demand += int(jobs[1]["job_gpu_demand"])
        return gpu_demand

    def _get_free_gpus(self, gpu_df: pd.DataFrame) -> dict:
        """
        Find the nodeID's which have free GPUs
        Args:
        gpu_df : DataFrame consisting of information about GPUs
        Returns:
        int
        """
        free_gpus = (
            gpu_df.loc[gpu_df["IN_USE"] == False]
            .groupby("Node_ID")["GPU_ID"]
            .apply(list)
            .to_dict()
        )
        number_free_gpus = sum([len(free_gpus[x]) for x in free_gpus])
        return number_free_gpus
