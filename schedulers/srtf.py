from .scheduler_policy import SchedulingPolicy
import pandas as pd
from operator import getitem

from typing import Optional


class Srtf(SchedulingPolicy):
    """
    Implements Shortest remaining time first
    """

    def __init__(self, args):
        """
        Use this to hold any extra state the scheduler wants to hold
        """
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.fair = True  # 默认使用公平共享模式
        pass


    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        """
        SRTF调度器按最短剩余时间优先调度
        排序规则：优先级（降序） -> 剩余时长（升序） -> 到达时间（升序）
        """
        # 计算每个作业的剩余时长

        # 检查长时间未执行的作业并提升优先级
        for job_id in job_dict:
            time_since_scheduled = job_dict[job_id].get("time_since_scheduled", 0)
            # 如果作业超过1000小时未执行，提升优先级
            if time_since_scheduled > 100 * 3600:
                job_dict[job_id]["job_priority"] = 1
        
        # 按优先级（降序，高优先级在前）、剩余时长（升序）、到达时间（升序）排序
        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (
                x[1].get("job_priority"),  # 负号表示降序，高优先级在前
                x[1].get("time_remaining")
            )
        )
        
        schedule_info = dict()
        schedule_info["job_order"] = sorted_job_order
        schedule_info["run_all_jobs"] = False
        return schedule_info
