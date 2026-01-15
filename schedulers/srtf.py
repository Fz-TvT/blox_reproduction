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
        pass

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:

        for job in job_dict:
            job_dict[job]["time_remaining"] = (
                job_dict[job]["job_total_iteration"]
                - job_dict[job]["job_executed_iteration"]
            )
        sorted_job_order = sorted(
            job_dict.items(),
            key=lambda x: (x[1]["job_priority"], x[1]["time_remaining"],x[1]["submit_time"])
        )
        for job in sorted_job_order:
            if job[1]["time_since_scheduled"] > 1000 * 3600:
                job[1]["job_priority"] = 1
        
        schedule_info = dict()
        schedule_info["job_order"] = sorted_job_order
        schedule_info["run_all_jobs"] = False

        return schedule_info
