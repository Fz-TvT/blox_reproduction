from .scheduler_policy import SchedulingPolicy
import pandas as pd
from operator import getitem

from typing import Optional
from placement.placement import find_num_free_GPUs

class Synergy_fifo(SchedulingPolicy):
    """
    Implements Fifo Scheduler+synergy
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
        """
        Schedules job based on input.
        Args:
            job_dict : Original Job dict which we have sort of maintained
            node_info: Same dict as received from the node register.
            gpu_df: Contains GPU dataframe.

        Returns:
                "order_job" : Mandatory key, list of dicts of jobs in the
                                 in the order they are supposed to run.
                "run_all_jobs": Some scheduler will only output the jobs to
                                    run which will fit on the GPU or expecting
                                    to perform over subscription. While some
                                    will just sort the value in the order and
                                    return the whole job sorted back. (I am not
                                    sure if we need this)
                Per Job key optional keys:
                placement_locations : If the scheduler is making placement decisions too.
                Like Gandiva does, we expect them to add a additional key in
                case of the dictionary. with probable places where we should
                place the jobs.

                placement_preference: Further each job could have a placement
                preference.
                Like in case of tiresias. Therefore if the scheduler wants to provide a
                placement they can insert a key "placement_preference" for each
                job. If we do not find this key. We perform placement by our
                case. There are two kinds of placement we support

                additional_keys:
                In case the user writes a custom placement policy. In that case
                they can update custom metrics and pass them to the placement
                policy. From the scheduler
                And read it as they like. More on this case.
        """
        ##Synergy-tune
        free_gpus = find_num_free_GPUs(gpu_df) ##剩余GPU数量
        sorted_job_order = sorted(
            job_dict.items(), key=lambda x: (x[1]["job_priority"], x[1]["submit_time"])
        )
        jobs_this_round = sorted_job_order
        for job in sorted_job_order:
            deficit_num = job[1]["job_gpu_demand"] - job[1].get("num_GPUs", 0)
            if deficit_num > 0 and free_gpus - deficit_num >= 0:
                jobs_this_round.append(job)
                free_gpus -= deficit_num
            elif deficit_num > 0 and free_gpus > 0:
                jobs_this_round.append(job)
                free_gpus =0
            if job[1]["time_since_scheduled"] > 100 * 3600:
                job[1]["job_priority"] = 1
        
        schedule_info = dict()
        schedule_info["jobs_this_round"] = jobs_this_round
        schedule_info["job_order"] = sorted_job_order
        schedule_info["run_all_jobs"] = False

        return schedule_info
