#!/usr/bin/env python3
"""
独立脚本：绘制 FIFO 调度方法的 CDF 图像
基于 simulator.py 中的 _plot_graphs 函数
"""

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import argparse

# 设置 matplotlib 参数
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_fifo_cdf(
    job_stats_file,
    output_dir="./plots",
    exp_prefix="test",
    job_id_range=(0, 1000),
    load=9.0,
    scheduler="Fifo",
    acceptance_policy="AcceptAll",
):
    """
    绘制 FIFO 调度方法的 CDF 图像
    
    Args:
        job_stats_file: job_stats.json 文件路径
        output_dir: 输出目录
        exp_prefix: 实验前缀
        job_id_range: 作业 ID 范围 (start, end)
        load: 负载值
        scheduler: 调度器名称
        acceptance_policy: 接受策略名称
    """
    # 读取作业统计数据
    print(f"Reading job stats from: {job_stats_file}")
    with open(job_stats_file, "r") as fin:
        data_job = json.load(fin)
    
    # 计算 JCT (Job Completion Time)
    # data_job 的格式是 {job_id: [start_time, end_time], ...}
    vals = data_job.values()
    jct_vals = [val[1] - val[0] for val in vals]  # end_time - start_time
    
    # 转换为小时（如果需要）
    # 假设时间单位是秒，转换为小时
    jct_vals_hours = [val / 3600.0 for val in jct_vals]
    
    # 排序以计算 CDF
    jct_vals_hours = sorted(jct_vals_hours)
    
    # 计算 CDF 值
    plot_y_val = list()
    for idx, val in enumerate(jct_vals_hours):
        plot_y_val.append(float(idx) / len(jct_vals_hours))
    
    # 绘制图像
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 3)
    
    ax1.plot(jct_vals_hours, plot_y_val, label=scheduler, linewidth=2)
    ax1.set_xlabel("Time (hrs)", fontsize=12)
    ax1.set_ylabel("CDF", fontsize=12)
    ax1.set_title(f"{scheduler}_load_{load}_CDF", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 创建输出目录
    write_folder = os.path.join(
        output_dir, f"{exp_prefix}_{job_id_range[0]}_{job_id_range[1]}_cdf_load_{load}"
    )
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    
    # 保存图像
    output_file = os.path.join(write_folder, f"load_{load}_cdf.pdf")
    plt.savefig(
        output_file,
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )
    print(f"CDF plot saved to: {output_file}")
    
    # 同时保存 PNG 格式（可选）
    output_file_png = os.path.join(write_folder, f"load_{load}_cdf.png")
    plt.savefig(
        output_file_png,
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"CDF plot (PNG) saved to: {output_file_png}")
    
    plt.close()
    
    # 打印统计信息
    print(f"\nStatistics:")
    print(f"  Total jobs: {len(jct_vals_hours)}")
    print(f"  Min JCT: {min(jct_vals_hours):.2f} hrs")
    print(f"  Max JCT: {max(jct_vals_hours):.2f} hrs")
    print(f"  Mean JCT: {sum(jct_vals_hours) / len(jct_vals_hours):.2f} hrs")
    print(f"  Median JCT: {jct_vals_hours[len(jct_vals_hours) // 2]:.2f} hrs")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="绘制 FIFO 调度方法的 CDF 图像"
    )
    parser.add_argument(
        "--job-stats-file",
        type=str,
        required=True,
        help="job_stats.json 文件路径（例如: result/test_0_1000_Fifo_AcceptAll_load_9.0_job_stats.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="输出目录（默认: ./plots）",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="test",
        help="实验前缀（默认: test）",
    )
    parser.add_argument(
        "--start-job-id",
        type=int,
        default=0,
        help="起始作业 ID（默认: 0）",
    )
    parser.add_argument(
        "--end-job-id",
        type=int,
        default=1000,
        help="结束作业 ID（默认: 1000）",
    )
    parser.add_argument(
        "--load",
        type=float,
        default=9.0,
        help="负载值（默认: 9.0）",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="Fifo",
        help="调度器名称（默认: Fifo）",
    )
    parser.add_argument(
        "--acceptance-policy",
        type=str,
        default="AcceptAll",
        help="接受策略名称（默认: AcceptAll）",
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.job_stats_file):
        print(f"Error: File not found: {args.job_stats_file}")
        return
    
    # 绘制 CDF
    plot_fifo_cdf(
        job_stats_file=args.job_stats_file,
        output_dir=args.output_dir,
        exp_prefix=args.exp_prefix,
        job_id_range=(args.start_job_id, args.end_job_id),
        load=args.load,
        scheduler=args.scheduler,
        acceptance_policy=args.acceptance_policy,
    )


if __name__ == "__main__":
    main()

