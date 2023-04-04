import matplotlib.pyplot as plt
import glob
import os
from pandas import read_csv
from typing import List
from collections import defaultdict

from utils.constants import GPT2, GPT2_MEDIUM, SVD, BILLSUM


def perplexity_scaling_plot(dataset: str = BILLSUM,
                            model_name: str = GPT2,
                            log_dir:str = "wikitext-2-raw-v1-n_20-s_1024",
                            acceleration_list: List[str] = ["None", SVD],
                            metric: str= "eval_rouge1",
                            is_logspace_x: bool = False,
                            is_logspace_y: bool=True,) -> None:
    """
    Print scaling plot where x-axis indicates inference time and y-axis indicates evaluation loss.

    :param dataset: Name of dataset
    :param acceleration_list: List of acceleration techniques applied to the mlp model which we want to plot
    :param metric: Evaluation metric
    :param is_logspace_x: Whether to apply log function to x-axis
    :param is_logspace_y: Whether to apply log to y-axis
    """
    curr_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(curr_dir, log_dir, "*")
    csv_files = glob.glob(plot_dir)
    accelerator_data_dict = defaultdict(lambda: [[], []])
    print(plot_dir)
    print(csv_files)

    for csv_file in csv_files:
        if model_name not in csv_file:
            continue
        data = read_csv(csv_file)
        for plot_idx in range(len(acceleration_list)):
            accelerator = acceleration_list[plot_idx]
            if accelerator in csv_file:
                accelerator_data_dict[accelerator][0].extend(data["inference_time"])
                accelerator_data_dict[accelerator][1].extend(data[metric])

    for accelerator, plot_data in accelerator_data_dict.items():
        plot_data[0], plot_data[1] = zip(*sorted(zip(plot_data[0], plot_data[1])))
        plt.plot(
            plot_data[0], plot_data[1], 'o--', label=accelerator
        )
    plt.title(f"{dataset}")
    plt.xlabel("Inference time per sample(s)")
    plt.ylabel(metric)
    if is_logspace_x:
        plt.xscale("symlog")

    if is_logspace_y:
        plt.yscale("symlog")

    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    # inference_perplexity([GPT2, GPT2_MEDIUM], [None, SVD])
    log_dir = os.path.join("exp_logs")
    perplexity_scaling_plot(acceleration_list=["None", SVD],
                            model_name=GPT2,
                            log_dir=log_dir,
                            metric="eval_rouge1",
                            is_logspace_y=False,
                            dataset=BILLSUM)