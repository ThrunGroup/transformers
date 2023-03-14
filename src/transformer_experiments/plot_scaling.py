import matplotlib.pyplot as plt
import glob
import os
from pandas import read_csv
from typing import List
from collections import defaultdict

from evaluate_models import inference_perplexity
from utils.constants import WIKITEXT2, GPT2, GPT2_MEDIUM, SVD, PCA


def perplexity_scaling_plot(dataset: str = WIKITEXT2, acceleration_list: List[str]= ["None", SVD]):
    curr_dir = os.path.dirname(__file__)
    plot_dir = os.path.join(curr_dir, "evaluation_logs", f"{dataset}*", "*")
    csv_files = glob.glob(plot_dir)
    accelerator_data_dict = defaultdict(lambda: [[], []])

    for csv_file in csv_files:
        data = read_csv(csv_file)
        for plot_idx in range(len(acceleration_list)):
            accelerator = acceleration_list[plot_idx]
            if accelerator in csv_file:
                accelerator_data_dict[accelerator][0].extend(data["inference_time"])
                accelerator_data_dict[accelerator][1].extend(data["perplexity"])

    for accelerator, plot_data in accelerator_data_dict.items():
        print(accelerator, plot_data)
        plt.plot(
            plot_data[0], plot_data[1], 'o--', label=accelerator
        )
    plt.title(f"{dataset}")
    plt.xlabel("Inference time per sample(s)")
    plt.ylabel("Perplexity")
    plt.yscale("symlog")
    plt.legend(loc="upper right")
    plt.show()



if __name__ == "__main__":
    inference_perplexity([GPT2, GPT2_MEDIUM], [None, SVD])
    perplexity_scaling_plot(acceleration_list=["None", "SVD"])
