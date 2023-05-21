import os
import matplotlib.pyplot as plt
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--dataset_folder", required=True, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    dataset_folder = args.dataset_folder
    envs = os.listdir(dataset_folder)
    envs.sort()
    for env in envs:
        env_folder = os.path.join(dataset_folder, env)
        data_folder = os.path.join(env_folder, "data")
        datapoints = os.listdir(data_folder)
        datapoints.sort()
        for datapoint in datapoints:
            datapoint_file = os.path.join(data_folder, datapoint)
            dtp = np.load(datapoint_file)
            image = dtp["image"]
            label = dtp["label"]
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            ax1.set_title(f"{env} - {datapoint}")
            plt.sca(ax1)
            plt.imshow(image)
            plt.sca(ax2)
            plt.plot(label)
            plt.show()


if __name__ == "__main__":
    main()
