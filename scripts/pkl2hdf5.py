"""Runs over dataset and stores every pkl file into a h5 file."""

from multiprocessing import Pool
import pdb
import h5py
import os
import torch
import pickle
import argparse
import blosc
from re import sub
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial

from core.utils.pytorch_utils import ten2ar
from core.configs.default_data_configs.babyai import *


def convert_to_hdf5(filenames, args, device):
    total_trajs, max_seq_len = 0, 0

    dataset_name = os.path.basename(os.path.dirname(filenames[0]))
    hdf5_dir = os.path.join(args.outfolder, dataset_name)

    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)

    for file in filenames:
        print(f"Current: {file}")

        with open(file, "rb") as f:
            datas = pickle.load(f)

        for n, data in enumerate(tqdm(datas)):
            (
                mission,
                obs,
                directions,
                actions,
                subtask_completes,
                subtasks,
                attributes,
            ) = data

            obs = np.asarray(blosc.unpack_array(obs), dtype=np.uint8)
            seq_len = len(obs)

            if seq_len < args.min_seq_len:
                continue

            total_trajs += 1

            hdf5_f = os.path.join(hdf5_dir, f"{n}.h5")

            with h5py.File(hdf5_f, "w") as F:
                F["traj_per_file"] = 1
                F["traj0/mission"] = mission
                F["traj0/images"] = obs
                F["traj0/actions"] = np.asarray(actions)
                F["traj0/directions"] = np.asarray(directions)
                F["traj0/pad_mask"] = np.ones((seq_len,))

                # if args.render_images:
                #     F["traj0/images_vis"] = get_obs_render(blosc.unpack_array(obs))
                F["traj0/subtask_completes"] = subtask_completes.astype(np.float64)

                num_attributes = attributes.shape[1]
                semantic_attributes = np.zeros(
                    (seq_len, 2, num_attributes)
                )  # 2 because there are 2 objects

                i = 0
                for t in range(seq_len):
                    semantic_attributes[t] = attributes[i : i + 2]

                    if actions[t] == "done":
                        i += 1

                F["traj0/semantic_attributes"] = semantic_attributes

        print(f"Total: {total_trajs}")


if __name__ == "__main__":
    N_THREADS = 6

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infolder",
        required=True,
        help="Path the pickle files containing training trajectories",
    )
    parser.add_argument(
        "--outfolder", required=True, help="Path the output folder to save HDF5 files"
    )
    parser.add_argument(
        "--render_images",
        default=False,
        action="store_true",
        help="Adds human viewable images to h5 files (file sizes will be large with this option)",
    )
    parser.add_argument(
        "--min_seq_len", default=11, type=int, help="Minimum lenght of sequence"
    )
    args = parser.parse_args()

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    # Collect all pkl files
    filenames = []
    for n, path in enumerate(Path(args.infolder).rglob("demo")):
        filenames.append(os.path.join(path))
    n_files = len(filenames)
    print("\nFound", n_files, " files")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if n_files > 1:
    #     chunk_size = int(np.floor(n_files / N_THREADS))
    #     filename_chunks = [
    #         filenames[i : i + chunk_size] for i in range(0, n_files, chunk_size)
    #     ]

    #     p = Pool(N_THREADS)
    #     fn = partial(convert_to_hdf5, args=args, device=device)
    #     p.map(fn, filename_chunks)
    # else:
    convert_to_hdf5(filenames, args, device)
