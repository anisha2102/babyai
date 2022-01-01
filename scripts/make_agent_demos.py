#!/usr/bin/env python3
"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
from subprocess import Popen

import os
import time
import numpy as np
import blosc
import torch
import re
import copy
import json

from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from pprint import pprint

from babyai.levels.verifier import *
from gym_minigrid.wrappers import *
from core.utils.general_utils import AttrDict, create_exp_name
from core.configs.default_data_configs.babyai import *
import babyai
import babyai.utils as utils

# Parse arguments
logger = logging.getLogger(__name__)

# Set seed for all randomness sources
def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def save_video(fname, frames, fps=10.0):
    outfolder = "demo_videos"
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    path = os.path.join(outfolder, fname)

    def f(t):
        frame_length = len(frames)
        new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
        idx = min(int(t * new_fps), frame_length - 1)
        return frames[idx]

    video = mpy.VideoClip(f, duration=len(frames) / fps + 2)
    video.write_videofile(path, fps, verbose=False, logger=None)


def get_attribute(desc, action):
    color, obj_type, loc = desc.color, desc.type, desc.loc

    if color is None or obj_type is None:
        raise NotImplementedError

    color_label = color_to_indx_map[color]
    obj_type_label = obj_type_to_indx_map[obj_type]
    action_label = action_to_indx_map[action.lower()]
    attrs = np.array([color_label, obj_type_label, action_label])
    return attrs


def get_attributes(env, verifiers):
    attributes = []
    for verifier in verifiers:
        action = verifier.surface(env).split(" ")[0]

        if isinstance(verifier, PutNextInstr):
            attrs_1 = get_attribute(verifier.desc_move, action)
            attrs_2 = get_attribute(verifier.desc_fixed, action)
            attrs = np.stack([attrs_1, attrs_2])
        else:
            attrs_1 = get_attribute(verifier.desc, action)
            attrs_2 = np.array([0.0, 0.0, 0.0])
            attrs = np.stack([attrs_1, attrs_2])

        attributes.append(attrs)
    return np.concatenate(attributes)


def render(env):
    full_img = Resize((400, 400))(Image.fromarray(env.render(mode="rgb_array")))
    plt.imshow(full_img)
    plt.show()


def generate_demos(args, valid=False):
    utils.seed(args.seed)
    demos_path = args.demos
    # print(args.demos)

    # Generate environment
    env_kwargs = {
        "agent_init": args.agent_init,
        "task_obj_init": args.task_obj_init,
        "distractor_obj_init": args.distractor_obj_init,
        "subtasks": args.subtasks,
        "num_subtasks": args.num_subtasks,
        "sequential": args.sequential,
    }

    # Make environment
    env = gym.make(args.env, **env_kwargs)
    action_to_indx_map = {e.value: e.name for e in env.Actions}
    assert isinstance(env.instrs, CompositionalInstr)

    # Make agent
    agent = utils.load_agent(
        env, args.model, args.demos, "agent", args.argmax, args.env
    )

    demos = []
    checkpoint_time = time.time()
    just_crashed = False

    if valid:
        n_episodes = args.valid_episodes
    else:
        n_episodes = args.episodes

    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info(
                "reset the environment to find a mission that the bot can solve"
            )
            env.reset()
        else:
            env.seed(args.seed + len(demos))

        obs = env.reset()
        mission = obs["mission"]
        if args.visualize:
            print(f"Mission: {mission}")

        agent.on_reset()

        actions, observations, directions, subtask_completes, images = (
            [],
            [],
            [],
            [],
            [],
        )

        # Get ground truth skill labels
        verifiers = env.instrs.instrs
        attributes = get_attributes(env, verifiers)
        subtasks = [verifier.surface(env) for verifier in verifiers]

        overall_mission = env.instrs
        overall_mission.reset_verifier(env)
        status = overall_mission.verify(action=env.Actions.done)

        prev_num_subtasks_completed = 0

        if args.visualize:
            render(env)

        try:
            while not done:
                if args.save_video:
                    images.append(env.render())

                action = agent.act(obs)["action"]
                if isinstance(action, torch.Tensor):
                    action = action.item()

                new_obs, reward, done, _ = env.step(action, verify=False)

                status = overall_mission.verify(action)
                subtask_status = list(overall_mission.dones.values())

                # After every env step, log which subtasks have been completed
                subtask_complete = [
                    1 if status == "success" else 0 for status in subtask_status
                ]
                subtask_completes.append(subtask_complete)

                # Set done to be true if task is completed
                if status == "success":
                    done = True
                    reward = env._reward()

                    # Make sure that all the subtasks are completed
                    assert sum(subtask_complete) == args.num_subtasks
                elif status == "failure":
                    done = True
                    reward = 0

                if args.visualize:
                    print(
                        f"Status: {status} | Subtasks: {subtask_complete} | Action: {action_to_indx_map[action]} | Reward: {reward}"
                    )
                    render(env)

                agent.analyze_feedback(reward, done)

                actions.append(action)
                observations.append(obs["image"])
                directions.append(obs["direction"])

                # Add a pause for when a new subtask is completed
                num_subtasks_completed = sum(subtask_complete)
                if done or num_subtasks_completed != prev_num_subtasks_completed:
                    actions.append(env.Actions.done)
                    observations.append(new_obs["image"])
                    directions.append(new_obs["direction"])
                    subtask_completes.append(subtask_complete)
                    prev_num_subtasks_completed = num_subtasks_completed

                if done and args.save_video:
                    images.append(env.render())
                    save_video("check" + str(len(demos)) + ".mp4", np.array(images))

                obs = new_obs

            if reward > 0 and (
                args.filter_steps == 0 or len(images) <= args.filter_steps
            ):
                demos.append(
                    (
                        mission,
                        blosc.pack_array(np.array(observations)),
                        directions,
                        actions,
                        np.array(subtask_completes),
                        subtasks,
                        attributes,
                    )
                )
                just_crashed = False

            if reward == 0:
                if args.on_exception == "crash":
                    raise Exception(
                        "mission failed, the seed is {}".format(args.seed + len(demos))
                    )
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == "crash":
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        if len(demos) and len(demos) % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info(
                "demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                    len(demos) - 1, demos_per_second, to_go
                )
            )
            checkpoint_time = now

        # Save demonstrations
        if (
            args.save_interval > 0
            and len(demos) < n_episodes
            and len(demos) % args.save_interval == 0
        ):
            logger.info("Saving demos...", demos_path)
            utils.save_demos(demos, demos_path)
            logger.info("{} demos saved".format(len(demos)))
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])

    # Save demonstrations
    logger.info("Saving demos...")
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])


def generate_demos_cluster():
    demos_per_job = args.episodes // args.jobs

    env_kwargs = {
        "agent_init": args.agent_init,
        "task_obj_init": args.task_obj_init,
        "distractor_obj_init": args.distractor_obj_init,
        "subtasks": args.subtasks,
        "num_subtasks": args.num_subtasks,
    }
    demo_name = create_exp_name(env_kwargs, keys_to_include=list(env_kwargs.keys()))
    demos_path = os.path.join(utils.storage_dir(), "demos", demo_name, "demo")

    job_demo_names = [
        os.path.realpath(demos_path + ".shard{}".format(i)) for i in range(args.jobs)
    ]
    for demo_name in job_demo_names:
        if os.path.exists(demo_name):
            os.remove(demo_name)

    command = [args.job_script]
    command += sys.argv[1:]

    child_processes = []
    for i in range(args.jobs):
        cmd_i = list(
            map(
                str,
                command
                + ["--seed", args.seed + i * demos_per_job]
                + ["--demos", job_demo_names[i]]
                + ["--episodes", demos_per_job]
                + ["--jobs", 0]
                + ["--valid-episodes", 0],
            )
        )
        logger.info("LAUNCH COMMAND")
        logger.info(cmd_i)
        p = subprocess.Popen(cmd_i)
        child_processes.append(p)

    for cp in child_processes:
        cp.wait()

    job_demos = [None] * args.jobs
    while True:
        jobs_done = 0
        for i in range(args.jobs):
            if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
                try:
                    logger.info("Trying to load shard {}".format(i))
                    demo_path = os.path.join(
                        utils.storage_dir(), "demos", job_demo_names[i]
                    )
                    job_demos[i] = utils.load_demos(demo_path)
                    logger.info(
                        "{} demos ready in shard {}".format(len(job_demos[i]), i)
                    )
                except Exception:
                    logger.exception("Failed to load the shard")
            if job_demos[i] and len(job_demos[i]) == demos_per_job:
                jobs_done += 1
        logger.info("{} out of {} shards done".format(jobs_done, args.jobs))
        if jobs_done == args.jobs:
            break
        logger.info("sleep for 60 seconds")
        time.sleep(60)

    # Training demos
    all_demos = []
    for demos in job_demos:
        all_demos.extend(demos)
    utils.save_demos(all_demos, demos_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    parser.add_argument(
        "--model", default="BOT", help="name of the trained model (REQUIRED)"
    )
    parser.add_argument(
        "--demos",
        default=None,
        help="path to save demonstrations (based on --model and --origin by default)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="number of episodes to generate demonstrations for",
    )
    parser.add_argument(
        "--valid-episodes",
        type=int,
        default=512,
        help="number of validation episodes to generate demonstrations for",
    )
    parser.add_argument("--seed", type=int, default=0, help="start random seed")
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="action with highest probability is selected",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="interval between progress reports",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10000,
        help="interval between demonstrations saving",
    )
    parser.add_argument(
        "--filter-steps",
        type=int,
        default=0,
        help="filter out demos with number of steps more than filter-steps",
    )
    parser.add_argument(
        "--on-exception",
        type=str,
        default="warn",
        choices=("warn", "crash"),
        help="How to handle exceptions during demo generation",
    )

    parser.add_argument(
        "--job-script",
        type=str,
        default=None,
        help="The script that launches make_agent_demos.py at a cluster.",
    )
    parser.add_argument(
        "--jobs", type=int, default=0, help="Split generation in that many jobs"
    )

    parser.add_argument("--agent-init", type=str, default="fixed", help="")
    parser.add_argument("--task-obj-init", type=str, default="fixed", help="")
    parser.add_argument("--distractor-obj-init", type=str, default="fixed", help="")
    parser.add_argument("--visualize", type=int, default=0, help="")
    parser.add_argument("--num-subtasks", type=int, default=3, help="")
    parser.add_argument("--subtasks", type=str, default=None, nargs="+", help="")
    parser.add_argument("--sequential", type=int, default=0, help="")
    parser.add_argument("--task", type=str, default="", help="")
    parser.add_argument(
        "--save_video", action="store_true", default=False, help="Save demo videos"
    )
    parser.add_argument("--screen-sz", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s: %(levelname)s: %(message)s")
    logger.info(args)

    # Training demos
    if args.jobs == 0:
        if not args.demos:
            env_kwargs = {
                "agent_init": args.agent_init,
                "task_obj_init": args.task_obj_init,
                "distractor_obj_init": args.distractor_obj_init,
                "subtasks": args.subtasks,
                "num_subtasks": args.num_subtasks,
                "sequential": args.sequential,
            }
            demo_name = create_exp_name(
                env_kwargs, keys_to_include=list(env_kwargs.keys())
            )
            demos_path = os.path.join(utils.storage_dir(), "demos", demo_name, "demo")
            args.demos = demos_path
        generate_demos(args)
    else:
        generate_demos_cluster()

    # Validation demos
    if args.valid_episodes:
        generate_demos(args, valid=True)
