#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
from babyai.levels.verifier import *
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

import babyai.utils as utils
from gym_minigrid.minigrid import COLOR_NAMES, DIR_TO_VEC
from torchvision.transforms import Resize
from PIL import Image

from spirl.utils.general_utils import AttrDict
from spirl.rl.envs.babyai import BabyAIEnv
from spirl.configs.default_data_configs.babyai import *

from torchvision.transforms import Resize
from PIL import Image
import matplotlib.pyplot as plt
import moviepy.editor as mpy

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

def save_video(fname, frames, fps=10.):
    outfolder = "demo_videos"
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    path = os.path.join(outfolder, fname)
    def f(t):
        frame_length = len(frames)
        new_fps = 1./(1./fps + 1./frame_length)
        idx = min(int(t*new_fps), frame_length-1)
        return frames[idx]

    video = mpy.VideoClip(f, duration=len(frames)/fps+2)
    video.write_videofile(path, fps, verbose=False, logger=None)


def get_single_one_hot(desc, action):
    color, obj_type, loc = (
        desc.color,
        desc.type,
        desc.loc,
    )

    if color is None or obj_type is None:
        raise NotImplementedError

    obj_desc = [color, obj_type, loc]
    color_one_hot = np.zeros(len(COLOR_NAMES))
    color_one_hot[color_indx_map[color]] = 1

    obj_type_one_hot = np.zeros(len(OBJ_TYPES))
    obj_type_one_hot[obj_type_indx_map[obj_type]] = 1

    action = action.split(" ")[0]
    action_one_hot = np.zeros(len(ACTION_TYPES))
    action_one_hot[action_indx_map[action.lower()]] = 1

    return color_one_hot, obj_type_one_hot, action_one_hot, obj_desc


def get_attributes(env, verifiers):
    attributes = []

    for verifier in verifiers:
        action = verifier.surface(env).split(" ")[0]

        if isinstance(verifier, PutNextInstr):
            attr_1 = get_single_one_hot(verifier.desc_move, action)
            attr_2 = get_single_one_hot(verifier.desc_fixed, action)
            attributes.append(attr_1)
            attributes.append(attr_2)
        else:
            attr = get_single_one_hot(verifier.desc, action)
            attributes.append(attr)

    colors = [attr[0] for attr in attributes]
    obj_types = [attr[1] for attr in attributes]
    actions = [attr[2] for attr in attributes]
    obj_descs = [attr[3] for attr in attributes]

    return np.array(colors), np.array(obj_types), np.array(actions), obj_descs


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)
    # Generate environment
    env_config = AttrDict(
        reward_norm=1.0,
        screen_height=8,
        screen_width=8,
        level_name=args.env,
    )
    env = BabyAIEnv(env_config)
    env._set_env_kwargs(
        agent_init=args.agent_init,
        task_obj_init=args.task_obj_init,
        distractor_obj_init=args.distractor_obj_init,
        num_subtasks=args.num_subtasks,
        subtasks=args.subtasks,
        task_objs=args.task_objs,
    )
    env = env._env

    agent = utils.load_agent(
        env, args.model, args.demos, "agent", args.argmax, args.env
    )
    demos_path = utils.get_demos_path(args.demos, args.env, "agent", valid)

    demos = []

    checkpoint_time = time.time()

    just_crashed = False

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
            env.seed(seed + len(demos))
            # env.seed(seed + np.random.randint(100))
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        # verifiers = breakdown_verifiers(env, env.instrs)
        # if type(verifiers) != list:
        #     verifiers = [verifiers]

        assert isinstance(env.instrs, CompositionalInstr)

        verifiers = env.instrs.instrs
        attributes = get_attributes(env, verifiers)

        subtasks = [verifier.surface(env) for verifier in verifiers]
        subtask_complete = []

        overall_mission = copy.deepcopy(env.instrs)
        overall_mission.reset_verifier(env)

        if args.debug:
            full_img = Resize((400, 400))(Image.fromarray(env.render(mode="rgb_array")))
            plt.imshow(full_img)
            plt.show()

        try:
            if args.save_video:
                imgs = []

            subtask_completes = []

            while not done:
                if args.save_video:
                    imgs.append(env.render())
                action = agent.act(obs)["action"]
                if isinstance(action, torch.Tensor):
                    action = action.item()

                new_obs, reward, done, _ = env.step(action, verify=False)
                subtask_status = list(overall_mission.dones.values())
                subtask_complete = []
                for status in subtask_status[::-1]:
                    if status == "success":
                        subtask_complete.append(1)
                    else:
                        subtask_complete.append(0)
                subtask_completes.append(subtask_complete)

                status = overall_mission.verify(action)

                if status == "success":
                    done = True
                    reward = env._reward()
                elif status == "failure":
                    done = True
                    reward = 0

                if (
                    done
                    and not np.all(np.sum(np.array(subtask_complete), axis=0))
                    and status == "success"
                ):
                    import ipdb

                    ipdb.set_trace()

                agent.analyze_feedback(reward, done)

                actions.append(action)
                images.append(obs["image"])
                directions.append(obs["direction"])

                if args.debug:
                    full_img = Resize((400, 400))(
                        Image.fromarray(env.render(mode="rgb_array"))
                    )
                    plt.imshow(full_img)
                    plt.show()

                if done:
                    images.append(new_obs["image"])
                    if args.save_video:
                        imgs.append(env.render())
                        save_video(f'check{str(len(demos))}.mp4',np.array(imgs))
                    actions.append(env.Actions.done)
                    directions.append(obs["direction"])

                obs = new_obs
            if reward > 0 and (
                args.filter_steps == 0 or len(images) <= args.filter_steps
            ):
                demos.append(
                    (
                        mission,
                        blosc.pack_array(np.array(images)),
                        directions,
                        actions,
                        np.array(subtask_completes),
                        subtasks[::-1],
                        *attributes,
                    )
                )
                just_crashed = False

            if reward == 0:
                if args.on_exception == "crash":
                    raise Exception(
                        "mission failed, the seed is {}".format(seed + len(demos))
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
            logger.info("Saving demos...")
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
    demos_path = utils.get_demos_path(demos=args.demos, env=args.env, subtasks=args.subtasks, origin="agent")
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
                    job_demos[i] = utils.load_demos(
                        utils.get_demos_path(job_demo_names[i], args.env, args.subtasks)
                    )
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
    parser.add_argument("--debug", type=int, default=0, help="")
    parser.add_argument("--num-subtasks", type=int, default=3, help="")
    parser.add_argument("--subtasks", type=str, default=None, nargs="+", help="")
    parser.add_argument("--task-objs", type=str, default=None, nargs="+", help="")
    parser.add_argument("--save_video", action="store_true", default=False, help="Save demo videos")
    

    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s: %(levelname)s: %(message)s")
    logger.info(args)
    # Training demos
    if args.jobs == 0:
        generate_demos(args.episodes, False, args.seed)
    else:
        generate_demos_cluster()

    # Validation demos
    if args.valid_episodes:
        generate_demos(args.valid_episodes, True, int(1e9))
