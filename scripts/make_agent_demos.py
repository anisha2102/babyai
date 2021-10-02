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
import os
import time
import numpy as np
import blosc
import torch
import re
import copy

import babyai.utils as utils
from gym_minigrid.minigrid import COLOR_NAMES, DIR_TO_VEC


# Object types we are allowed to describe in language
OBJ_TYPES = ["box", "ball", "key", "door"]

# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t != "door", OBJ_TYPES))

# Locations are all relative to the agent's starting position
LOC_NAMES = ["left", "right", "front", "behind"]

ACTION_TYPES = ["go", "pick", "open", "put"]

# Environment flag to indicate that done actions should be
# used by the verifier
use_done_actions = os.environ.get("BABYAI_DONE_ACTIONS", False)

obj_type_indx_map = {obj_type: indx for indx, obj_type in enumerate(OBJ_TYPES)}
color_indx_map = {color: indx for indx, color in enumerate(COLOR_NAMES)}
action_indx_map = {action: indx for indx, action in enumerate(ACTION_TYPES)}
# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    "--log-interval", type=int, default=100, help="interval between progress reports"
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

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info(
        "Demo length: {:.3f}+-{:.3f}".format(
            np.mean(num_frames_per_episode), np.std(num_frames_per_episode)
        )
    )


def breakdown_verifiers(env, verifier):
    if type(verifier) in [AndInstr, BeforeInstr, AfterInstr]:
        a = breakdown_verifiers(env, verifier.instr_a)
        b = breakdown_verifiers(env, verifier.instr_b)
        if type(a) == list and type(b) == list:
            verifiers = [*a, *b]
        elif type(a) == list:
            verifiers = [*a, b]
        elif type(b) == list:
            verifiers = [a, *b]
        else:
            verifiers = [a, b]

        return verifiers

    return verifier

def get_single_one_hot(env, verifier, desc):
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

    action = verifier.surface(env).split(" ")[0]
    action_one_hot = np.zeros(len(ACTION_TYPES))
    action_one_hot[action_indx_map[action.lower()]] = 1

    return color_one_hot, obj_type_one_hot, action_one_hot, obj_desc


def get_one_hot_attributes(env, verifiers):
    obj_descs = []
    colors_oh, obj_types_oh, actions_oh = [], [], []

    for verifier in verifiers:

        if isinstance(verifier, PutNextInstr):
            attr_1 = get_single_one_hot(env, verifier, verifier.desc_move)
            colors_oh.append(attr_1[0])
            obj_types_oh.append(attr_1[1])
            actions_oh.append(attr_1[2])

            attr_2 = get_single_one_hot(env, verifier, verifier.desc_fixed)
            colors_oh.append(attr_2[0])
            obj_types_oh.append(attr_2[1])
            actions_oh.append(attr_2[2])
        else:
            attr = get_single_one_hot(env, verifier, verifier.desc)
            colors_oh.append(attr[0])
            obj_types_oh.append(attr[1])
            actions_oh.append(attr[2])

    return np.array(colors_oh), np.array(obj_types_oh), np.array(actions_oh), obj_descs


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)

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
        obs = env.reset()
        agent.on_reset()

        actions = []
        mission = obs["mission"]
        images = []
        directions = []

        verifiers = breakdown_verifiers(env, env.instrs)

        if type(verifiers) != list:
            verifiers = [verifiers]

        (
            color_one_hot,
            obj_type_one_hot,
            action_one_hot,
            obj_descs,
        ) = get_one_hot_attributes(env, verifiers)

        subtasks = [verifier.surface(env) for verifier in verifiers]
        subtask_complete = []

        overall_mission = copy.deepcopy(env.instrs)
        overall_mission.reset_verifier(env)

        try:
            while not done:
                action = agent.act(obs)["action"]
                if isinstance(action, torch.Tensor):
                    action = action.item()

                new_obs, reward, done, _ = env.step(action, verify=False)

                tmp = [0 for _ in range(len(verifiers))]
                for i, verifier in enumerate(verifiers):
                    status = verifier.verify(action)
                    if status == "success":
                        tmp[i] = 1
                    else:
                        tmp[i] = 0
                subtask_complete.append(tmp)

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
                        np.array(subtask_complete),
                        subtasks,
                        color_one_hot,
                        obj_type_one_hot,
                        action_one_hot,
                        obj_descs,
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
    demos_path = utils.get_demos_path(args.demos, args.env, "agent")
    job_demo_names = [
        os.path.realpath(demos_path + ".shard{}".format(i)) for i in range(args.jobs)
    ]
    for demo_name in job_demo_names:
        job_demos_path = utils.get_demos_path(demo_name)
        if os.path.exists(job_demos_path):
            os.remove(job_demos_path)

    command = [args.job_script]
    command += sys.argv[1:]
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
        output = subprocess.check_output(cmd_i)
        logger.info("LAUNCH OUTPUT")
        logger.info(output.decode("utf-8"))

    job_demos = [None] * args.jobs
    while True:
        jobs_done = 0
        for i in range(args.jobs):
            if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
                try:
                    logger.info("Trying to load shard {}".format(i))
                    job_demos[i] = utils.load_demos(
                        utils.get_demos_path(job_demo_names[i])
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
