import copy
import gym
from .verifier import *
from .levelgen import *
from gym_minigrid.minigrid import *
from core.configs.default_data_configs.babyai import *


class Level_PresetMaze(RoomGridLevel):
    def __init__(
        self,
        agent_init,
        task_obj_init,
        distractor_obj_init,
        sequential,
        full_observability=False,
        rand_dir=True,
        **kwargs
    ):
        self.agent_init = agent_init
        self.task_obj_init = task_obj_init
        self.distractor_obj_init = distractor_obj_init
        self.sequential = sequential
        self.full_observability = full_observability
        self.rand_dir = rand_dir

        # [type, color, x, y]
        self.objs = [
            ["ball", "red", 11, 12],
            ["ball", "blue", 9, 4],
            ["box", "red", 2, 16],
            ["box", "blue", 2, 10],
            ["ball", "yellow", 13, 5],
            ["key", "green", 4, 6],
            ["key", "grey", 16, 16],
            ["box", "green", 13, 13],
            ["key", "red", 18, 11],
            ["ball", "grey", 17, 4],
            ["ball", "green", 4, 19],
            ["ball", "grey", 10, 18],
            ["box", "grey", 4, 9],
            ["ball", "purple", 15, 17],
            ["key", "purple", 12, 3],
            ["key", "yellow", 8, 4],
            ["key", "blue", 13, 16],
            ["box", "yellow", 18, 6],
            ["key", "blue", 3, 10],
            ["box", "purple", 17, 3],
        ]

        # colors: purple, blue, yellow, green, red, grey

        # i, j is the room, the room is 3x3
        # [color, idx, i, j, is_open]
        self.doors = [
            ["purple", 0, 0, 0, True],
            ["blue", 1, 0, 0, True],
            ["yellow", 0, 0, 1, True],
            ["green", 1, 0, 1, True],
            ["red", 0, 0, 2, True],
            ["grey", 0, 1, 0, True],
            ["purple", 1, 1, 0, True],
            ["blue", 0, 1, 1, True],
            ["yellow", 1, 1, 1, True],
            ["green", 0, 1, 2, True],
            ["red", 1, 2, 0, True],
            ["grey", 1, 2, 1, True],
        ]

        super().__init__()

    def place_agent_start(self, agent_init, rand_dir=True):
        if agent_init == "random":
            self.place_agent()
        elif agent_init == "same_room":
            self.place_agent(1, 1)
        elif agent_init == "fixed":
            self.put_obj(None, 11, 11)
            self.agent_pos = np.array([11, 11])
            if rand_dir:
                self.agent_dir = self._rand_int(0, 4)
        else:
            raise NotImplementedError

    def get_obj(self, kind, color):
        if kind == "key":
            obj = Key(color)
        elif kind == "ball":
            obj = Ball(color)
        elif kind == "box":
            obj = Box(color)
        elif kind == "door":
            obj = Door(color)
        return obj

    def add_objects(self, objs, obj_init="fixed"):
        for obj_desc in objs:
            obj_type, color, x, y = obj_desc
            obj = self.get_obj(obj_type, color)

            if obj_init == "fixed":
                self.put_obj(obj, x, y)
            elif obj_init == "same_room":
                i, j = self.room_from_pos(x, y)
                x, y = self.place_in_room(i, j, obj)
            else:
                x, y = self.place_obj(obj)

    def add_door(self, i, j, door_idx=None, color=None, locked=None, is_open=True):
        room = self.get_room(i, j)

        if door_idx == None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if color == None:
            color = self._rand_color()

        if locked is None:
            locked = self._rand_bool()

        assert room.doors[door_idx] is None, "door already exists"

        room.locked = locked
        door = Door(color, is_locked=locked)

        pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.cur_pos = pos
        door.is_open = is_open

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx + 2) % 4] = door

        return door, pos

    def add_doors(self, doors):
        for door_desc in doors:
            color, idx, i, j, is_open = door_desc
            door, pos = self.add_door(i, j, idx, color, locked=False, is_open=is_open)

    def gen_mission(self):
        # Fix agent
        self.place_agent_start(self.agent_init, self.rand_dir)

        # Add objects
        if hasattr(self, "task_objs"):
            # Place distractor objects first because usually they're fixed
            self.add_objects(self.distractor_objs, self.distractor_obj_init)
            self.add_objects(self.task_objs, self.task_obj_init)
        else:
            self.add_objects(self.objs, self.task_obj_init)

        # Add doors
        if hasattr(self, "closed_doors"):
            self.add_doors(self.closed_doors)
            self.add_doors(self.opened_doors)
        else:
            self.add_doors(self.doors)

        self.check_objs_reachable()

    def gen_obs(self):
        """
        Override gen obs for full observability
        """

        if self.full_observability:
            grid = self.grid
            image = grid.encode()

            # Add two more channels, one for dir and one for inventory
            # if self.use_additional_channels:
            #     h, w, c = image.shape
            #     new_image = np.zeros((h, w, c + 2))

            #     new_image[self.agent_pos[0]][self.agent_pos[0]] = [
            #         OBJECT_TO_IDX["agent"],
            #         COLOR_TO_IDX["red"],
            #         0,
            #         self.agent_dir
            #         + 1,  # add one so that 0 could denote agent not there
            #         0,
            #     ]

            #     if self.carrying:
            #         new_image[self.agent_pos[0]][self.agent_pos[1]][
            #             -1
            #         ] = self.carrying.encode()[0]

            #     image = new_image
            # else:
            image[self.agent_pos[0]][self.agent_pos[1]] = np.array(
                [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir]
            )

            # Make it so the agent sees what it's carrying
            # We do this by placing the picked object's id at the agent's color channel
            if self.carrying:
                image[self.agent_pos[0]][self.agent_pos[1]][1] = self.carrying.encode()[
                    0
                ]

        # Partial Observability
        else:
            return super().gen_obs()

        assert hasattr(
            self, "mission"
        ), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs


class Level_PresetMazeCompositionalTask(Level_PresetMaze):
    def __init__(
        self,
        agent_init,
        task_obj_init,
        distractor_obj_init,
        sequential,
        subtasks=[],
        num_subtasks=0,
        rand_dir=True,
        **kwargs
    ):

        """
        subtasks - list of "skill color type"
        """
        self.subtasks = subtasks
        self.action_to_instr = {
            "goto": GoToInstr,
            "pickup": PickupInstr,
            "put": PutNextInstr,
            "open": OpenInstr,
        }
        Level_PresetMaze.__init__(
            self, agent_init, task_obj_init, distractor_obj_init, sequential, rand_dir
        )

    def create_subtask(self, subtask_str):
        tokens = subtask_str.split(" ")

        semantic_action = tokens[0]
        tokens = tokens[1:]

        if semantic_action == "put":
            color_1, kind_1, color_2, kind_2 = tokens
            o1_desc = ObjDesc(kind_1, color_1)
            o2_desc = ObjDesc(kind_2, color_2)
            instr = self.action_to_instr[semantic_action](o1_desc, o2_desc)
        else:
            color, kind = tokens
            o1_desc = ObjDesc(kind, color)
            o2_desc = None
            instr = self.action_to_instr[semantic_action](o1_desc)

        return instr, o1_desc, o2_desc

    def gen_mission(self):
        if self.subtasks == []:
            raise NotImplementedError

        instrs = []
        o1s = []
        o2s = []

        for i, subtask in enumerate(self.subtasks):
            instr, o1_desc, o2_desc = self.create_subtask(subtask)
            o1s.append(o1_desc)
            o2s.append(o2_desc)
            instrs.append(instr)

        self.task_objs, self.distractor_objs = [], []
        self.closed_doors, self.opened_doors = [], []

        self.task_objs_indx = []
        self.closed_door_indx = []

        random.shuffle(self.objs)
        random.shuffle(self.doors)

        # Separate the task objs and distractor objs
        for obj_desc in o1s + o2s:
            if obj_desc is not None:
                kind, color = obj_desc.type, obj_desc.color

                # Loop over list of objects
                if kind == "door":
                    for i, door in enumerate(self.doors):
                        if door[0] == color:
                            # set door to be closed
                            door[-1] = False
                            self.closed_doors.append(door)
                            self.closed_door_indx.append(i)
                            break

                else:
                    for i, obj in enumerate(self.objs):
                        if obj[0] == kind and obj[1] == color:
                            self.task_objs.append(obj)
                            self.task_objs_indx.append(i)
                            break

        self.distractor_objs = [
            obj for i, obj in enumerate(self.objs) if i not in self.task_objs_indx
        ]
        self.opened_doors = [
            obj for i, obj in enumerate(self.doors) if i not in self.closed_door_indx
        ]

        self.instrs = CompositionalInstr(instrs, sequential=False)
        Level_PresetMaze.gen_mission(self)


class Level_PresetMazeRandomCompositionalTask(Level_PresetMazeCompositionalTask):
    def __init__(
        self,
        agent_init,
        task_obj_init,
        distractor_obj_init,
        sequential,
        num_subtasks=0,
        rand_dir=True,
        **kwargs
    ):
        self.num_subtasks = num_subtasks
        self.semantic_skills = ["goto", "pickup", "put", "open"]
        Level_PresetMazeCompositionalTask.__init__(
            self, agent_init, task_obj_init, distractor_obj_init, sequential, rand_dir
        )

    def gen_mission(self):
        if self.num_subtasks == 0:
            raise NotImplementedError

        semantic_actions = np.random.choice(
            self.semantic_skills, self.num_subtasks, replace=True
        ).tolist()

        # print(semantic_actions)

        num_task_objs = 0
        num_doors = 0

        # Figure out how many task objs are needed
        for semantic_action in semantic_actions:
            if semantic_action == "put":
                num_task_objs += 2
            elif semantic_action == "open":
                num_doors += 1
            else:
                num_task_objs += 1

        all_objs = copy.deepcopy(self.objs)
        all_doors = copy.deepcopy(self.doors)
        random.shuffle(all_objs)
        self.task_objs, self.distractor_objs = (
            all_objs[:num_task_objs],
            all_objs[num_task_objs:],
        )
        random.shuffle(all_doors)
        self.closed_doors, self.opened_doors = (
            all_doors[:num_doors],
            all_doors[num_doors:],
        )

        # Set door to be closed
        for door in self.closed_doors:
            door[-1] = False

        instrs = []

        task_obj_indx = 0
        door_indx = 0
        for semantic_action in semantic_actions:
            if semantic_action == "put":
                kind_1, color_1, _, _ = self.task_objs[task_obj_indx]
                kind_2, color_2, _, _ = self.task_objs[task_obj_indx + 1]

                o1_desc = ObjDesc(kind_1, color_1)
                o2_desc = ObjDesc(kind_2, color_2)
                instr = self.action_to_instr[semantic_action](o1_desc, o2_desc)
                task_obj_indx += 2
            elif semantic_action == "open":
                color, _, _, _, is_open = self.closed_doors[door_indx]
                o_desc = ObjDesc("door", color)
                instr = self.action_to_instr[semantic_action](o_desc)
                door_indx += 1
            else:
                kind, color, _, _ = self.task_objs[task_obj_indx]
                o_desc = ObjDesc(kind, color)
                instr = self.action_to_instr[semantic_action](o_desc)
                task_obj_indx += 1

            instrs.append(instr)
        # print(instrs)
        # print(self.closed_doors)

        self.instrs = CompositionalInstr(instrs)
        Level_PresetMaze.gen_mission(self)


# Register the levels in this file
register_levels(__name__, globals())
