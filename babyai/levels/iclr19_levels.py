"""
Levels described in the ICLR 2019 submission.
"""

import gym
from .verifier import *
from .levelgen import *
from gym_minigrid.minigrid import *
from spirl.configs.default_data_configs.babyai import *


class Level_GoToRedBallGrey(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """
    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(num_rows=1,
                         num_cols=1,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        dists = self.add_distractors(num_distractors=self.num_dists,
                                     all_unique=False)

        for dist in dists:
            dist.color = "grey"

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBall(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """
    def __init__(self,
                 room_size=8,
                 num_dists=7,
                 num_rows=1,
                 num_cols=1,
                 seed=None):
        self.num_dists = num_dists
        super().__init__(num_rows=num_rows,
                         num_cols=num_cols,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")
        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBallR3(Level_GoToRedBall):
    """
    Same as Level_GoToRedBall with grid size 3x3
    Ensures that only one RedBall is present in the Grid
    """
    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(num_rows=3,
                         num_cols=3,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent(1, 1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors(num_distractors=self.num_dists,
                                     all_unique=False)
        for dist in dists:
            if dist.type == "ball" and (dist.color == "red"):
                raise RejectSampling("can only have one blue or red ball")
        # self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)

        if i == 1 and j == 1:
            raise RejectSampling(
                "agent and obj should not be in the same room")
        obj, _ = self.add_object(i, j, "ball", "red")

        self.connect_all()

        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBallNoDists(Level_GoToRedBall):
    """
    Go to the red ball. No distractors present.
    """
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=0, seed=seed)


class Level_GoToObj(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """
    def __init__(self, room_size=8, seed=None):
        super().__init__(num_rows=1,
                         num_cols=1,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=1)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToObjS4(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=4, seed=seed)


class Level_GoToObjS6(Level_GoToObj):
    def __init__(self, seed=None):
        super().__init__(room_size=6, seed=seed)


class Level_GoToLocal(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, no distractors
    """
    def __init__(self, room_size=8, num_dists=8, seed=None):
        self.num_dists = num_dists
        super().__init__(num_rows=1,
                         num_cols=1,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists,
                                    all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocalS5N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_dists=2, seed=seed)


class Level_GoToLocalS6N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=2, seed=seed)


class Level_GoToLocalS6N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=3, seed=seed)


class Level_GoToLocalS6N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_dists=4, seed=seed)


class Level_GoToLocalS7N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=4, seed=seed)


class Level_GoToLocalS7N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=7, num_dists=5, seed=seed)


class Level_GoToLocalS8N2(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=2, seed=seed)


class Level_GoToLocalS8N3(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=3, seed=seed)


class Level_GoToLocalS8N4(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=4, seed=seed)


class Level_GoToLocalS8N5(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=5, seed=seed)


class Level_GoToLocalS8N6(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=6, seed=seed)


class Level_GoToLocalS8N7(Level_GoToLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=8, num_dists=7, seed=seed)


class Level_PutNextLocal(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """
    def __init__(self, room_size=8, num_objs=8, seed=None):
        self.num_objs = num_objs
        super().__init__(num_rows=1,
                         num_cols=1,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs,
                                    all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(ObjDesc(o1.type, o1.color),
                                   ObjDesc(o2.type, o2.color))


class Level_PutNextLocalS5N3(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_objs=3, seed=seed)


class Level_PutNextLocalS6N4(Level_PutNextLocal):
    def __init__(self, seed=None):
        super().__init__(room_size=6, num_objs=4, seed=seed)


class Level_GoTo(RoomGridLevel):
    """
    Go to an object, the object may be in another room. Many distractors.
    """
    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        doors_open=False,
        seed=None,
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(num_rows=num_rows,
                         num_cols=num_cols,
                         room_size=room_size,
                         seed=seed)

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists,
                                    all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

        # If requested, open all the doors
        if self.doors_open:
            self.open_all_doors()


class Level_GoToOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(doors_open=True, seed=seed)


class Level_GoToObjMaze(Level_GoTo):
    """
    Go to an object, the object may be in another room. No distractors.
    """
    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=False, seed=seed)


class Level_GoToObjMazeOpen(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, doors_open=True, seed=seed)


class Level_GoToObjMazeS4R2(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1,
                         room_size=4,
                         num_rows=2,
                         num_cols=2,
                         seed=seed)


class Level_GoToObjMazeS4(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=4, seed=seed)


class Level_GoToObjMazeS5(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=5, seed=seed)


class Level_GoToObjMazeS6(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=6, seed=seed)


class Level_GoToObjMazeS7(Level_GoTo):
    def __init__(self, seed=None):
        super().__init__(num_dists=1, room_size=7, seed=seed)


class Level_GoToImpUnlock(RoomGridLevel):
    """
    Go to an object, which may be in a locked room.
    Competencies: Maze, GoTo, ImpUnlock
    No unblocking.
    """
    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i,
                                         j,
                                         num_distractors=2,
                                         all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        # Add a single object to the locked room
        # The instruction requires going to an object matching that description
        (obj, ) = self.add_distractors(id,
                                       jd,
                                       num_distractors=1,
                                       all_unique=False)
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_Pickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room.
    """
    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnblockPickup(RoomGridLevel):
    """
    Pick up an object, the object may be in another room. The path may
    be blocked by one or more obstructors.
    """
    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=20, all_unique=False)

        # Ensure that at least one object is not reachable without unblocking
        # Note: the selected object will still be reachable most of the time
        if self.check_objs_reachable(raise_exc=False):
            raise RejectSampling("all objects reachable")

        obj = self._rand_elem(objs)
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_Open(RoomGridLevel):
    """
    Open a door, which may be in another room
    """
    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()

        # Collect a list of all the doors in the environment
        doors = []
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        doors.append(door)

        door = self._rand_elem(doors)
        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_Unlock(RoomGridLevel):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """
    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i,
                                         j,
                                         num_distractors=3,
                                         all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class Level_PutNext(RoomGridLevel):
    """
    Put an object next to another object. Either of these may be in another room.
    """
    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=18, all_unique=False)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        self.instrs = PutNextInstr(ObjDesc(o1.type, o1.color),
                                   ObjDesc(o2.type, o2.color))


class Level_PickupLoc(LevelGen):
    """
    Pick up an object which may be described using its location. This is a
    single room environment.

    Competencies: PickUp, Loc. No unblocking.
    """
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            action_kinds=["pickup"],
            instr_kinds=["action"],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False,
        )


class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """
    def __init__(self,
                 room_size=8,
                 num_rows=3,
                 num_cols=3,
                 num_dists=18,
                 seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=["goto"],
            locked_room_prob=0,
            locations=False,
            unblocking=False,
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None):
        super().__init__(room_size=5,
                         num_rows=2,
                         num_cols=2,
                         num_dists=4,
                         seed=seed)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """
    def __init__(self,
                 room_size=8,
                 num_rows=3,
                 num_cols=3,
                 num_dists=18,
                 seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=["action"],
            locations=False,
            unblocking=True,
            implicit_unlock=False,
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(room_size=5,
                         num_rows=2,
                         num_cols=2,
                         num_dists=7,
                         seed=seed)


class Level_SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            seed=seed,
            instr_kinds=["action"],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
        )


class Level_SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """
    def __init__(self, seed=None):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(seed=seed,
                         locations=True,
                         unblocking=True,
                         implicit_unlock=False)


class Level_PresetMaze(RoomGridLevel):
    """
    Preset Maze level: Fixed doors, distractors and objects
    Agent Start location is variable
    Instruction : put the blue key next to a purple box and open the yellow door
    """
    def add_object_pos(self, kind, color, pos_x, pos_y, init="fixed"):
        if kind == "key":
            obj = Key(color)
        elif kind == "ball":
            obj = Ball(color)
        elif kind == "box":
            obj = Box(color)
        elif kind == "door":
            obj = Door(color)

        if init == "fixed":
            self.put_obj(obj, pos_x, pos_y)
        elif init == "same_room":
            i, j = self.room_from_pos(pos_x, pos_y)
            [pos_x, pos_y] = self.place_in_room(i, j, obj)
        elif init == "bimodal":
            i, j = self.room_from_pos(pos_x, pos_y)
            [pos_x, pos_y
             ] = random.choice(self.sampled_pos[f"{color}_{kind}_{i}_{j}"])
            self.put_obj(obj, pos_x, pos_y)
        elif init == "random":
            [pos_x, pos_y] = self.place_obj(obj)
        else:
            return NotImplementedError

        return obj, [pos_x, pos_y]

    def room_from_pos(self, x, y):
        """Get the room a given position maps to"""

        assert x >= 0
        assert y >= 0

        i = x // (self.room_size - 1)
        j = y // (self.room_size - 1)

        assert i < self.num_cols
        assert j < self.num_rows

        return i, j

    def set_door_state(self,
                       color=None,
                       i=None,
                       j=None,
                       x=None,
                       y=None,
                       state=None):
        """
        Close all the doors in the maze
        """
        if x and y:
            door = self.grid.get(x, y)
            assert door.color == color
            door.is_open = state
            return

        if i and j:
            room = self.get_room(i, j)

        for door in room.doors:
            if door:
                if color is None:
                    door.is_open = False
                else:
                    if color == door.color:
                        door.is_open = state

    def place_agent_start(self, agent_init):
        if agent_init == "random":
            self.place_agent()
        elif agent_init == "same_room":
            self.place_agent(1, 1)
        elif agent_init == "fixed":
            self.put_obj(None, 11, 11)
            self.agent_pos = np.array([11, 11])
            self.agent_dir = np.random.randint(4)
        else:
            raise NotImplementedError

    def replace_objs(self, objs, init="fixed"):
        for obj in objs:
            if obj.type.lower() == "door":
                return
            pos_x, pos_y = obj.cur_pos
            self.put_obj(None, pos_x, pos_y)
            dist, pos = self.add_object_pos(obj.type, obj.color, pos_x, pos_y,
                                            init)
            # TODO: fix the world objs list
        return

    def add_objects(self, init="fixed", objs=[]):
        """
        Add specific objects as distractors that can potentially distract/confuse the agent.
        """
        dists = []

        for desc in objs:
            kind, color, room_i, room_j, pos_x, pos_y = desc
            dist, pos = self.add_object_pos(kind, color, pos_x, pos_y, init)
            dists.append(dist)

        # self.world_objs = dists
        return dists

    def add_door(
        self,
        i,
        j,
        door_idx=None,
        color=None,
        state=None,
        locked=None,
        pos_x=None,
        pos_y=None,
    ):
        """
        Add a door to a room, connecting it to a neighbor
        """

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

        if pos_x is not None and pos_y is not None:
            pos = [pos_x, pos_y]
        else:
            pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.init_pos = pos
        door.cur_pos = pos
        door.is_open = state

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx + 2) % 4] = door

        return door, pos

    def connect_all(self, door_desc=[]):
        """
        Overrides the connect_all() method
        Connects a 3x3 maze with doors at pre-specified location
        """
        if not door_desc:
            door_desc = [
                ["green", 2, 0, 1, 16, 7, True],
                ["blue", 2, 0, 2, 14, 6, True],
                ["grey", 1, 0, 2, 7, 3, True],
                ["yellow", 1, 1, 2, 7, 9, True],
                ["purple", 1, 1, 0, 14, 12, True],
                ["red", 1, 0, 1, 11, 7, True],
                ["blue", 2, 1, 1, 16, 14, True],
                ["grey", 0, 2, 3, 2, 14, True],
                ["green", 1, 2, 3, 11, 14, True],
                ["purple", 0, 1, 3, 2, 7, True],
                ["yellow", 1, 2, 0, 14, 18, True],
                ["red", 1, 2, 2, 7, 15, True],
            ]

        for desc in door_desc:
            color, i, j, idx, x, y, state = desc
            door, pos = self.add_door(i,
                                      j,
                                      idx,
                                      color,
                                      state,
                                      locked=False,
                                      pos_x=x,
                                      pos_y=y)

    def gen_obs(self):
        # TODO Replace with a parameter or full observability
        if self.agent_view_size > 8:  # Full Observability
            grid = self.grid
            image = grid.encode()

            # Add two more channels, one for dir and one for inventory
            if self.use_additional_channels:
                h, w, c = image.shape
                new_image = np.zeros((h, w, c + 2))

                new_image[self.agent_pos[0]][self.agent_pos[0]] = [
                    OBJECT_TO_IDX["agent"],
                    COLOR_TO_IDX["red"],
                    0,
                    self.agent_dir +
                    1,  # add one so that 0 could denote agent not there
                    0,
                ]

                if self.carrying:
                    new_image[self.agent_pos[0]][
                        self.agent_pos[1]][-1] = self.carrying.encode()[0]

                image = new_image
            else:
                image[self.agent_pos[0]][self.agent_pos[1]] = np.array([
                    OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir
                ])

                # Make it so the agent sees what it's carrying
                # We do this by placing the picked object's id at the agent's color channel
                if self.carrying:
                    image[self.agent_pos[0]][
                        self.agent_pos[1]][1] = self.carrying.encode()[0]

        else:  # Partial Observability
            grid, vis_mask = self.gen_obs_grid()
            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask)
        assert hasattr(
            self,
            "mission"), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission
        }

        return obs


class Level_PresetMazeCompositionalTask(Level_PresetMaze):
    """
    Put an object next to another object. Either of these may be in another room.
    """
    def sample_new_object(self, cur_obj_descs, all_objs):
        new_obj_desc = cur_obj_descs[0]
        while new_obj_desc in cur_obj_descs:
            new_obj = self._rand_elem(all_objs)
            new_obj_desc = ObjDesc(new_obj.type, new_obj.color)
        return new_obj_desc

    def create_subtask(self, i, subtask, o1_desc, o2_desc, instrs,
                       distractors):
        if i != 0:
            # Ensure using different objects from previous subtask
            if isinstance(instrs[-1], PutNextInstr):
                prev_obj_descs = [
                    str(instrs[-1].desc_move),
                    str(instrs[-1].desc_fixed)
                ]
            else:
                prev_obj_descs = [str(instrs[-1].desc)]

            for desc in prev_obj_descs:
                if str(desc) == str(o1_desc):
                    o1_desc = self.sample_new_object(prev_obj_descs,
                                                     distractors)

        if subtask == "go":
            instr = GoToInstr(o1_desc)
        elif subtask == "pickup":
            instr = PickupInstr(o1_desc)
        elif subtask == "put":
            instr = PutNextInstr(o1_desc, o2_desc)
        elif subtask == "open":
            if o1_desc.type != "door":
                obj_type, obj_color = "door", random.choice(COLOR_NAMES)
                o1_desc = ObjDesc(obj_type, obj_color)
                self.close_all_doors(color=obj_color)
            else:
                self.close_all_doors(color=o1_desc.color)

            instr = OpenInstr(o1_desc)
        return instr, o1_desc, o2_desc

    def gen_mission(
        self,
        agent_init="fixed",
        task_obj_init="fixed",
        distractor_obj_init="fixed",
        num_subtasks=3,
        subtasks=[],
        task_objs=[],
        distractor_objs=[],
        doors=[],
        sequential=False,
    ):
        self.place_agent_start(agent_init)
        self.connect_all(doors)

        if not task_objs:
            task_objs = []

        task_objs_list = [
            task_objs[i:i + 2] for i in range(0, len(task_objs), 2)
        ]
        task_objs_desc = [
            obj for obj in distractor_objs
            if [obj[1], obj[0]] in task_objs_list
        ]
        distractor_objs_desc = [
            obj for obj in distractor_objs
            if [obj[1], obj[0]] not in task_objs_list
        ]

        if not hasattr(self, "init") or distractor_objs_desc == []:
            distractor_objs_desc = [
                ["ball", "red", 1, 1, 11, 12],
                ["ball", "blue", 1, 0, 9, 4],
                ["ball", "blue", 1, 0, 9, 3],
                ["ball", "blue", 1, 0, 9, 2],
            ]

        distractor_objs = self.add_objects(init=distractor_obj_init,
                                           objs=distractor_objs_desc)

        if (not hasattr(self, "sampled_pos") and task_objs_desc
                and task_obj_init == "bimodal"):
            self.ctr = 0
            self.sampled_pos = {}

            task_obj_poss = []
            for desc in task_objs_desc:
                kind, color, i, j, x, y = desc
                room = self.get_room(*self.room_from_pos(x, y))
                pos_1 = room.rand_pos(self)
                while pos_1 in task_obj_poss:
                    pos_1 = room.rand_pos(self)
                task_obj_poss.append(pos_1)

                pos_2 = room.rand_pos(self)
                while pos_2 in task_obj_poss:
                    pos_2 = room.rand_pos(self)
                task_obj_poss.append(pos_2)
                self.ctr += 1

                self.sampled_pos[f"{color}_{kind}_{i}_{j}"] = [pos_1, pos_2]

        task_objs_ = self.add_objects(init=task_obj_init, objs=task_objs_desc)

        # self.open_all_doors()
        self.check_objs_reachable()

        skills = ["go", "pickup", "put", "open"]
        if subtasks is None or len(subtasks) == 0:
            subtasks = np.random.choice(skills, num_subtasks,
                                        replace=True).tolist()

        instrs = []

        for i, subtask in enumerate(subtasks):
            if task_objs:
                o1_desc = ObjDesc(task_objs[1], task_objs[0])
                task_objs = task_objs[2:]

                if subtask == "put":
                    o2_desc = ObjDesc(task_objs[1], task_objs[0])
                    task_objs = task_objs[2:]
                else:
                    o2_desc = None
            else:
                o1, o2 = self._rand_subset(distractor_objs, 2)
                o1_desc = ObjDesc(o1.type, o1.color)
                o2_desc = ObjDesc(o2.type, o2.color)

            instr, o1_desc, o2_desc = self.create_subtask(
                i, subtask, o1_desc, o2_desc, instrs, distractor_objs)

            # Randomize task obj positions
            if task_obj_init != "fixed":
                o1_matches = o1_desc.find_matching_objs(self)

                # Remove obj from scene and randomize it
                self.replace_objs(o1_matches[0], task_obj_init)

                if subtask == "put":
                    o2_matches = o2_desc.find_matching_objs(self)
                    self.replace_objs(o2_matches[0], task_obj_init)

            instrs.append(instr)

        self.instrs = CompositionalInstr(instrs, sequential=sequential)
        self.init = True


class Level_PutNextOpen(Level_SynthSeq):
    """
    Custom SynthSeq level:
    put the blue key next to a purple box and open the yellow door
    """
    def gen_mission(self):
        self.place_agent(1, 1)

        # Ensure there is only one red or blue ball
        print("Adding distractors")
        dists = self.add_distractors(num_distractors=self.num_dists,
                                     all_unique=False)
        for dist in dists:
            if dist.type == "key" and (dist.color == "blue"):
                raise RejectSampling("can only have one blue key")
            if dist.type == "box" and (dist.color == "purple"):
                raise RejectSampling("can only have one purple box")

        # Instantiates objects inside specific rooms & walls
        blue_key, _ = self.add_object(0, 1, "key", "blue")
        purple_box, _ = self.add_object(2, 0, "box", "purple")
        yellow_door, _ = self.add_door(1, 2, 0, "yellow", locked=False)

        self.connect_all(
            door_colors=["blue", "red", "purple", "grey", "green"])

        self.check_objs_reachable()

        instr_a = PutNextInstr(
            ObjDesc(blue_key.type, blue_key.color),
            ObjDesc(purple_box.type, purple_box.color),
        )
        instr_b = OpenInstr(ObjDesc(yellow_door.type, yellow_door.color))
        self.instrs = AndInstr(instr_a, instr_b)


class Level_MiniBossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25,
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(seed=seed)


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None):
        super().__init__(seed=seed, locked_room_prob=0, implicit_unlock=False)


# Register the levels in this file
register_levels(__name__, globals())
