"""
Levels described in the ICLR 2019 submission.
"""

import gym
from .verifier import *
from .levelgen import *
from gym_minigrid.minigrid import *

class Level_GoToRedBallGrey(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    The distractors are all grey to reduce perceptual complexity.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        for dist in dists:
            dist.color = 'grey'

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToRedBall(RoomGridLevel):
    """
    Go to the red ball, single room, with distractors.
    This level has distractors but doesn't make use of language.
    """

    def __init__(self, room_size=8, num_dists=7, num_rows=1, num_cols=1, seed=None):
        self.num_dists = num_dists
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, 'ball', 'red')
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
        super().__init__(
            num_rows=3,
            num_cols=3,
            room_size=room_size,
            seed=seed
        )
    def gen_mission(self):
        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        for dist in dists:
            if dist.type == 'ball' and (dist.color == 'red'):
                raise RejectSampling('can only have one blue or red ball')
        #self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)

        if i==1 and j==1:
            raise RejectSampling('agent and obj should not be in the same room')
        obj, _ = self.add_object(i, j, 'ball', 'red')

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
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

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
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
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
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


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
        seed=None
    ):
        self.num_dists = num_dists
        self.doors_open = doors_open
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
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
        super().__init__(num_dists=1, room_size=4, num_rows=2, num_cols=2, seed=seed)


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
            self.add_object(ik, jk, 'key', door.color)
            break

        self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=2,
                        all_unique=False
                    )

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
        obj, = self.add_distractors(id, jd, num_distractors=1, all_unique=False)
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
            raise RejectSampling('all objects reachable')

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
            self.add_object(ik, jk, 'key', door.color)
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
                    self.add_distractors(
                        i,
                        j,
                        num_distractors=3,
                        all_unique=False
                    )

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
        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


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
            action_kinds=['pickup'],
            instr_kinds=['action'],
            num_rows=1,
            num_cols=1,
            num_dists=8,
            locked_room_prob=0,
            locations=True,
            unblocking=False
        )


class Level_GoToSeq(LevelGen):
    """
    Sequencing of go-to-object commands.

    Competencies: Maze, GoTo, Seq
    No locked room.
    No locations.
    No unblocking.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            action_kinds=['goto'],
            locked_room_prob=0,
            locations=False,
            unblocking=False
        )


class Level_GoToSeqS5R2(Level_GoToSeq):
    def __init__(self, seed=None):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=4, seed=seed)


class Level_Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        seed=None
    ):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            seed=seed,
            instr_kinds=['action'],
            locations=False,
            unblocking=True,
            implicit_unlock=False
        )


class Level_SynthS5R2(Level_Synth):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=2,
            num_cols=2,
            num_dists=7,
            seed=seed
        )


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
            instr_kinds=['action'],
            locations=True,
            unblocking=True,
            implicit_unlock=False
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
        super().__init__(
            seed=seed,
            locations=True,
            unblocking=True,
            implicit_unlock=False
        )

class Level_PresetMaze(RoomGridLevel):
    """
    Preset Maze level: Fixed doors, distractors and objects
    Agent Start location is variable
    Instruction : put the blue key next to a purple box and open the yellow door
    """
    def add_object_pos(self, kind, color, pos_x, pos_y):
        if kind == 'key':
            obj = Key(color)
        elif kind == 'ball':
            obj = Ball(color)
        elif kind == 'box':
            obj = Box(color)
        self.put_obj(obj, pos_x, pos_y)
        return obj, [pos_x, pos_y]

    def add_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        """
        Add specific objects as distractors that can potentially distract/confuse the agent.
        """

        objs = []
        # type, color, room_i, room_j, pos_x, pos_y
        objs.append(['ball', 'red', 1, 1, 11, 12])
        objs.append(['ball', 'yellow', 1, 0, 9, 4])
        objs.append(['box', 'red', 0, 2, 2, 16])
        objs.append(['box', 'green', 0, 1, 2, 10])
        objs.append(['ball', 'blue', 1, 0, 13, 5])
        objs.append(['key', 'green', 0, 0, 4, 6])
        objs.append(['key', 'grey', 2, 2, 16, 16])
        objs.append(['box', 'green', 1, 1, 13, 13])
        objs.append(['key', 'red', 1, 0, 12, 3])
        objs.append(['ball', 'grey', 2, 0, 17, 4])
        objs.append(['ball', 'red', 0, 2, 4, 19])
        objs.append(['ball', 'grey', 1, 2, 10, 18])
        objs.append(['ball', 'yellow', 0, 1, 4, 9])
        objs.append(['ball', 'purple', 2, 2, 15, 17])
        objs.append(['key', 'red', 2, 1, 15, 11])
        objs.append(['key', 'yellow', 1, 0, 8, 4])
        objs.append(['key', 'green', 1, 2, 13, 16])
        objs.append(['ball', 'yellow', 2, 0, 16, 6])
        objs.append(['key', 'blue', 0, 1, 3, 10])
        objs.append(['box', 'purple', 2, 0, 17, 3])

        dists = []
        for desc in objs:
            kind, color, room_i, room_j, pos_x, pos_y = desc
            dist, pos = self.add_object_pos(kind, color, pos_x, pos_y)
            dists.append(dist)

        return dists

    def add_door(self, i, j, door_idx=None, color=None, locked=None, pos_x=None, pos_y=None):
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
        door.cur_pos = pos

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx+2) % 4] = door

        return door, pos

    def connect_all(self):
        '''
        Overrides the connect_all() method
        Connects a 3x3 maze with doors at pre-specified location
        '''
        door_desc = []
        door_desc.append([2 ,0, 1, 16, 7, 'green'])
        door_desc.append([2 ,0 ,2 ,14, 6, 'purple'])
        door_desc.append([1 ,1 ,2 ,7, 9, 'green'])
        door_desc.append([1 ,1 ,0 ,14, 12, 'purple'])
        door_desc.append([1 ,0 ,1 ,11, 7, 'red'])
        door_desc.append([2 ,1 ,1 ,16, 14, 'blue'])
        door_desc.append([0 ,2 ,3 ,2, 14, 'grey'])
        door_desc.append([1 ,2 ,3 ,11, 14, 'purple'])
        door_desc.append([0 ,1 ,3 ,2, 7, 'red'])
        door_desc.append([1 ,2 ,0 ,14, 18, 'yellow'])

        for desc in door_desc:
            i,j,idx,x,y,color = desc
            door, pos = self.add_door(i, j, idx, color, locked=False, pos_x=x, pos_y=y)

class Level_PresetMazePutNextOpen(Level_PresetMaze):

    def gen_mission(self):
        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors()

        # Instantiates objects inside specific rooms & walls
        blue_key, _ = self.add_object_pos('key', 'blue', 3, 10 )
        purple_box, _ = self.add_object_pos('box', 'purple', 17, 3)

        self.connect_all()
        self.open_all_doors()

        self.check_objs_reachable()

        instr_a = PutNextInstr(ObjDesc(blue_key.type, blue_key.color),
                                ObjDesc(purple_box.type, purple_box.color))
        instr_b = OpenInstr(ObjDesc('door', 'yellow'))
        self.instrs = AndInstr(instr_a, instr_b)

class Level_PresetMazeGoToBlueKey(Level_PresetMaze):

    def gen_mission(self):
        obj_kind, obj_color = 'key', 'blue'

        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors()

        self.connect_all()
        self.open_all_doors()

        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj_kind, obj_color))

class Level_PresetMazePutBlueKeyPurpleBox(Level_PresetMaze):

    def gen_mission(self):
        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors()

        # Instantiates objects inside specific rooms & walls
        blue_key, _ = self.add_object_pos('key', 'blue', 3, 10 )
        purple_box, _ = self.add_object_pos('box', 'purple', 17, 3)

        self.connect_all()
        self.open_all_doors()

        self.check_objs_reachable()

        self.instrs = PutNextInstr(ObjDesc(blue_key.type, blue_key.color),
                                ObjDesc(purple_box.type, purple_box.color))



class Level_PresetMazePutPick(Level_PresetMaze):

    def gen_mission(self):
        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        dists = self.add_distractors()

        # Instantiates objects inside specific rooms & walls
        blue_key, _ = self.add_object_pos('key', 'blue', 3, 10 )
        purple_box, _ = self.add_object_pos('box', 'purple', 17, 3)

        self.connect_all()
        self.open_all_doors()

        self.check_objs_reachable()

        instr_a = PutNextInstr(ObjDesc(blue_key.type, blue_key.color),
                                ObjDesc(purple_box.type, purple_box.color))
        instr_b = PickupInstr(ObjDesc('box','red'))
        self.instrs = AndInstr(instr_a, instr_b)

class Level_PresetMazeGoPutPick(Level_PresetMaze):

    def gen_mission(self):

        self.place_agent(1, 1)
        dists = self.add_distractors()
        self.connect_all()
        self.open_all_doors()
        self.check_objs_reachable()

        self.instr_a = GoToInstr(ObjDesc('box', 'purple'))
        self.instr_b = PutNextInstr(ObjDesc('key', 'red'),
                                    ObjDesc('ball', 'purple'))
        self.instr_c = PickupInstr(ObjDesc('box','red'))

        self.instrs = TripleAndInstr(self.instr_a, self.instr_b, self.instr_c)

class Level_PresetMazeGoTripleSeq(Level_PresetMaze):

    def gen_mission(self):

        self.place_agent(1, 1)
        dists = self.add_distractors()
        self.connect_all()
        self.open_all_doors()
        self.check_objs_reachable()

        self.instr_a = GoToInstr(ObjDesc('box', 'purple'))
        self.instr_b = GoToInstr(ObjDesc('key', 'red'))
        self.instr_c = GoToInstr(ObjDesc('box','red'))

        self.instrs = TripleAndInstr(self.instr_a, self.instr_b, self.instr_c)


class Level_PresetMazeGoTo(Level_PresetMaze):

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors()
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.open_all_doors()
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

class Level_PresetMazePickup(Level_PresetMaze):
    """
    Pick up an object in a Preset Maze, the object may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors()
        self.check_objs_reachable()
        obj = self._rand_elem(objs)
        self.open_all_doors()
        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_PresetMazePutNext(Level_PresetMaze):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors()
        self.check_objs_reachable()
        self.open_all_doors()
        o1, o2 = self._rand_subset(objs, 2)
        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PresetMazeGoToSeq(Level_PresetMaze):
    """
    Put an object next to another object. Either of these may be in another room.
    """

    def gen_mission(self):
        self.place_agent()
        self.connect_all()
        objs = self.add_distractors()
        self.open_all_doors()
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)
        instr_a = GoToInstr(ObjDesc(o1.type, o1.color))
        instr_b = GoToInstr(ObjDesc(o2.type, o2.color))
        self.instrs = AndInstr(instr_a, instr_b)

class Level_PutNextOpen(Level_SynthSeq):
    """
    Custom SynthSeq level:
    put the blue key next to a purple box and open the yellow door
    """

    def gen_mission(self):
        self.place_agent(1,1)

        # Ensure there is only one red or blue ball
        print("Adding distractors")
        dists = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
        for dist in dists:
            if dist.type == 'key' and (dist.color == 'blue'):
                raise RejectSampling('can only have one blue key')
            if dist.type == 'box' and (dist.color == 'purple'):
                raise RejectSampling('can only have one purple box')

        # Instantiates objects inside specific rooms & walls
        blue_key, _ = self.add_object(0, 1, 'key', 'blue')
        purple_box, _ = self.add_object(2, 0, 'box', 'purple')
        yellow_door, _ = self.add_door(1, 2, 0, 'yellow', locked=False)

        self.connect_all(door_colors=['blue','red','purple','grey','green'])

        self.check_objs_reachable()

        instr_a = PutNextInstr(ObjDesc(blue_key.type, blue_key.color),
                                ObjDesc(purple_box.type, purple_box.color))
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
            locked_room_prob=0.25
        )


class Level_BossLevel(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )


class Level_BossLevelNoUnlock(LevelGen):
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            locked_room_prob=0,
            implicit_unlock=False
        )


# Register the levels in this file
register_levels(__name__, globals())
