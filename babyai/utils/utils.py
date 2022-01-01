import numpy as np
from gym_minigrid.minigrid import Grid


def get_obs_render(obs, agent_view_size=7, agent_dir=0):
    """
    takes 7x7x3 obs array and renders it to corresponding RGB Image array
    """
    imgs = []
    tile_size = 32
    for i in range(1):
        ob = np.array(obs)
        grid, vis_mask = Grid.decode(ob, direction=agent_dir)
        img = grid.render(
            tile_size,
            # agent_pos=(agent_view_size // 2, agent_view_size - 1),
            # agent_dir=3,
            agent_pos=None,
            agent_dir=None,
            highlight_mask=vis_mask,
        )
        imgs.append(img)
    return np.asarray(np.array(imgs), dtype=np.uint8)
