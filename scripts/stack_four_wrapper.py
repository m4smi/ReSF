import gym
import numpy as np

# COLOR_ORDER = ["red", "green", "blue", "plank"]
# PLANK_RANK = 3
PLANK_RGB = np.array([255, 0, 255]) / 255.0
# half of cube's
PLANK_HALF_H = 0.015

# CUBE_RGB = np.stack([
#     np.array([1.0, 0.0, 0.0]), # red
#     np.array([0.0, 1.0, 0.0]), # green
#     np.array([0.12, 0.56, 1.0]), # blue
# ])
CUBE_COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.12, 0.56, 1.0),
}
CUBE_RGB = np.array(list(CUBE_COLORS.values()))
COLOR_ORDER = list(CUBE_COLORS) + ["plank"]
# last one
PLANK_RANK = len(CUBE_RGB)

class StackFour_Env(gym.Wrapper):
    """Shuffled red/green/blue over the cubes each reset, plus a fixed-colour plank as
    the last object. Also hides the goal marker and blacks out the green skybox."""

    def __init__(self, env, plank_half_h: float = PLANK_HALF_H, plank_rgb=PLANK_RGB,
                 hide_goal: bool = True, bg_gray: int = 0):
        super().__init__(env)
        raw = env.unwrapped
        self.num_blocks = raw.num_blocks
        # last object --> plank
        self.plank_idx = self.num_blocks - 1          
        self.num_cubes = self.num_blocks - 1
        self.plank_half_h = plank_half_h
        self.plank_rgb = np.asarray(plank_rgb, dtype=float)
        self.hide_goal = hide_goal
        # [num_blocks, 3]
        self.block_colors = None
        # [num_blocks,] ; plank entry == PLANK_RANK
        self.block_color_idx = None

        m = raw.sim.model
        # mujoco box is half width (0.045); override z (0.015)
        gid = m.geom_name2id(raw.object_names[self.plank_idx])
        m.geom_size[gid] = [raw.object_size, raw.object_size, plank_half_h]

        if bg_gray is not None:
            for texid in range(m.ntex):
                # green blackground --> gray
                # mjtexture_skybox
                if m.tex_type[texid] == 2:
                    adr = m.tex_adr[texid]
                    sz = int(m.tex_height[texid]) * int(m.tex_width[texid]) * 3
                    m.tex_rgb[adr:adr + sz] = bg_gray
                    break

    @property
    def table_top(self):
        raw = self.env.unwrapped
        return raw.height_offset - raw.object_size

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        raw = self.env.unwrapped
        m, d = raw.sim.model, raw.sim.data

        perm = raw.np_random.permutation(len(CUBE_RGB))[:self.num_cubes]
        self.block_color_idx = np.append(perm, PLANK_RANK).astype(np.int64)
        self.block_colors = np.vstack([CUBE_RGB[perm], self.plank_rgb[None]])

        for name, color in zip(raw.object_names, self.block_colors):
            m.geom_rgba[m.geom_name2id(name)][:3] = color

        # _reset_sim sample random xy for each obj, set qpos[2] = height offset
        # plank has different height offset from cubes - cause half
        pname = raw.object_names[self.plank_idx]
        q = d.get_joint_qpos(f"{pname}:joint").copy()
        # plank height offset
        q[2] = self.table_top + self.plank_half_h
        # identify quat -> flat
        q[3:] = [1.0, 0.0, 0.0, 0.0]
        d.set_joint_qpos(f"{pname}:joint", q)

        if self.hide_goal:
            m.site_rgba[m.site_name2id("target0")][3] = 0.0
        # hide red site marker (originally used as goal, ref) within cubes  
        for name in raw.object_names:
            m.site_rgba[m.site_name2id(name)][3] = 0.0

        # push xpos/site_xpos before render/labels
        raw.sim.forward()
        return obs
