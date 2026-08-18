import itertools
import numpy as np

#ALIGN_R = 2.0
#CLEAR_H = 0.0
# mujuco penetration registers resting cubes with gap < 0
#PEN_TOL = 0.01

def is_wrong_order(upper, lower, rank, plank_idx):
    """plank is universal, doesn't count in RGB ranking."""
    if plank_idx in (upper, lower):
        return False
    return int(rank[upper]) - int(rank[lower]) != 1

def pair_margins(pos, half, rank, plank_idx, r, h, s):
    """
    Convention: unsafe: -1, safe: 1
    Params:
         pos: [N, 3] cubes'xyz
         half_z: [N,] half-heights of cubes
         rank: [N,] colour ranks
         r: ALIGN_R * object_size
         h: CLEAR_H * object_size
         s: object_size
    Returns:
         out: [K=6, ]
    """
    out = []
    for a, b in itertools.combinations(range(len(pos)), 2):
        upper, lower = (a, b) if pos[a][2] > pos[b][2] else (b, a)
        
        if not is_wrong_order(upper, lower, rank, plank_idx):
            out.append(1.0)
            continue
        # when lowering a cube onto a stack, if side by side, gap < 0.0 (safe)
        gap = (pos[upper][2] - half[upper]) - (pos[lower][2] + half[lower])
        #if gap < -PEN_TOL:
        #    out.append(1.0)
        #    continue
        dxy = float(np.hypot(pos[upper][0] - pos[lower][0], 
                             pos[upper][1] - pos[lower][1]))
        # check if dxy < r and gap < h
        #threshold_check = float(np.clip(max(dxy / r - 1.0, gap / h - 1.0), -1.0, 1.0))
        #TODO: check: /s would make thresh unitness; might affect downstream
        # cube is 0.09m -> s (half-width): 0.045, 2 cubes side by side --> gap = 0.09 = 2s = r
        threshold_check = float(max(dxy - r, gap - h) / s)
        out.append(threshold_check)
    return np.array(out, dtype=np.float32)

def get_labels(env, align_r, clear_h):
    """
    Return labels for current sim states
    Returns:
        glob_label: [T, ]
        pair_labels: [T, K]
    """
    raw = env.unwrapped
    d, m = raw.sim.data, raw.sim.model
    pos = np.array([d.get_joint_qpos(f"{o}:joint")[:3] for o in raw.object_names])

    # half height, read from the model so the plank override is picked up
    half_z = np.array([m.geom_size[m.geom_name2id(o)][2] for o in raw.object_names])

    #NOTE: s (raw.object_size) return half width (0.045m)
    margins = pair_margins(pos, half_z, env.block_color_idx, env.plank_idx,
                           align_r * raw.object_size, clear_h * raw.object_size, raw.object_size)
    pair_labels = (margins < 0).astype(np.int64) 
    return int(pair_labels.any()), pair_labels, margins


def labels_from_arrays(obj_pos, half_z, rank, plank_idx, object_size,
                       align_r, clear_h):
    """Re-label a recorded episode offline.
    Params:
        obj_pos: [T, N, 3]
    """
    mg = np.stack([pair_margins(p, half_z, rank, plank_idx,
                                align_r * object_size, clear_h * object_size, object_size)
                   for p in obj_pos])
    pl = (mg < 0).astype(np.int64)
    return pl.any(axis=1).astype(np.int64), pl, mg
