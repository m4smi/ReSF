import numpy as np
import os, re, glob

def _finger_geoms(model):
    """
    Returns:
        {geom_id: "l", geom_id: "r"} for the two gripper pads
    """
    return {model.geom_name2id(f"robot0:{s}_gripper_finger_link"): s for s in ("l", "r")}

def _scan_contacts(env):
    """
    Returns:
        block_pairs_in_contact: set[frozenset]
        grasped: set[int]) for the current sim state.
    """
    raw = env.unwrapped
    m, d = raw.sim.model, raw.sim.data
    id_to_block = {m.geom_name2id(n): i for i, n in enumerate(raw.object_names)}
    fingers = _finger_geoms(m)
    pairs, touched_by = set(), {}
    # dncon -> active contact points in the scene
    for k in range(d.ncon):
        c = d.contact[k]
        # every contact is inherent betwen 2 geoms
        b1, b2 = id_to_block.get(c.geom1), id_to_block.get(c.geom2)
        # none for any objects not in {table, arm-links, floor, fingers} 
        if b1 is not None and b2 is not None:
            pairs.add(frozenset((b1, b2)))
        # one geom is block, the other is gripper pad
        # --> record which side touched
        elif b1 is not None and c.geom2 in fingers:
            touched_by.setdefault(b1, set()).add(fingers[c.geom2])
        # mirror as we don't know which is block, which is gripper pad
        elif b2 is not None and c.geom1 in fingers:
            touched_by.setdefault(b2, set()).add(fingers[c.geom1])
    # true if both pad pinched
    grasped = {b for b, sides in touched_by.items() if len(sides) == 2}
    return pairs, grasped

def get_gripper_state(env):
    """
    Returns:
        gripper_qpos: left/right finger openings
        grasped: bool[num_blocks]).
    """
    raw = env.unwrapped
    d = raw.sim.data
    gripper_qpos = np.array(
        [float(d.get_joint_qpos(f"robot0:{s}_gripper_finger_joint")) for s in ("l", "r")],
        dtype=np.float32)
    _, grasped = _scan_contacts(env)
    grasped_bool = np.array([i in grasped for i in range(raw.num_blocks)], dtype=np.bool_)
    return gripper_qpos, grasped_bool

def stack_heights(env, tol_frac: float = 1.5):
    """Return: Tallest stacked height (for diag)."""
    raw = env.unwrapped
    d = raw.sim.data
    pairs, _ = _scan_contacts(env)
    pos = {i: np.asarray(d.get_joint_qpos(f"{raw.object_names[i]}:joint"))
           for i in range(raw.num_blocks)}
    # upper -> lower
    below = {}
    for a, b in pairs:
        upper, lower = (a, b) if pos[a][2] > pos[b][2] else (b, a)
        dxy = float(np.hypot(pos[upper][0] - pos[lower][0], pos[upper][1] - pos[lower][1]))
        if dxy < tol_frac * raw.object_size:
            below[upper] = lower
    best = 1
    for i in range(raw.num_blocks):
        n, cur, seen = 1, i, set()
        while cur in below and cur not in seen:
            seen.add(cur)
            cur = below[cur]
            n += 1
        best = max(best, n)
    return best

def next_idx(data_dir, name):
    idxs = [int(re.search(r"_(\d+)\.npz$", f).group(1))
            for f in glob.glob(os.path.join(data_dir, f"{name}_*.npz"))]
    return max(idxs) + 1 if idxs else 0
