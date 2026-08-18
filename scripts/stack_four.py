import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import argparse
import time

import numpy as np
import cv2
import gym
import imageio
import ruamel.yaml as yaml 
# registers the envs
import multi_object_fetch

from stack_four_wrapper import StackFour_Env
from label_utils import get_labels
from stack_four_utils import get_gripper_state, stack_heights, next_idx
from gamepad import Pad

def load_configs(file: str):
    with open(file, 'r') as f:
        yml = yaml.YAML(typ="safe", pure=True)
        configs = yml.load(f)
    return configs['defaults']
    # configs_dict = configs['defaults']
    # configs = argpare.Namespace()
    # for key, value in configs_dict.items():
    #     setattr(configs, key, value)
    # return configs

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "stack_four.yaml"))
    known, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(parents=[pre])
    for key, value in load_configs(known.config).items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    ap.add_argument("--mp4", action="store_true", help="also save a preview mp4 per episode")
    ap.add_argument("--no-view", action="store_true", help="skip the live teleop window")
    args = ap.parse_args()
    args.data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)

    env = StackFour_Env(gym.make(args.env), plank_half_h=args.plank_h)
    raw = env.unwrapped
    pad = Pad()
    dt = 1.0 / args.fps
    view = not args.no_view
    if view:
        cv2.namedWindow("stack_plank", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("stack_plank", 512, 512)

    print(f"env={args.env}  blocks={raw.num_blocks}  plank=obj{env.plank_idx} "
          f"({args.plank_h * 200:.0f} cm tall)  r={args.align_r} h={args.clear_h}")
    print("R1=grip  Square=save+next  Triangle=discard+reset  X=quit")

    # keys = ("images", "actions", "obj_pos", "obj_quat", "glob_label", "pair_labels",
    #         "pair_margins", "gripper_qpos", "grasped", "is_first", "is_last",
    #         "plank_idx", "half_z", "align_r", "clear_h",)
    keys = ("images", "actions", "obj_pos", "obj_quat", "glob_label", "pair_labels",
            "pair_margins", "gripper_qpos", "grasped", "is_first", "is_last",)
    print("Initing sim")
    print("-" * 50)
    env.reset()
    while not pad.terminate:
        buf = {k: [] for k in keys}
        first = True
        max_h = 1
        while not pad.done:
            if pad.qk_reset:
                env.reset()
                pad.qk_reset = False
                buf = {k: [] for k in keys}
                first = True
                max_h = 1
            t0 = time.time()
            # frame(t); before stepping
            big = raw.render(size=(args.disp_size, args.disp_size))
            frame = cv2.resize(big, (args.size, args.size), interpolation=cv2.INTER_AREA)
            # state + labels read at the SAME sim state as the rendered frame (before step),
            # --> align with images[t]; action is transition: t-->t+1
            qpos = np.array([raw.sim.data.get_joint_qpos(f"{n}:joint") for n in raw.object_names],
                            dtype=np.float32)  # (num_blocks, 7)
            glob_label, pair_labels, margins = get_labels(env, args.align_r, args.clear_h)
            gripper_qpos, grasped = get_gripper_state(env)

            max_h = max(max_h, stack_heights(env))
            a = pad.action()

            buf["images"].append(frame)
            buf["actions"].append(a)
            buf["obj_pos"].append(qpos[:, :3])
            buf["obj_quat"].append(qpos[:, 3:])
            buf["glob_label"].append(glob_label)
            buf["pair_labels"].append(pair_labels)
            buf["pair_margins"].append(margins)
            buf["gripper_qpos"].append(gripper_qpos)
            buf["grasped"].append(grasped)
            buf["is_first"].append(first)
            buf["is_last"].append(False)
            #--episode-constant, written once after the loop
            # buf["half_z"].append(half_z)
            # buf["align_r"].append(ALIGN_R)
            # buf["clear_h"].append(CLEAR_H)
            first = False
            env.step(a)
            if view:
                cv2.imshow("stack_plank", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            time.sleep(max(0.0, dt - (time.time() - t0)))

        buf["is_last"][-1] = True
        eps = {k: np.asarray(v) for k, v in buf.items()}
        fail_pairs = eps["pair_labels"][-1].sum()
        # a_t: obs_t -> obs_{t+1}; 
        eps["actions"] = eps["actions"][:-1]
        eps["block_colors"] = env.block_colors.astype(np.float32)
        eps["block_color_idx"] = env.block_color_idx.astype(np.int64)
        eps["source"] = np.asarray("teleop")
        # for data-check offline
        eps["plank_idx"] = np.int64(env.plank_idx)
        eps["half_z"] = np.array([raw.sim.model.geom_size[raw.sim.model.geom_name2id(o)][2]
                                  for o in raw.object_names], dtype=np.float32)
        eps["align_r"] = np.float32(args.align_r)
        eps["clear_h"] = np.float32(args.clear_h)
        eps["object_size"] = np.float32(raw.object_size)
        idx = next_idx(args.data_dir, args.name)
        out = os.path.join(args.data_dir, f"{args.name}_{idx}.npz")
        np.savez_compressed(out, **eps)

        flip = bool((np.diff(eps["pair_labels"], axis=0) == -1).any())
        print(f"saved eps {idx}, frames={len(eps['images'])}, act={len(eps['actions'])}, "
              f"label={eps['glob_label'][-1]}, fail_pairs={fail_pairs}, flip={flip}, max_stack={max_h}")
        if args.mp4:
            imageio.mimwrite(out.replace(".npz", ".mp4"), eps["images"],
                             fps=args.fps, macro_block_size=None)
        env.reset()
        pad.reset()

    if view:
        cv2.destroyAllWindows()
    env.close()
    print("done.")

if __name__ == "__main__":
    main()
