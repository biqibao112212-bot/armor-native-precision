"""Fresh height-scoped confirmation, gated by frozen model hashes.

Only public SDK scenario inputs are changed. The release and fixed15deg armor
mount are unchanged. Every exposure is audited against actual physical centers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import collect_broad as collector

BASE_PLAN = collector.plan
HEIGHT_LIMIT_M = .4
# Midpoint of the two number3 physical armor-center Z offsets in locked r3.
# These planning offsets are checked against actual exposure metadata afterward.
ARMOR_CENTER_Z_MID = (-.06142231822013855 - .07542234659194946) / 2


def plan(seed=2027102212):
    result = BASE_PLAN(seed)
    rng = np.random.default_rng(seed + 91271)
    episodes = []
    for variant in range(4):
        source = BASE_PLAN(seed + 10000 * (variant + 1))["episodes"]
        for band in range(8):
            for j in range(12):
                episode = source[band * 16 + j]
                cfg = episode["config"]
                desired = float(rng.uniform(*[(-.025,.025),(.075,.125),(.20,.385),(-.385,-.20)][variant]))
                # h = cameraZ - (bodyZ + armorOffset); preserve the requested h
                # by choosing a compatible camera height instead of clipping bodyZ.
                low = max(.25, .12 + desired + ARMOR_CENTER_Z_MID)
                high = min(1.1, 1.3 + desired + ARMOR_CENTER_Z_MID)
                camera_z = float(rng.uniform(low, high))
                body_z = camera_z - desired - ARMOR_CENTER_Z_MID
                distance = episode["root_distance_m"]
                dx = float(np.sqrt(distance**2 - (body_z-camera_z)**2))
                old_target = np.asarray(cfg["target_position_world_m"])
                look_offset = np.asarray(cfg["camera_look_at_world_m"]) - old_target
                target = np.array([5.2-dx, 5.8, body_z])
                cfg.update(camera_position_world_m=[5.2,5.8,camera_z],
                           target_position_world_m=target.tolist(),
                           camera_look_at_world_m=(target+look_offset).tolist(),
                           camera_roll_deg=0.)
                cfg["target_rpy_deg"][:2] = [0.,0.]
                episode.update(id=f"level-height40-confirm-v{variant:02d}-b{band}-l{j%3}-m{j%4}",
                               split="sealed", frames=4,
                               planned_camera_minus_armor_mid_height_m=desired,
                               height_stratum=["near_level","around10cm","positive_boundary","negative_boundary"][variant])
                episodes.append(episode)
    result.update(
        episodes=episodes, roll_fixed_zero=True, target_level_fixed_zero=True,
        train_episodes=0, dev_episodes=0, sealed_episodes=len(episodes),
        scope="Fresh independent single-frame confirmation after candidate freeze; level body, camera roll0, fixed armor mount, number3 original pixels, actual camera-to-armor height within+-40cm.",
        height_definition="camera optical center worldZ minus physical armor center worldZ",
        height_scope_m=[-.4,.4], height_layout="4 strata x8 distances x3lights x4motions; report each stratum separately",
        planning_armor_center_z_mid_m=ARMOR_CENTER_Z_MID,
        caps={"frames":1536,"seconds":1800,"maximum_new_bytes":int(1.25*2**30),"minimum_free_bytes":int(1.5*2**30)},
    )
    return result


def validate_height_episode(directory, spec):
    heights=[]
    for line in (directory/"frames.jsonl").open():
        row=json.loads(line)
        assert row["requested_level_target_verified"] and row["requested_roll_zero_verified"]
        q=np.asarray(row["target_quaternion_wxyz"])
        rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        root=np.asarray(row["target_position_world"])
        for armor in row["armors"]:
            center=rot@np.asarray(armor["position"])+root
            heights.append(float(row["camera_position_world"][2]-center[2]))
    assert heights and max(map(abs,heights))<=HEIGHT_LIMIT_M+1e-4, "actual armor height outside+-40cm"
    with (directory/"height-audit.json").open("x") as f:
        json.dump({"verified":True,"physical_plate_centers":len(heights),"height_min_m":min(heights),"height_max_m":max(heights),"definition":"cameraZ minus armorCenterZ","scope_m":[-.4,.4]},f,indent=2)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def verify_freeze(path):
    receipt=json.loads(Path(path).read_text())
    assert receipt["frozen"] is True and receipt["new_confirmation_not_used"] is True
    for role in ["candidate","proposal"]:
        asset=receipt[role];assert sha256(asset["path"])==asset["sha256"],role+" changed after freeze"
    for role,asset in receipt.get("dependencies",{}).items():
        assert sha256(asset["path"])==asset["sha256"],role+" dependency changed after freeze"
    return receipt


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freeze-receipt");parser.add_argument("--plan-only",action="store_true");parser.add_argument("--plan-output")
    args,remaining=parser.parse_known_args();recipe=plan()
    if args.plan_only:
        if args.plan_output:
            with Path(args.plan_output).open("x") as stream:json.dump(recipe,stream,indent=2)
        print(json.dumps({"plan_only":True,"frames":recipe["caps"]["frames"],"episodes":len(recipe["episodes"]),"seed":recipe["seed"],"height_scope_m":recipe["height_scope_m"]}))
        return
    if not args.freeze_receipt:parser.error("Actual confirmation collection requires --freeze-receipt")
    recipe["candidate_freeze"]=verify_freeze(args.freeze_receipt)
    collector.plan=lambda:recipe;collector.episode_validator=validate_height_episode
    sys.argv=[sys.argv[0]]+remaining;collector.main()

if __name__=="__main__":main()
