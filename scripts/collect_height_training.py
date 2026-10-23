"""TRAIN-only independent height-scoped scenes; all RGB remains native."""
import argparse, json, sys, signal
import numpy as np
import collect_broad as collector
from collect_level_confirmation import ARMOR_CENTER_Z_MID, validate_height_episode

def plan():
    seed=2027102305
    result=collector.plan(seed)
    rng=np.random.default_rng(seed+19127)
    episodes=[]
    for variant in range(20):
        source=collector.plan(seed+10000*(variant+1))["episodes"]
        for band in range(8):
            for j in range(12):
                e=source[band*16+j]
                if variant<14:
                    kind="typical"
                    # Equal near-level and approximately10cm examples; both signs.
                    h=float(rng.uniform(-.025,.025)) if variant%2==0 else float((1 if variant%4==1 else -1)*rng.uniform(.075,.125))
                elif variant<17:kind="camera_above";h=float(rng.uniform(.18,.38))
                else:kind="camera_below";h=-float(rng.uniform(.18,.38))
                cfg=e["config"]
                lo=max(.25,.12+h+ARMOR_CENTER_Z_MID)
                hi=min(1.1,1.3+h+ARMOR_CENTER_Z_MID)
                camera_z=float(rng.uniform(lo,hi))
                body_z=camera_z-h-ARMOR_CENTER_Z_MID
                distance=e["root_distance_m"]
                dx=float(np.sqrt(distance**2-(body_z-camera_z)**2))
                look_offset=np.asarray(cfg["camera_look_at_world_m"])-cfg["target_position_world_m"]
                target=np.array([5.2-dx,5.8,body_z])
                cfg.update(camera_position_world_m=[5.2,5.8,camera_z],
                           target_position_world_m=target.tolist(),
                           camera_look_at_world_m=(target+look_offset).tolist(),
                           camera_roll_deg=0.)
                cfg["target_rpy_deg"][:2]=[0.,0.]
                e.update(id=f"height40-train-v{variant:02d}-b{band}-l{j%3}-m{j%4}",
                         split="train",frames=1 if e["mode"]=="stationary" else 2,
                         planned_camera_minus_armor_mid_height_m=h,height_stratum=kind)
                episodes.append(e)
    result.update(episodes=episodes,train_episodes=len(episodes),dev_episodes=0,sealed_episodes=0,
        roll_fixed_zero=True,target_level_fixed_zero=True,
        scope="Additional TRAIN-only unique configurations inside user height scope; existing DEV unchanged; no confirmation data",
        height_scope_m=[-.4,.4],
        height_definition="camera optical center worldZ minus physical armor center worldZ",
        height_allocation="Within each distance/light/motion cell:14/20 typical,3/20 positive boundary,3/20 negative boundary",
        stationary_duplicate_policy="One exposure per static configuration; two per moving configuration",
        planning_armor_center_z_mid_m=ARMOR_CENTER_Z_MID,
        caps={"frames":sum(e["frames"] for e in episodes),"seconds":2400,
              "maximum_new_bytes":int(2.5*2**30),"minimum_free_bytes":int(1.5*2**30)})
    assert len(episodes)==1920 and result["caps"]["frames"]==3360
    return result

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plan-only",action="store_true");parser.add_argument("--plan-output")
    parser.add_argument("--pilot",action="store_true")
    a,remaining=parser.parse_known_args()
    recipe=plan()
    if a.pilot:
        recipe["episodes"]=recipe["episodes"][:8]+[recipe["episodes"][i*12] for i in range(1,8)]
        recipe["caps"]["frames"]=sum(e["frames"] for e in recipe["episodes"])
        recipe["train_episodes"]=len(recipe["episodes"])
        recipe["scope"]="TRAIN collection identity integration pilot; excluded from final confirmation"
    if a.plan_only:
        if a.plan_output:
            with open(a.plan_output,"x") as f:json.dump(recipe,f,indent=2)
        print(json.dumps({"episodes":len(recipe["episodes"]),"frames":recipe["caps"]["frames"],"split":"train","height_scope_m":recipe["height_scope_m"]}))
        return
    def interrupted(signum, frame):
        raise InterruptedError("TRAIN collection interrupted; releasing simulator")
    signal.signal(signal.SIGTERM, interrupted)
    collector.plan=lambda:recipe
    collector.episode_validator=validate_height_episode
    sys.argv=[sys.argv[0]]+remaining
    collector.main()
if __name__=="__main__":main()
