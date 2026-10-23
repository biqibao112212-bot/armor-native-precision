"""Native collection with target roll=pitch=0 and camera roll=0.

Relative height, camera look direction, target yaw, lighting and motion vary.
Old tilted-body episodes remain diagnostic data and are never silently relabeled.
"""
import collect_broad as c
original_plan = c.plan

def plan():
 seed = 2026102208
 p = original_plan(seed)
 rows = []
 for variant in range(24):
  batch = original_plan(seed + 10000 * (variant + 1))['episodes']
  for band in range(8):
   for j in range(12):
    r = batch[band * 16 + j]
    r['id'] = f'level-v{variant:02d}-b{band}-l{j%3}-m{j%4}'
    r['split'] = 'train' if variant < 20 else 'dev'
    r['frames'] = 4
    r['config']['camera_roll_deg'] = 0.
    r['config']['target_rpy_deg'][:2] = [0., 0.]
    rows.append(r)
 p.update(episodes=rows, roll_fixed_zero=True, target_level_fixed_zero=True,
  train_episodes=1920, dev_episodes=384, sealed_episodes=0,
  scope='Level target: world roll=pitch=0; camera roll=0. Vary yaw, position, relative height, camera pitch, light and motion. Number3 native pixels. Fresh confirmation is collected only after model freeze.',
  caps={'frames':9216, 'seconds':5400, 'maximum_new_bytes':int(6.75*2**30), 'minimum_free_bytes':2*2**30})
 return p

c.plan = plan
if __name__ == '__main__':
 c.main()
