"""Current stage: camera AND target roll fixed0; other basic domains remain broad."""
import sys
import collect_broad as c
original_plan=c.plan
confirmation='--confirmation' in sys.argv
if confirmation:sys.argv.remove('--confirmation')
def zero_roll(row,prefix,split,frames):
 row['id']=prefix+row['id'];row['split']=split;row['frames']=frames;row['config']['camera_roll_deg']=0.;row['config']['target_rpy_deg'][0]=0.;return row
def plan():
 p=original_plan(20261030 if confirmation else 20261028);episodes=[]
 if confirmation:
  for r in p['episodes']:
   if int(r['id'].split('-')[-1])<8:episodes.append(zero_roll(r,'roll0-confirm-','sealed',8))
  p.update(train_episodes=0,dev_episodes=0,sealed_episodes=64)
 else:
  for r in p['episodes']:
   j=int(r['id'].split('-')[-1]);episodes.append(zero_roll(r,'roll0-base-','train' if j<12 else 'dev',16))
  for r in original_plan(20261029)['episodes']:
   if int(r['id'].split('-')[-1])<4:episodes.append(zero_roll(r,'roll0-sealed-','sealed',16))
  p.update(train_episodes=96,dev_episodes=32,sealed_episodes=32)
 p.update(episodes=episodes,roll_fixed_zero=True,scope='camera roll0 and target roll0; keep genuine pitch/relative-height/yaw/light/motion coverage; nonzero roll deferred by user',confirmation_only=confirmation,caps={'frames':sum(r['frames'] for r in episodes),'seconds':900 if confirmation else 1800,'maximum_new_bytes':2*2**30 if confirmation else 4*2**30,'minimum_free_bytes':20*2**30})
 return p
c.plan=plan
if __name__=='__main__':c.main()
