"""Expanded independent configurations; fixed zero roll, native RGB, disjoint episodes."""
import collect_broad as c
original=c.plan
def plan():
 seed=20261112;p=original(seed);rows=[]
 for variant in range(20):
  # A separate deterministic random draw for every continuous configuration.
  batch=original(seed+10000*(variant+1))['episodes']
  for b in range(8):
   for j in range(12):
    r=batch[b*16+j];r['id']=f'roll0-expanded-v{variant:02d}-b{b}-l{j%3}-m{j%4}'
    r['split']='train' if variant<16 else 'dev' if variant<18 else 'sealed';r['frames']=4
    r['config']['camera_roll_deg']=0.;r['config']['target_rpy_deg'][0]=0.;rows.append(r)
 p.update(episodes=rows,roll_fixed_zero=True,train_episodes=1536,dev_episodes=192,sealed_episodes=192,
  scope='1920 independent configurations:8 distance bins x3 light groups x4 motions x20 continuous view/height/pose variants; both rolls0',
  caps={'frames':7680,'seconds':7200,'maximum_new_bytes':9*2**30,'minimum_free_bytes':8*2**30})
 return p
c.plan=plan
if __name__=='__main__':c.main()
