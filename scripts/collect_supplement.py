"""Pre-score acquisition amendment: every distance must have all4 motions in dev/test."""
import collect_broad as c
original_plan=c.plan
def plan():
 p=original_plan(20261023);episodes=[]
 for r in p['episodes']:
  j=int(r['id'].split('-')[-1])
  if j>=4:continue
  r['id']='supplement-'+r['id'];r['split']='sealed' if j<2 else 'dev'
  if j==0:r['light']='normal';r['config']['ambient_brightness']=220.;r['config']['directional_illuminance_lux']=10000.
  episodes.append(r)
 p.update(episodes=episodes,train_episodes=0,dev_episodes=16,sealed_episodes=16,amendment='Before any model score: supplement stationary/linear sealed episodes and spin/combined dev episodes in every range; ensure all4 motions and all3 light regimes in each split',caps={'frames':512,'seconds':600,'maximum_new_bytes':2*2**30,'minimum_free_bytes':20*2**30})
 return p
c.plan=plan
# c.main writes completion frames dynamically in the amended collection implementation.
if __name__=='__main__':c.main()
