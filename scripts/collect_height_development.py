"""Fresh DEV configurations, explicitly distinct from sealed confirmation."""
import sys
import collect_broad as collector
from collect_level_confirmation import plan as confirmation_layout,validate_height_episode

def plan():
    p=confirmation_layout(seed=2027102306)
    for e in p['episodes']:
        e['id']=e['id'].replace('level-height40-confirm','height40-development')
        e['split']='dev';e['frames']=1 if e['mode']=='stationary' else 2
    p.update(scope='Fresh DEV only for repaired near-range coverage; not sealed confirmation',train_episodes=0,dev_episodes=len(p['episodes']),sealed_episodes=0)
    p['caps']['frames']=sum(e['frames'] for e in p['episodes'])
    return p

def main():
    collector.plan=plan;collector.episode_validator=validate_height_episode;collector.main()
if __name__=='__main__':main()
