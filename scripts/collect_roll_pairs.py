"""Fixed before model scoring: paired real3D roll interventions, not image rotation."""
import collect_broad as c

def plan():
 episodes=[]
 for b in range(8):
  distance=[3.3,4.3][b%2];hc=.25 if (b//2)%2==0 else 1.1;ht=.45;tx=5.2-(distance**2-(hc-ht)**2)**.5;light='normal' if b<4 else 'dark'
  for j,(cr,tr) in enumerate([(0,0),(12,0),(-12,0),(0,12),(0,-12)]):
   cfg={'seed':2026102400+b,'target':3,'camera_position_world_m':[5.2,5.8,hc],'camera_look_at_world_m':[tx,5.8,ht+.04],'camera_roll_deg':cr,'target_position_world_m':[tx,5.8,ht],'target_rpy_deg':[tr,8 if b%2 else -8,15 if b%2 else -25],'ambient_brightness':220 if light=='normal' else 20,'ambient_rgb':[.92,.95,1.],'directional_illuminance_lux':10000 if light=='normal' else 300,'light_direction_world':[1.,0.,-1.],'directional_rgb':[.9,.95,1.],'shadows':False,'allow_partial_corner_labels':True}
   episodes.append({'id':f'roll-pair-{b:02d}-{j}','split':'sealed','range_bin':[3,5],'root_distance_m':distance,'light':light,'frames':4,'mode':'stationary','direction':0,'speed':0,'span':.5,'spin':0,'pair_group':b,'intervention':[cr,tr],'config':cfg})
 return {'schema':'armor-native.paired-roll/1','release':'1.4.0-learning-r3','target_number':3,'native_pixels':[1440,1080],'resize':False,'paired_groups':8,'episodes':episodes,'train_episodes':0,'dev_episodes':0,'sealed_episodes':40,'split_unit':'entire paired intervention group','selection':'eight independently fixed base conditions, five signed roll interventions, no model score used; never training or model selection','primary':'within-base roll0 versus camera/target roll+12/-12 separately; report failures and valid pose coverage','physical_blur':False,'caps':{'frames':160,'seconds':600,'maximum_new_bytes':2**30,'minimum_free_bytes':20*2**30}}
c.plan=plan
if __name__=='__main__':c.main()
