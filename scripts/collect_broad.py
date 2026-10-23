"""Bounded broad-domain collection, release SDK consumer; preserves all native frames."""
import argparse,json,os,subprocess,signal,time,struct,select,hashlib,shutil
from pathlib import Path
import numpy as np,cv2

def dump(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def plan(seed=20261022):
 rng=np.random.default_rng(seed);rows=[];bins=[(.6,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)]
 for b,(lo,hi) in enumerate(bins):
  for j in range(16):
   split='train' if j<12 else 'dev' if j<14 else 'sealed';distance=float(rng.uniform(lo+.04,hi-.04));hc=float(rng.uniform(.18,1.25));dz=float(rng.uniform(-min(.85,.65*distance),min(.85,.65*distance)));ht=float(np.clip(hc+dz,.12,1.3));dx=float(np.sqrt(distance**2-(ht-hc)**2));tx=5.2-dx
   light=['dark','normal','bright'][j%3];ambient,lux={'dark':(rng.uniform(5,40),rng.uniform(100,600)),'normal':(rng.uniform(120,350),rng.uniform(2000,16000)),'bright':(rng.uniform(600,1600),rng.uniform(35000,95000))}[light]
   # Offset the look point in world Y/Z to cover the image, without modifying K.
   uy=float(rng.uniform(-.28,.28)*distance);uz=float(rng.uniform(-.18,.18)*distance)
   cfg={'seed':seed+b*100+j,'target':3,'camera_position_world_m':[5.2,5.8,hc],'camera_look_at_world_m':[tx,5.8+uy,ht+.04+uz],'camera_roll_deg':float(rng.uniform(-12,12)),'target_position_world_m':[tx,5.8,ht],'target_rpy_deg':[float(rng.uniform(-12,12)),float(rng.uniform(-25,25)),float(rng.uniform(-180,180))],'ambient_brightness':float(ambient),'ambient_rgb':rng.uniform(.75,1,3).tolist(),'directional_illuminance_lux':float(lux),'light_direction_world':[float(rng.uniform(-1,1)),float(rng.uniform(-1,1)),-float(rng.uniform(.3,1))],'directional_rgb':rng.uniform(.7,1,3).tolist(),'shadows':j%2==0,'allow_partial_corner_labels':True}
   mode=['stationary','linear','spin','combined'][j%4]
   rows.append({'id':f'broad-b{b}-{j:02d}','split':split,'range_bin':[lo,hi],'root_distance_m':distance,'light':light,'frames':16,'mode':mode,'direction':float(rng.uniform(0,360)),'speed':float(rng.uniform(.25,min(1.2,.65*distance))),'span':float(min(.6,.3*distance)),'spin':float(rng.choice([-1,1])*rng.uniform(40,180)),'config':cfg})
 return {'schema':'armor-native.broad-collection/1','seed':seed,'release':'1.4.0-learning-r3','target_number':3,'native_pixels':[1440,1080],'resize':False,'split_unit':'independent continuous configuration and entire episode; no shared frames','train_episodes':96,'dev_episodes':16,'sealed_episodes':16,'label_schema':'daedalus.offline-exact-corners/2; missing slots unknown, never negatives','geometry':'physical armor-root corner order; full-frame/facing filtering recorded downstream','physical_blur':False,'augmentation_blur':'synthetic only, separately tagged','caps':{'frames':2048,'seconds':1800,'maximum_new_bytes':4*2**30,'minimum_free_bytes':20*2**30},'episodes':rows}
def read_exact(pipe,n,deadline):
 chunks=[]
 while n:
  if time.monotonic()>deadline:raise TimeoutError('frame acquisition deadline')
  if not select.select([pipe],[],[],min(1,max(.01,deadline-time.monotonic())))[0]:continue
  x=os.read(pipe.fileno(),n)
  if not x:raise EOFError('collector ended early')
  chunks.append(x);n-=len(x)
 return b''.join(chunks)
def main():
 a=argparse.ArgumentParser();a.add_argument('--workspace',required=True);a.add_argument('--output',required=True);a.add_argument('--sdk-root',required=True);a.add_argument('--capture-source');a.add_argument('--labels-file',help='Optional new protected annotation path on another filesystem; never overwritten');a.add_argument('--cpp-standard',default='c++17');args=a.parse_args();W=Path(args.workspace);out=Path(args.output);out.mkdir(exist_ok=False,parents=True);p=plan();dump(out/'plan.json',p);(out/'episodes').mkdir();(out/'ipc').mkdir()
 source=Path(args.capture_source) if args.capture_source else Path(__file__).with_name('capture_broad.cpp');binary=out/'capture-broad';cmd=['c++','-std='+args.cpp_standard,'-O2','-pthread','-I'+str(Path(args.sdk_root)/'include'),str(source),str(Path(args.sdk_root)/'lib/libdaedalus_sim_sdk.a'),'-o',str(binary)];dump(out/'build-command.json',cmd);subprocess.run(cmd,check=True,timeout=90)
 label_path=Path(args.labels_file).resolve() if args.labels_file else out/'exact-corners.jsonl';assert not label_path.exists();label_path.parent.mkdir(parents=True,exist_ok=True);dump(out/'label-source.json',{'path':str(label_path),'protected_asset':True})
 env=os.environ.copy();env.update(TALOS_IPC_DIR=str(out/'ipc'),DAEDALUS_CORNER_LABELS_JSONL=str(label_path))
 log=(out/'simulator.log').open('x');sim=subprocess.Popen([str(W/'tools/start-simulator-headless')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);dump(out/'job.json',{'pid':os.getpid(),'simulator_pid':sim.pid,'started_unix':time.time(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()});started=time.monotonic();total=0
 try:
  deadline=time.monotonic()+50
  while 'prediction runtime ready:' not in (out/'simulator.log').read_text(errors='replace'):
   if sim.poll() is not None:raise RuntimeError('simulator exited')
   if time.monotonic()>deadline:raise TimeoutError('simulator readiness')
   time.sleep(.2)
  for spec in p['episodes']:
   if time.monotonic()-started>p['caps']['seconds']:raise TimeoutError('collection global budget')
   d=out/'episodes'/spec['id'];d.mkdir();dump(d/'scenario.json',spec['config']);dump(d/'spec.json',spec)
   command=[str(binary),str(out/'ipc'),spec['id'],str(d/'scenario.json'),str(spec['frames']),spec['mode'],*[str(spec[k]) for k in ['direction','speed','span','spin']]]
   with (d/'collector.log').open('x') as clog,(d/'frames.jsonl').open('x') as ledger:
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=clog,bufsize=0);dump(d/'execution.json',{'command':command,'pid':proc.pid})
    try:
     for i in range(spec['frames']):
      if min(shutil.disk_usage(out).free,shutil.disk_usage(label_path.parent).free)<p['caps']['minimum_free_bytes']:raise RuntimeError('free space reserve reached')
      n=struct.unpack('<I',read_exact(proc.stdout,4,time.monotonic()+30))[0];assert 0<n<100000
      row=json.loads(read_exact(proc.stdout,n,time.monotonic()+10))
      if p.get('roll_fixed_zero'):
       from scipy.spatial.transform import Rotation
       rc=Rotation.from_quat(np.array(row['camera_quaternion_wxyz'])[[1,2,3,0]]).as_matrix();cfg=spec['config'];forward=np.array(cfg['camera_look_at_world_m'])-cfg['camera_position_world_m'];forward/=np.linalg.norm(forward);left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left);up=np.cross(forward,left);camera_roll=float(np.rad2deg(np.arctan2(rc[:,1]@up,rc[:,1]@left)));target_roll=float(Rotation.from_quat(np.array(row['target_quaternion_wxyz'])[[1,2,3,0]]).as_euler('ZYX',degrees=True)[-1]);assert abs(camera_roll)<.01 and abs(target_roll)<.01,(camera_roll,target_roll);row.update(actual_camera_roll_deg=camera_roll,actual_target_roll_deg=target_roll,requested_roll_zero_verified=True)
      if p.get('target_level_fixed_zero'):
       from scipy.spatial.transform import Rotation
       yaw,pitch,roll=Rotation.from_quat(np.array(row['target_quaternion_wxyz'])[[1,2,3,0]]).as_euler('ZYX',degrees=True)
       assert abs(pitch)<.01 and abs(roll)<.01,(pitch,roll)
       row.update(actual_target_pitch_deg=float(pitch),actual_target_roll_deg=float(roll),requested_level_target_verified=True)
      raw=read_exact(proc.stdout,6220800,time.monotonic()+20);rgba=np.frombuffer(raw,np.uint8).reshape(1080,1440,4);assert (rgba[:,:,3]==255).all();bgr=rgba[:,:,:3][:,:,::-1].copy();name=f'{i:03d}.png';assert cv2.imwrite(str(d/name),bgr,[cv2.IMWRITE_PNG_COMPRESSION,2]);assert np.array_equal(cv2.imread(str(d/name)),bgr)
      row.update(image_path=str((d/name).relative_to(out)),split=spec['split'],light=spec['light'],mode=spec['mode'],configured_range_bin=spec['range_bin'],rgb_sha256=hashlib.sha256(rgba[:,:,:3].tobytes()).hexdigest());ledger.write(json.dumps(row)+'\n');ledger.flush();total+=(d/name).stat().st_size
      if total>p['caps']['maximum_new_bytes']:raise RuntimeError('native image byte budget')
     code=proc.wait(timeout=10)
     if code:raise RuntimeError(f'collector exit {code}')
    finally:
     if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
   validator=globals().get('episode_validator')
   if validator is not None:validator(d,spec)
   print(json.dumps({'episode':spec['id'],'split':spec['split'],'completed_frames':sum(1 for x in (out/'episodes').glob('*/frames.jsonl') for _ in x.open()),'native_png_bytes':total}),flush=True)
  dump(out/'collection-result.json',{'complete':True,'frames':sum(e['frames'] for e in p['episodes']),'episodes':len(p['episodes']),'native_png_bytes':total,'seconds':time.monotonic()-started,'input_rgb_resized':False})
 finally:
  if sim.poll() is None:os.killpg(sim.pid,signal.SIGTERM);sim.wait(timeout=15)
  log.close()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  print(json.dumps({'failure':repr(e)}),flush=True);raise
