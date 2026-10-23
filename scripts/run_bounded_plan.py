"""Execute an explicitly saved sequence of bounded local experiment commands."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();path=Path(a.plan);plan=json.loads(path.read_text());root=path.parent;prefix=plan['state_prefix']
 def save(suffix,value):(root/(prefix+'-'+suffix+'.json')).write_text(json.dumps(value,indent=2))
 save('job',{'supervisor_pid':os.getpid(),'plan':str(path),'source_commit':plan['source_commit'],'started_unix':time.time(),'local_backup':False})
 env=os.environ.copy();env['PYTHONPATH']=str(Path(plan['source_directory'])/'src');start=time.monotonic()
 try:
  for name,argv,seconds in plan['commands']:
   stage_start=time.monotonic()
   with (root/(prefix+'-'+name+'.log')).open('x') as log:
    process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=plan['source_directory'],start_new_session=True)
    save('active',{'name':name,'pid':process.pid,'started_unix':time.time(),'command':argv,'timeout_s':seconds})
    try:rc=process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15);raise TimeoutError(name)
   if rc:raise RuntimeError(name+' exit '+str(rc))
   save(name+'-complete',{'complete':True,'returncode':rc,'seconds':time.monotonic()-stage_start})
  save('complete',{'complete':True,'seconds':time.monotonic()-start,'source_commit':plan['source_commit']})
 except Exception as e:
  save('failure',{'type':type(e).__name__,'message':str(e),'seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
