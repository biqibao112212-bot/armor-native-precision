"""Reference existing native images and merge audited new TRAIN metadata only."""
import argparse,json,hashlib,shutil
from pathlib import Path
from armor_native.height_scope import height_region

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old',required=True);ap.add_argument('--new',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    old,new,out=map(Path,[a.old,a.new,a.output]);out.mkdir(exist_ok=False);seen=set();counts={}
    with (out/'train.jsonl').open('x') as f:
        for tag,root in [('old',old),('new',new)]:
            frames=objects=primary=0
            for line in (root/'train.jsonl').open():
                r=json.loads(line);assert r['split']=='train' and r['requested_level_target_verified'];key=(r['session'],r['frame_seq']);assert key not in seen;seen.add(key)
                assert all(abs(o['height_difference_m'])<=.40001 for o in r['objects'])
                for o in r['objects']:height_region(o['height_difference_m'])
                if tag=='new':assert 'bound_initial_position_world' in r and r['binding_frame_seq']<=r['ack_seq']<=r['frame_seq']
                f.write(json.dumps(r)+'\n');frames+=1;objects+=len(r['objects']);primary+=sum(3<=o['range_m']<=5 for o in r['objects'])
            counts[tag]={'frames':frames,'objects':objects,'primary_objects':primary,'source_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest()}
    shutil.copyfile(old/'dev.jsonl',out/'dev.jsonl');shutil.copyfile(new/'geometry.json',out/'geometry.json')
    dev_sessions={json.loads(l)['session'] for l in (out/'dev.jsonl').open()};assert not dev_sessions&{s for s,q in seen}
    (out/'merge.json').write_text(json.dumps({'complete':True,'counts':counts,'native_images_referenced_in_place':True,'new_confirmatory_data_used':False},indent=2));print(json.dumps(counts))
if __name__=='__main__':main()
