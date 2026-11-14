"""Build evaluation inputs from released metadata and licensed local assets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    text=Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def under(root,relative):
    relative=Path(relative)
    if relative.anchor or '..' in relative.parts or ':' in str(relative):
        raise ValueError('Unsafe asset reference: '+str(relative))
    # Explicitly supplied local data roots may contain dataset symlinks.
    path=(Path(root).resolve()/relative).resolve()
    if not path.is_file():
        raise FileNotFoundError('Missing licensed image/original cache: '+str(path))
    return str(path)


def replace_queries(rows,patches,dataset,to_archived=False):
    result=copy.deepcopy(rows)
    key='groundbench_sample_id' if dataset=='groundbench' else 'degraded_sample_id'
    lookup={p[key]:p for p in patches}
    if len(lookup)!=17 or len(patches)!=17:
        raise ValueError('Expected exactly17 approved query patches')
    found=set()
    for r in result:
        if r['sample_id'] not in lookup:continue
        patch=lookup[r['sample_id']]
        before,after=(patch['new_query'],patch['old_query']) if to_archived else (patch['old_query'],patch['new_query'])
        if r['query'] not in (before,after):
            raise ValueError('Query differs from recorded versions: '+r['sample_id'])
        if r.get('prompt') not in (None,[]):
            raise ValueError('Nonempty prompt requires manual query synchronization')
        r['query']=after
        found.add(r['sample_id'])
    if found!=set(lookup):
        raise ValueError('Input does not contain all17 source IDs')
    return result


def degraded(args):
    release=Path(args.release_root)
    rows=read_degraded_annotations(release)
    if args.version=='v1-archived-queries':
        rows=replace_queries(rows,read(release/'query_patches_v2.json'),'degraded',True)
    else:
        rows=copy.deepcopy(rows)
    from PIL import Image
    for r in rows:
        r['images']=[under(args.image_root,p) for p in r['images']]
        if 'clean_images' in r:
            r['clean_images']=[under(args.image_root,p) for p in r['clean_images']]
        with Image.open(r['images'][0]) as image:
            if list(image.size)!=r['image_size']:
                raise ValueError('RGB dimensions changed: '+r['sample_id'])
            image.verify()
        with Image.open(r['images'][1]) as image:image.verify()
        w,h=r['image_size'];x1,y1,x2,y2=r['target_bbox']
        if not(0<=x1<x2<=w and 0<=y1<y2<=h):
            raise ValueError('Invalid pixel target box: '+r['sample_id'])
        r['dataset_version']='degraded-1k-evaluation' if args.version in ['evaluation','v2-quality-screening'] else args.version
    counts=[sum(r['degradation']==c for r in rows) for c in ['rgb_degraded','tir_degraded','both_degraded']]
    if counts!=[333,333,334]:raise ValueError('Wrong corruption-partition counts')
    return rows


def read_degraded_annotations(release):
    release=Path(release)
    path=release/'data/degraded.json'
    if not path.is_file():path=release/'data/degraded_v2.json'
    return read(path)


def dv_pair(root,image,dataset=None):
    if Path(image).name!=image:raise ValueError('Expected an image basename')
    candidates=[dataset] if dataset else ['LLVIP','HDRT']
    matches=[]
    for source in candidates:
        if source not in ['LLVIP','HDRT']:raise ValueError('Unknown DV image source')
        pair=[Path(root)/source/modality/image for modality in ['visible','infrared']]
        if all(p.is_file() for p in pair):matches.append(pair)
    if len(matches)!=1:
        raise ValueError('Missing or ambiguous paired image source: '+image)
    return [str(p.resolve()) for p in matches[0]]


def dv1k(args):
    selection=read(Path(args.release_root)/'selection.json')
    train,dv500=read(args.train),read(args.dv500)
    for name,path in [('train',args.train),('dv500',args.dv500)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=selection['sources'][name]:
            raise ValueError('Upstream annotation version differs: '+name)
    if len(dv500)!=500 or len(selection['samples'])!=500:
        raise ValueError('Expected500 RGB source rows and500 selected IR rows')
    rows=[]
    if not args.ir_only:
        for index,raw in enumerate(dv500):
            images=dv_pair(args.image_root,raw['image'])
            rows.append(dict(sample_id=f'dv1k_original_dv500_{index:06d}',images=images,image_paths=images,
                question=raw['text'],answer=raw['ground_truth_answer'],category='rgb',dataset_version='dv1k-selector-v1'))
    for e in selection['samples']:
        raw=train[e['source_row']]
        if raw['category']!='ir' or Path(raw['image']).name!=e['image']:
            raise ValueError('IR selector no longer matches source')
        pair=raw['conversations'][:2]
        if len(pair)!=2 or [x['from'] for x in pair]!=['human','gpt']:
            raise ValueError('Invalid upstream first-QA roles')
        images=dv_pair(args.image_root,e['image'],e['dataset'])
        rows.append(dict(sample_id=e['sample_id'],images=images,image_paths=images,
            question=pair[0]['value'].replace('<image>','').strip(),answer=pair[1]['value'].strip(),
            category='ir',dataset_version='dv1k-selector-v1'))
    return rows


def groundbench(args):
    rows=read(args.groundbench)
    return replace_queries(rows,read(Path(args.release_root)/'query_patches_v2.json'),'groundbench')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['degraded','dv1k','groundbench-v2'])
    p.add_argument('--release-root',required=True,help='Downloaded HF metadata directory')
    p.add_argument('--image-root',help='Local licensed image root; Degraded also requires the original cache')
    p.add_argument('--groundbench',help='Existing normalized10124-row GroundBench JSON, not raw upstream archives')
    p.add_argument('--train');p.add_argument('--dv500')
    p.add_argument('--version',choices=['evaluation','v2-quality-screening','v1-archived-queries'],default='evaluation')
    p.add_argument('--ir-only',action='store_true',help='Export added500 examples only; not the full DV-1K benchmark')
    p.add_argument('--output');p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.dataset in ['degraded','dv1k'] and not args.image_root:p.error('--image-root is required')
    if args.dataset=='dv1k' and not(args.train and args.dv500):p.error('--train and --dv500 are required')
    if args.dataset=='groundbench-v2' and not args.groundbench:p.error('--groundbench is required')
    if args.dataset!='dv1k' and args.ir_only:p.error('--ir-only applies only to DV')
    if args.dataset!='degraded' and args.version not in ['evaluation','v2-quality-screening']:p.error('--version applies only to Degraded')
    if not args.check_only and not args.output:p.error('--output is required unless --check-only')
    if args.output and Path(args.output).exists():p.error('Output exists; choose a new file')
    rows={'degraded':degraded,'dv1k':dv1k,'groundbench-v2':groundbench}[args.dataset](args)
    expected=10124 if args.dataset=='groundbench-v2' else (500 if args.ir_only else 1000)
    if len(rows)!=expected or len({r['sample_id'] for r in rows})!=expected:
        raise ValueError('Unexpected sample count or duplicate IDs')
    if not args.check_only:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as f:
            if args.dataset=='dv1k':
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            else:json.dump(rows,f,ensure_ascii=False,indent=2)
    print('Validated',len(rows),'records. No training/inference was run.')


if __name__=='__main__':main()
