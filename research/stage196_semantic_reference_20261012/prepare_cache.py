"""Bind frozen label-free caches to a single input protocol and model identity."""
import os
os.umask(0o077)
from pathlib import Path
import hashlib
import json
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    audit=json.loads((ROOT/'DATA_AUDIT.json').read_text())
    if audit['state']!='AUDITED_WITH_LIMITATIONS' or audit['cross_patient_shared_image_hashes']:
        raise RuntimeError('Do not infer before the duplicate-image gate clears')
    rows=json.loads((ROOT/'private/COHORT.json').read_text())
    cfg=json.loads((BASE/'05_rah_confirmation/exact_configs/confirmation_v1.json').read_text())
    assert cfg['frame_indices']==[0,3,6,9]
    old=json.loads((BASE/'194_qwen_semantic_evidence_20261011/PRIVATE_LABEL_FREE_MANIFEST.json').read_text())
    cache={};manifest={};missing=[]
    for row in rows:
        key=row['key'];name=hashlib.sha256(key.encode()).hexdigest()[:24]+'.npz'
        source=BASE/'194_qwen_semantic_evidence_20261011/private_intervention_cache'/name
        if key in old:
            item={k:old[key][k] for k in ('oct_images','colpo_paths','mask')}
            assert len(item['mask'])==20
        else:
            site_images={}
            for s in range(1,13):
                frames=row['oct_sites'].get(str(s),[])
                if not frames:continue
                if row['old_input'] is not None:
                    path=Path(row['old_input']['oct_images'][str(s)])
                else:
                    path=ROOT/'private/new_oct_montages'/f'{key}_C{s:02d}.jpg'
                    if not path.exists():
                        path.parent.mkdir(parents=True,exist_ok=True);tile=cfg['collage_tile']
                        canvas=Image.new('RGB',(tile*2,tile*2),'black')
                        for slot,index in enumerate(cfg['frame_indices']):
                            if index<len(frames):
                                with Image.open(frames[index]) as image:canvas.paste(image.convert('RGB').resize((tile,tile)),((slot%2)*tile,(slot//2)*tile))
                        canvas.save(path,quality=92)
                site_images[str(s)]=str(path.resolve())
            colpo=row['colpo_paths'][:8]
            item=dict(oct_images=site_images,colpo_paths=colpo,
                      mask=[str(s) in site_images for s in range(1,13)]+[i<len(colpo) for i in range(8)])
        assert set(item)=={'oct_images','colpo_paths','mask'}
        manifest[key]=item
        images=[p for p in item['oct_images'].values()]+item['colpo_paths']
        input_hash=hashlib.sha256(json.dumps([(str(Path(p).resolve()),digest(p)) for p in images],sort_keys=True).encode()).hexdigest()
        if source.exists():cache[key]=dict(path=str(source.resolve()),sha256=digest(source),input_hash=input_hash,origin='unchanged frozen Stage194')
        else:missing.append(key)
    save(ROOT/'private/CACHE_INDEX.json',cache)
    save(ROOT/'private/LABEL_FREE_INPUTS.json',manifest)
    save(ROOT/'private/MISSING_FROZEN_FEATURES.json',missing)
    save(ROOT/'CACHE_PREPARATION.json',dict(total_patients=len(rows),existing_frozen_cache=len(cache),new_inference_needed=len(missing),
        model='autotrust/JEV-27B-VL',revision='fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc',
        frozen_cache_is_fold_agnostic=True,fold_binding='source clinical prior/heads separately hash-locked',
        OCT='four-frame montage [0,3,6,9], original locked protocol; all methods identical',
        patient_adapter_raw_protocol='separate adaptation experiment, not mixed into this factorial',
        cache_index_sha256=digest(ROOT/'private/CACHE_INDEX.json'),
        label_free_input_sha256=digest(ROOT/'private/LABEL_FREE_INPUTS.json')))
    print(json.dumps(dict(existing=len(cache),new=len(missing))),flush=True)


if __name__=='__main__':main()
