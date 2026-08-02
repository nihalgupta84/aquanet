"""Reproducible Phase 3 audit and experiment pipeline for AquaNet."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, hashlib, json, os, random, time
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel, wilcoxon

from dataset.water_dataset import WaterQualityDataset, get_weighted_sampler, CLASSES
from dataset.transforms import get_transforms
from models.proposed.aquanet_v3 import AquaNetV3
from models.deep_learning.dl_baselines import get_dl_baseline_model
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.soft_gating import SoftProbabilisticGating


OUT=ROOT/'phase3_results'; CKPT=ROOT/'checkpoints'/'phase3'
OUT.mkdir(exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2))

def image_files(root):
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})

def split_name(p):
    s=str(p)
    if 'cleaned_water_dataset' in s:
        for x in ('train','val','test'):
            if f'/{x}/' in s:return 'D1-'+x
    return 'D2' if 'finetune' in s else 'D3'

def dhash(p):
    a=np.asarray(Image.open(p).convert('L').resize((9,8),Image.Resampling.LANCZOS))
    bits=(a[:,1:]>a[:,:-1]).flatten(); return int(''.join('1' if x else '0' for x in bits),2)

def audit():
    files=image_files(ROOT/'data'); md5=defaultdict(list); ph=[]; counts=Counter()
    for p in files:
        counts[split_name(p)]+=1
        md5[hashlib.md5(p.read_bytes()).hexdigest()].append(str(p.relative_to(ROOT)))
        ph.append((p,dhash(p)))
    exact=[v for v in md5.values() if len(v)>1]
    exact_cross=[v for v in exact if len({split_name(ROOT/x) for x in v})>1]
    near=[]
    for i,(a,ha) in enumerate(ph):
        for b,hb in ph[i+1:]:
            if split_name(a)!=split_name(b) and (ha^hb).bit_count()<=5:
                near.append({'a':str(a.relative_to(ROOT)),'b':str(b.relative_to(ROOT)),'distance':(ha^hb).bit_count()})
    result={'files':len(files),'counts':dict(counts),'exact_duplicate_groups':exact,
            'cross_split_exact_groups':exact_cross,'cross_split_near_pairs_threshold_5':near,
            'method':{'exact':'MD5','perceptual':'64-bit dHash','hamming_threshold':5}}
    dump('dataset_audit.json',result); return result

def loaders(batch=64):
    root=ROOT/'data'/'cleaned_water_dataset'; tr=get_transforms(224,True); ev=get_transforms(224,False)
    train=WaterQualityDataset(root,'train',tr); val=WaterQualityDataset(root,'val',ev); test=WaterQualityDataset(root,'test',ev)
    return train,val,test,DataLoader(train,batch_size=batch,sampler=get_weighted_sampler(train),num_workers=4),DataLoader(val,batch_size=batch,num_workers=4),DataLoader(test,batch_size=batch,num_workers=4)

def make_model(model,variant,pretrained=True):
    if model=='aquanet_v3':
        return AquaNetV3(7,pretrained,use_msrb=variant!='no_msrb',use_csab=variant!='no_csab')
    return get_dl_baseline_model(model,7,pretrained)

def probs(model,x,name,variant,gating):
    out=model(x)
    if name=='aquanet_v3':
        return torch.softmax(out['flat_logits'],1) if variant=='flat' else gating(out['binary_logits'],out['type_logits'])
    return torch.softmax(out,1)

def ece(prob,y,bins=15):
    conf,pred=prob.max(1); val=0.
    for lo in torch.linspace(0,1,bins+1)[:-1]:
        hi=lo+1/bins; mask=(conf>lo)&(conf<=hi)
        if mask.any(): val+=mask.float().mean()*abs(pred[mask].eq(y[mask]).float().mean()-conf[mask].mean())
    return float(val)

def train_one(name,variant,seed,epochs=30,batch=16):
    set_seed(seed); train,val,test,tr,va,te=loaders(batch); device=torch.device('cuda')
    model=make_model(name,variant,True).to(device); gating=SoftProbabilisticGating().to(device)
    cnt=np.bincount(train.labels,minlength=7); weights=torch.tensor(1/(cnt+1e-5),dtype=torch.float32,device=device); weights/=weights.sum()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3)
    scaler=torch.amp.GradScaler('cuda')
    best=float('inf'); path=CKPT/f'{name}_{variant}_seed{seed}.pth'; history=[]
    for epoch in range(1,epochs+1):
        model.train()
        for x,y,*_ in tr:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out=model(x)
                if name=='aquanet_v3':
                    flat=nn.functional.cross_entropy(out['flat_logits'],y,weight=weights)
                    if variant=='flat': loss=flat
                    else: loss=.5*nn.functional.nll_loss(torch.log(gating(out['binary_logits'],out['type_logits'])+1e-7),y,weight=weights)+.5*flat
                else:
                    ce=nn.functional.cross_entropy(out,y,reduction='none',weight=weights); pt=torch.exp(-ce); loss=(((1-pt)**2)*ce).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); total=0.; yp=[]; yt=[]
        with torch.no_grad():
            for x,y,*_ in va:
                x,y=x.to(device),y.to(device); p=probs(model,x,name,variant,gating); loss=nn.functional.nll_loss(torch.log(p+1e-7),y,weight=weights)
                total+=loss.item()*len(y); yp+=p.argmax(1).cpu().tolist(); yt+=y.cpu().tolist()
        vl=total/len(val); sch.step(vl); history.append({'epoch':epoch,'val_loss':vl,'val_accuracy':accuracy_score(yt,yp),'val_macro_f1':f1_score(yt,yp,average='macro')}); print(f'{name} {variant} seed={seed} epoch={epoch} val_loss={vl:.4f} val_acc={history[-1]["val_accuracy"]:.4f}',flush=True)
        if vl<best: best=vl; torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path,map_location=device)); model.eval(); allp=[]; ally=[]; lat=[]
    with torch.no_grad():
        for x,y,*_ in te:
            x=x.to(device); torch.cuda.synchronize(); t=time.perf_counter(); p=probs(model,x,name,variant,gating); torch.cuda.synchronize(); lat.append((time.perf_counter()-t)*1000/len(x)); allp.append(p.cpu()); ally.append(y)
    p=torch.cat(allp); y=torch.cat(ally); m=compute_metrics(y,p.argmax(1),CLASSES)
    m.update({'ece_15bin':ece(p,y),'nll':float(nn.functional.nll_loss(torch.log(p+1e-7),y)),'latency_ms_per_image':float(np.mean(lat)),
              'params_m':sum(q.numel() for q in model.parameters())/1e6,'model':name,'variant':variant,'seed':seed,'history':history})
    dump(f'{name}_{variant}_seed{seed}.json',m)
    # RESEARCH_PLAN.md sec.9: per-image IDs/labels/probabilities/predictions must be saved.
    # Eval-time only; does not touch training and cannot change any metric above.
    save_predictions(f'{name}_{variant}_seed{seed}',test,y,p,{'model':name,'variant':variant,'seed':seed,'protocol':'phase3'})
    return m

PRED=ROOT/'predictions'/'phase3'

def save_predictions(tag,dataset,y,p,meta):
    """Write per-image predictions for McNemar / bootstrap (RESEARCH_PLAN.md sec.9)."""
    PRED.mkdir(parents=True,exist_ok=True)
    paths=[str(Path(s[0]).relative_to(ROOT)) for s in dataset.samples]
    assert len(paths)==len(y), f'{len(paths)} paths vs {len(y)} labels: eval loader order is not dataset order'
    rec={'meta':meta,'classes':list(CLASSES),
         'image_path':paths,'y_true':[int(v) for v in y.tolist()],
         'y_pred':[int(v) for v in p.argmax(1).tolist()],
         'y_prob':[[round(float(v),6) for v in row] for row in p.tolist()]}
    (PRED/f'{tag}.json').write_text(json.dumps(rec))
    print(f'[preds] wrote {PRED/f"{tag}.json"} ({len(paths)} images)',flush=True)

def aggregate():
    rows=[]
    for p in OUT.glob('*_seed*.json'):
        x=json.loads(p.read_text()); rows.append({k:x[k] for k in ('model','variant','seed','accuracy','macro_f1','weighted_f1','ece_15bin','nll','latency_ms_per_image','params_m')})
    groups=defaultdict(list)
    for r in rows: groups[(r['model'],r['variant'])].append(r)
    summary=[]
    for (m,v),rs in groups.items():
        z={'model':m,'variant':v,'seeds':[r['seed'] for r in rs]}
        for key in ('accuracy','macro_f1','weighted_f1','ece_15bin','nll','latency_ms_per_image','params_m'):
            z[key+'_mean']=float(np.mean([r[key] for r in rs])); z[key+'_std']=float(np.std([r[key] for r in rs],ddof=1)) if len(rs)>1 else None
        summary.append(z)
    stats={}
    full=groups.get(('aquanet_v3','full'),[])
    for base in ('resnet50','mobilenetv2'):
        br=groups.get((base,'full'),[])
        if len(full)==len(br) and len(full)>1:
            a=np.array([x['accuracy'] for x in sorted(full,key=lambda z:z['seed'])]); b=np.array([x['accuracy'] for x in sorted(br,key=lambda z:z['seed'])])
            stats[base]={'paired_t_p':float(ttest_rel(a,b).pvalue),'wilcoxon_p':float(wilcoxon(a,b).pvalue),'mean_difference':float((a-b).mean())}
    out={'runs':rows,'summary':summary,'paired_statistics':stats}; dump('phase3_summary.json',out); return out

JOBS=[('aquanet_v3','full',s) for s in (7,21,42)]+[('resnet50','full',s) for s in (7,21,42)]+[('mobilenetv2','full',s) for s in (7,21,42)]+[('aquanet_v3',v,s) for v in ('no_msrb','no_csab','flat') for s in (7,21,42)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit-only',action='store_true')
    ap.add_argument('--no-audit',action='store_true',help='skip the O(n^2) dHash audit (it is already in dataset_audit.json)')
    ap.add_argument('--list-missing',action='store_true',help='print jobs with no result JSON, one "model variant seed" per line, then exit')
    ap.add_argument('--model'); ap.add_argument('--variant'); ap.add_argument('--seed',type=int)
    ap.add_argument('--aggregate-only',action='store_true')
    a=ap.parse_args()
    if a.list_missing:
        for m,v,s in JOBS:
            if not (OUT/f'{m}_{v}_seed{s}.json').exists(): print(f'{m} {v} {s}')
        return
    if a.aggregate_only: aggregate(); return
    if not a.no_audit: audit()
    if a.audit_only: return
    # Single-job mode: lets an external runner execute jobs in parallel.
    # NOTE: batch=16 and num_workers=4 are LOCKED here. The 12 completed runs used them,
    # and both alter results (batch changes the gradient; worker count changes the
    # augmentation RNG streams). Changing either breaks seed-matched comparability.
    if a.model:
        target=OUT/f'{a.model}_{a.variant}_seed{a.seed}.json'
        if target.exists(): print(f'[skip] {target.name} exists'); return
        train_one(a.model,a.variant,a.seed); return
    for j in JOBS:
        if not (OUT/f'{j[0]}_{j[1]}_seed{j[2]}.json').exists(): train_one(*j)
    aggregate()
if __name__=='__main__':main()
