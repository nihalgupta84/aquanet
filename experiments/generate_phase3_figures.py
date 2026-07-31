import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'paper_v3'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((ROOT/'phase3_results'/'phase3_summary.json').read_text())
lookup={(x['model'],x['variant']):x for x in s['summary']}
models=[('AquaNet v3',lookup[('aquanet_v3','full')]),('ResNet50',lookup[('resnet50','full')]),('MobileNetV2',lookup[('mobilenetv2','full')])]
fig,ax=plt.subplots(figsize=(6.8,4)); x=np.arange(3); vals=[z['accuracy_mean']*100 for _,z in models]; err=[z['accuracy_std']*100 for _,z in models]
ax.bar(x,vals,yerr=err,capsize=5,color=['#2369a1','#dd8452','#55a868']); ax.set_xticks(x,[a for a,_ in models]); ax.set_ylabel('Test accuracy (%)'); ax.set_ylim(80,89); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'multiseed_accuracy.pdf'); plt.close(fig)
variants=[('Full',lookup[('aquanet_v3','full')]),('No MSRB',lookup[('aquanet_v3','no_msrb')]),('No CSAB',lookup[('aquanet_v3','no_csab')]),('Flat head',lookup[('aquanet_v3','flat')])]
fig,ax=plt.subplots(figsize=(6.8,4)); vals=[z['accuracy_mean']*100 for _,z in variants]; ax.bar(range(4),vals,color='#4c72b0'); ax.set_xticks(range(4),[a for a,_ in variants]); ax.set_ylabel('Seed-42 test accuracy (%)'); ax.set_ylim(82,91); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'ablation_accuracy.pdf'); plt.close(fig)
r=json.loads((ROOT/'phase3_results'/'aquanet_v3_full_seed42.json').read_text()); cm=np.asarray(r['confusion_matrix']); cm=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='.2f',cmap='Blues',xticklabels=['Clean','Algae','Debris','Foam','Oil','Turbid','Uncertain'],yticklabels=['Clean','Algae','Debris','Foam','Oil','Turbid','Uncertain'],ax=ax); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); fig.tight_layout(); fig.savefig(OUT/'confusion_matrix.pdf'); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.8,4)); labels=['D1 test','D3 pre-FT','D3 post-FT']; vals=[90.40,39.04,57.53]; ax.bar(labels,vals,color=['#4c72b0','#c44e52','#55a868']); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,100); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(OUT/'domain_transfer.pdf'); plt.close(fig)
