#!/usr/bin/env python3
"""Generate the camera-ready figures from immutable aggregate results."""
from pathlib import Path
import json, shutil
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

report = json.loads((ROOT / "reports/final/final_report.json").read_text())
summary = report["summary"]
pretty = {"swin_tiny":"Swin-T", "deit_small":"DeiT-S", "convnext_tiny":"ConvNeXt-T",
 "resnet50":"ResNet-50", "densenet121":"DenseNet-121", "efficientnet_b0":"EfficientNet-B0",
 "mobilenetv2":"MobileNetV2", "aquanet[hier_tf,msrb=on,csab=off]":"AquaNet-hier",
 "aquanet[flat,msrb=off,csab=off]":"AquaNet-no-neck", "aquanet[flat,msrb=on,csab=on]":"AquaNet-full"}

def save(name):
    plt.tight_layout(); plt.savefig(OUT / name, bbox_inches="tight"); plt.close()

# Five-seed ranking with 95% t intervals.
names = [pretty[x["config"]] for x in summary][::-1]
means = [x["test_macro_f1_mean"] for x in summary][::-1]
lo = [x["test_macro_f1_ci95"][0] for x in summary][::-1]
hi = [x["test_macro_f1_ci95"][1] for x in summary][::-1]
colors = ["#c44e52" if "AquaNet" in n else "#4c72b0" for n in names]
plt.figure(figsize=(5.7, 3.4)); y=np.arange(len(names))
plt.errorbar(means,y,xerr=[np.array(means)-lo,np.array(hi)-means],fmt="none",ecolor="#444",capsize=2,lw=.9)
plt.scatter(means,y,c=colors,s=24,zorder=3); plt.yticks(y,names); plt.xlabel("Test macro-F1 (mean and 95% CI; five seeds)")
plt.axvline(.866061,color="#c44e52",ls="--",lw=.7); save("ranking.pdf")

# Identical-budget sensitivity.
sense={"ResNet-50":.1901,"ConvNeXt-T":.1636,"Swin-T":.1128,"DenseNet-121":.0935,"DeiT-S":.0836,
"EfficientNet-B0":.0742,"MobileNetV2":.0653,"AquaNet-no-neck":.0477,"AquaNet-hier":.0396,"AquaNet-full":.0324}
items=sorted(sense.items(),key=lambda z:z[1]); n=[x[0] for x in items]; v=[x[1] for x in items]
plt.figure(figsize=(5.7,3.3)); plt.barh(n,v,color=["#c44e52" if "AquaNet" in x else "#4c72b0" for x in n])
plt.xlabel("Best minus worst validation macro-F1 in identical tuning grid"); save("protocol_sensitivity.pdf")

# OOD transfer and adaptation.
ood={"Swin-T":.5035,"DeiT-S":.4980,"ConvNeXt-T":.4955,"AquaNet-no-neck":.4787,"ResNet-50":.4484,
"DenseNet-121":.4399,"EfficientNet-B0":.4349,"AquaNet-hier":.4094,"AquaNet-full":.3957,"MobileNetV2":.3726}
items=sorted(ood.items(),key=lambda z:z[1]); plt.figure(figsize=(5.7,3.3))
plt.barh([x[0] for x in items],[x[1] for x in items],color=["#c44e52" if "AquaNet" in x[0] else "#55a868" for x in items])
plt.xlabel("D3 zero-shot macro-F1 (146 images)"); save("ood_transfer.pdf")

plt.figure(figsize=(3.8,2.7)); x=[25,50,75,100]; y=[.5961,.6653,.6456,.7069]
plt.plot(x,y,"o-",color="#4c72b0"); plt.xticks(x); plt.ylim(.55,.74); plt.xlabel("Available D2 adaptation data (%)"); plt.ylabel("D3 macro-F1"); save("adaptation.pdf")

# Decision-quality and explanation summaries.
models=["Swin-T","DeiT-S","ConvNeXt-T","AquaNet-hier","AquaNet-full"]
aurc=[.0104,.0141,.0141,.0175,.0178]; corrupt=[.0387,.0255,.0315,.0843,.0811]
fig,ax=plt.subplots(1,2,figsize=(6.8,2.5)); ax[0].bar(models,aurc,color="#4c72b0"); ax[0].set_ylabel("AURC (lower is better)")
ax[1].bar(models,corrupt,color="#dd8452"); ax[1].set_ylabel("Mean corruption macro-F1 loss")
for a in ax: a.tick_params(axis="x",rotation=35)
save("decision_quality.pdf")

exp={"AquaNet-hier":(.7739,.9044),"ResNet-50":(.7747,.8914),"AquaNet-no-neck":(.7934,.9004),
"EfficientNet-B0":(.7845,.8824),"DenseNet-121":(.8088,.8973),"AquaNet-full":(.8330,.8870)}
plt.figure(figsize=(4.2,3.0))
for n,(d,i) in exp.items(): plt.scatter(d,i,label=n,s=30)
plt.xlabel("Deletion AUC (lower better)"); plt.ylabel("Insertion AUC (higher better)"); plt.legend(fontsize=6,loc="lower left"); save("explanations.pdf")

# Preserve one representative qualitative panel as a supplementary figure.
src=ROOT/"reports/stageF_figures/P_aquanet-flat-on-on_seed42_gradcam.png"
if src.exists(): shutil.copy2(src,OUT/"gradcam_aquanet_seed42.png")
print(f"Wrote figures to {OUT}")
