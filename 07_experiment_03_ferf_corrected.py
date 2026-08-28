from pathlib import Path
import argparse
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score,matthews_corrcoef,roc_auc_score
from ferf_common import metric_set,save_json
PROJECT=Path(r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments")
SRC=PROJECT/"Results"/"Experiment_03_Framework_Validation"/"Experiment_02_Multiview_Evidence"/"Phase_02_View_Models"
OUT=PROJECT/"Results"/"Experiment_03_Framework_Validation"/"Experiment_03_FERF_Validation_Corrected"
ALPHA,BETA,GAMMA=.35,.45,.20
THRESHOLDS=np.arange(.20,.801,.01)
def simplex(n,step=.05):
    units=int(round(1/step))
    def rec(prefix,left,k):
        if k==1:yield prefix+[left];return
        for i in range(left+1):yield from rec(prefix+[i],left-i,k-1)
    for z in rec([],units,n):yield np.array(z,float)/units
def fuse(p,r,w):
    wr=r*w.reshape(1,-1);return (wr*p).sum(1)/np.maximum(wr.sum(1),1e-12)
def run(dataset):
    df=pd.read_csv(SRC/dataset/"Predictions"/"view_predictions_and_reliability.csv")
    views=[c[:-13] for c in df if c.endswith("__probability")]
    val=df["split"].eq("validation").to_numpy();test=df["split"].eq("test").to_numpy();y=df["true_label"].to_numpy(int)
    p=np.column_stack([df[f"{v}__probability"].to_numpy(float) for v in views])
    r=np.column_stack([np.clip(ALPHA*df[f"{v}__integrity"]+BETA*df[f"{v}__quality"]+GAMMA*df[f"{v}__temporal"],.05,1) for v in views])
    best=None;rows=[]
    for w in simplex(len(views)):
        pv=fuse(p[val],r[val],w);auc=roc_auc_score(y[val],pv)
        for t in THRESHOLDS:
            pr=(pv>=t).astype(int);ba=balanced_accuracy_score(y[val],pr);f=f1_score(y[val],pr);m=matthews_corrcoef(y[val],pr);key=(ba,f,m,auc,-abs(t-.5))
            rows.append({"threshold":t,"balanced_accuracy":ba,"f1":f,"mcc":m,"roc_auc":auc,**{f"weight_{v}":w[i] for i,v in enumerate(views)}})
            if best is None or key>best[0]:best=(key,w.copy(),float(t))
    _,w,t=best;ptest=fuse(p[test],r[test],w);result=metric_set(y[test],ptest,t);meanp=np.nanmean(p[test],axis=1);mean_result=metric_set(y[test],meanp,.5)
    singles=[(metric_set(y[test],p[test,i],.5),v) for i,v in enumerate(views)];single=max(singles,key=lambda x:x[0]["balanced_accuracy"])
    out=OUT/dataset
    for d in ("Reports","Predictions","Manifests"):(out/d).mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).sort_values(["balanced_accuracy","f1","mcc"],ascending=False).to_csv(out/"Reports"/"validation_search.csv",index=False)
    pd.DataFrame([{"method":"Best single view","view":single[1],**single[0]},{"method":"Unweighted mean","view":"all",**mean_result},{"method":"FERF","view":"all",**result}]).to_csv(out/"Reports"/"ferf_comparison.csv",index=False)
    pd.DataFrame({"true_label":y[test],"ferf_probability":ptest,"ferf_prediction":(ptest>=t).astype(int),"unweighted_probability":meanp}).to_csv(out/"Predictions"/"ferf_test_predictions.csv",index=False)
    save_json(out/"Manifests"/"ferf_configuration.json",{"views":views,"weights":dict(zip(views,w.tolist())),"threshold":t,"reliability":{"integrity":ALPHA,"quality":BETA,"temporal":GAMMA},"selection":"validation only","scope":"multi-view network-flow evidence"})
    print(dataset,result,dict(zip(views,w)),"threshold",t)
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["CICIDS2017","CSE-CIC-IDS2018"],required=True);a=p.parse_args();run(a.dataset)
if __name__=="__main__":main()
