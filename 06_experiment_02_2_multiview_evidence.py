from pathlib import Path
import argparse,pickle,time
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from ferf_common import select_predictors,build_views,preprocessor,xgb,Quality,integrity,temporal,metric_set,save_json

PROJECT=Path(r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions\Experiments")
CLEAN=PROJECT/"Results"/"Experiment_01_Data_Preparation"/"Phase_02_Integrity_Verification"/"Step_02_Data_Cleaning"
EXP21=PROJECT/"Results"/"Experiment_03_Framework_Validation"/"Experiment_02_Original_Feature_Representation"/"Phase_01_Original_Feature_Evaluation"
OUT=PROJECT/"Results"/"Experiment_03_Framework_Validation"/"Experiment_02_Multiview_Evidence"/"Phase_02_View_Models"
def discover(d):
    p=sorted(d.glob("cleaned_part_*.parquet")); return p or sorted(d.glob("cleaned_part_*.csv.gz"))
def read(p): return pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p,compression="gzip",low_memory=False)
def count(p):
    if p.suffix==".parquet":
        import pyarrow.parquet as pq; return pq.ParquetFile(p).metadata.num_rows
    return sum(len(x) for x in pd.read_csv(p,compression="gzip",usecols=[0],chunksize=250000))
def sample(dataset,max_rows,seed):
    ps=discover(CLEAN/dataset/"Cleaned_Data"); sizes=[count(p) for p in ps]; total=sum(sizes); target=total if max_rows<=0 else min(total,max_rows)
    exact=[target*s/total for s in sizes]; alloc=[min(s,int(np.floor(v))) for s,v in zip(sizes,exact)]
    for i in np.argsort([v-np.floor(v) for v in exact])[::-1]:
        if sum(alloc)>=target:break
        if alloc[i]<sizes[i]:alloc[i]+=1
    frames=[]
    for i,(p,n,s) in enumerate(zip(ps,alloc,sizes),1):
        if n<=0:continue
        f=read(p); frames.append(f if n>=s else f.sample(n=n,random_state=seed+i))
    return pd.concat(frames,ignore_index=True,sort=False),total
def run(dataset,max_rows,seed):
    out=OUT/dataset
    for d in ("Models","Predictions","Reports","Manifests"):(out/d).mkdir(parents=True,exist_ok=True)
    raw,total=sample(dataset,max_rows,seed); X,y,removed=select_predictors(raw)
    a=pd.read_csv(EXP21/dataset/"Splits"/"fixed_split_assignments.csv").sort_values("row_position")
    if len(a)!=len(X):raise ValueError("Sample/split mismatch")
    split={n:a.loc[a["split"].eq(n),"row_position"].to_numpy(int) for n in ("train","validation","test")}
    views=build_views(X); source=raw.loc[X.index,"source_file"] if "source_file" in raw else pd.Series(["unknown"]*len(X))
    integ=integrity(CLEAN/dataset,source.reset_index(drop=True)); temp=temporal(raw.loc[X.index].reset_index(drop=True))
    pred=pd.DataFrame({"true_label":y.reset_index(drop=True),"split":""})
    for n,idx in split.items():pred.loc[idx,"split"]=n
    report=[]
    for name,cols in views.items():
        model=Pipeline([("preprocessing",preprocessor(X.iloc[split["train"]][cols])),("classifier",xgb(seed))])
        start=time.perf_counter();model.fit(X.iloc[split["train"]][cols],y.iloc[split["train"]]);elapsed=time.perf_counter()-start
        q=Quality().fit(X.iloc[split["train"]][cols]).transform(X[cols]);p=np.full(len(X),np.nan)
        for part in ("validation","test"):p[split[part]]=model.predict_proba(X.iloc[split[part]][cols])[:,1]
        pred[f"{name}__probability"]=p;pred[f"{name}__integrity"]=integ;pred[f"{name}__quality"]=q;pred[f"{name}__temporal"]=temp if name=="temporal" else np.ones(len(X))
        report.append({"view":name,"features":len(cols),"training_seconds":elapsed,**{f"validation_{k}":v for k,v in metric_set(y.iloc[split["validation"]],p[split["validation"]]).items()},**{f"test_{k}":v for k,v in metric_set(y.iloc[split["test"]],p[split["test"]]).items()}})
        with (out/"Models"/f"{name}.pkl").open("wb") as f:pickle.dump(model,f)
    pred.to_csv(out/"Predictions"/"view_predictions_and_reliability.csv",index=False)
    pd.DataFrame(report).to_csv(out/"Reports"/"view_metrics.csv",index=False)
    save_json(out/"Manifests"/"design.json",{"dataset":dataset,"rows":len(X),"total_available":total,"views":views,"removed":removed,"scope":"multi-view network-flow evidence"})
def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["CICIDS2017","CSE-CIC-IDS2018"],required=True);p.add_argument("--max-rows",type=int,default=1000000);p.add_argument("--seed",type=int,default=42);a=p.parse_args();run(a.dataset,a.max_rows,a.seed)
if __name__=="__main__":main()
