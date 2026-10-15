import json
import pickle
import warnings
import numpy as np
import pandas as pd
from common import local_path, write_json, file_hash

def load_archive(model_index):
    folder=local_path(f"inputs/model_{model_index}")
    def one(pattern):
        matches=list(folder.glob(pattern))
        if len(matches)!=1:
            raise ValueError(f"Expected one {pattern}, got {len(matches)}")
        return matches[0]
    with one("source_tasks_*.pkl").open("rb") as f:
        source=pickle.load(f)
    with one("target_tasks_*.pkl").open("rb") as f:
        target=pickle.load(f)
    # Trusted copies of this project's own artifacts; scaler API changed after the original run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with (folder/"step_scaler.pkl").open("rb") as f:
            scaler=pickle.load(f)
    scale=lambda values: scaler.transform(np.asarray(values,dtype=np.float32).reshape(-1,1)).reshape(-1)
    source_params=np.load(folder/"source_params.npy").reshape(len(source),609)
    generated=np.load(one("sampleSeq_RealParams_*.npy")).reshape(len(target),609)
    record=json.loads((folder/"sample_run_check.json").read_text(encoding="utf-8"))
    assert record["source_task_ids"]==[int(t["task_id"]) for t in source]
    assert record["target_task_ids"]==[int(t["task_id"]) for t in target]
    for pattern,tasks in [("source_conditions_*.csv",source),("target_conditions_*.csv",target)]:
        conditions=pd.read_csv(one(pattern))
        assert len(conditions)==len(tasks)
        for col in ("WL","Retention","PEC"):
            assert np.array_equal(conditions[col].to_numpy(),[t[col] for t in tasks])
    return source,target,source_params,generated,scale

def condition_matrix(tasks):
    return np.array([[t["WL"],t["Retention"],t["PEC"]] for t in tasks],float)

def nearest_indices(source, target):
    x,z=condition_matrix(source),condition_matrix(target)
    scale=np.maximum(x.std(axis=0),1)
    return (((z[:,None,:]-x[None,:,:])/scale)**2).sum(axis=2).argmin(axis=1)

def prepare_bake():
    """Preserve bake in the curve identity, never infer physical time/temperature."""
    frame=pd.read_csv(local_path("data/pre-data.csv"))
    frame=frame.rename(columns={"步长":"step","wl":"WL","模型序号":"model","retention":"Retention","擦写次数":"PEC"})
    keys=["model","bake","WL","Retention","PEC","step"]
    grouped=frame.groupby(keys,sort=True,observed=True)["frequency"].mean().reset_index()
    stats=grouped.groupby(["model","bake","Retention"],observed=True).agg(records=("step","size"),wordlines=("WL","nunique"))
    stats.reset_index().to_csv(local_path("data/bake_coverage.csv"),index=False)
    with local_path("data/bake_tasks.jsonl").open("w",encoding="utf-8") as out:
        for key,group in grouped.groupby(keys[:-1],sort=True,observed=True):
            if len(group)<12:
                continue
            model,bake,wl,retention,pec=map(int,key)
            row=dict(model=model,bake=bake,WL=wl,Retention=retention,PEC=pec,
                     steps=group.step.tolist(),freqs=group.frequency.tolist(),num_points=len(group))
            out.write(json.dumps(row,separators=(",",":"))+"\n")
    write_json("data/preparation.json",dict(raw_rows=len(frame),unique_coordinates=len(grouped),
                curve_identity=keys[:-1],aggregation="mean only within identical model,bake,WL,Retention,PEC,step",
                physical_metadata="time,temperature,chip/die/block unavailable; do not infer"))

def load_bake_tasks(model,bake,source_limit,target_limit,seed,target_seed=None,excluded_target_ids=None):
    source,target=[],[]
    with local_path("data/bake_tasks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            if row["model"]!=model or row["bake"]!=bake:
                continue
            row["task_id"]=len(source)+len(target)
            if row["Retention"] in (1,2,3): source.append(row)
            elif row["Retention"] in (4,5,6): target.append(row)
    if excluded_target_ids:
        target=[task for task in target if task["task_id"] not in excluded_target_ids]
    def choose(tasks,limit,offset,selection_seed=None):
        # Stratified by retention and PEC, then balanced deterministic sampling.
        rng=np.random.default_rng((seed if selection_seed is None else selection_seed)+offset)
        groups={}
        for task in tasks: groups.setdefault((task["Retention"],task["PEC"]),[]).append(task)
        pools=[]
        for key,rows in sorted(groups.items()):
            order=rng.permutation(len(rows))
            pools.append([rows[i] for i in order])
        result=[]
        while pools and len(result)<limit:
            next_pools=[]
            for pool in pools:
                if len(result)<limit: result.append(pool.pop())
                if pool: next_pools.append(pool)
            pools=next_pools
        return sorted(result,key=lambda t:t["task_id"])
    source,target=choose(source,source_limit,0),choose(target,target_limit,1,target_seed)
    if not source or not target: raise ValueError("Empty bake domain")
    lo=min(min(t["steps"]) for t in source)
    hi=max(max(t["steps"]) for t in source)
    return source,target,lambda values: (np.asarray(values,float)-lo)/max(hi-lo,1)

def migration_manifest():
    rows=[]
    for folder in ("inputs","data","legacy_reference","docs"):
        for path in sorted(local_path(folder).rglob("*")):
            if path.is_file(): rows.append(dict(path=str(path.relative_to(local_path("."))),bytes=path.stat().st_size,sha256=file_hash(path)))
    write_json("migration_manifest.json",rows)
