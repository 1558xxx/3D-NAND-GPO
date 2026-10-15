import numpy as np
import pandas as pd

def summaries(frame):
    keys=["state","protocol","budget","method","strength"]
    rows=[]
    for key,part in frame.groupby(keys,dropna=False):
        row=dict(zip(keys,key))
        row.update(curves=int(part.task_id.nunique()),seeds=int(part.seed.nunique()),runs=len(part))
        for metric in ["test_r2","test_nrmse_pct_peak","tail_test_rmse_pct_peak","tail_full_rmse_pct_peak","tail_unobserved_rmse_pct_peak","peak_error_steps","tail_test_symmetric_kl"]:
            values=part[metric].dropna().to_numpy()
            row[metric+"_mean"]=float(np.mean(values)) if len(values) else np.nan
            for q in [5,25,50,75,95]: row[metric+f"_p{q}"]=float(np.percentile(values,q)) if len(values) else np.nan
        row["negative_r2_fraction"]=float((part.test_r2<0).mean())
        row["classical_fallback_fraction"]=float(part.classical_fallback.mean())
        # Per-seed pooled point R2, followed by averaging across repeated runs.
        pooled=[]
        for _,run in part.groupby("seed"):
            denom=run.sum_y2.sum()-run.sum_y.sum()**2/run.test_n.sum()
            pooled.append(1-run.sse.sum()/denom if denom>0 else np.nan)
        row["point_r2_seed_mean"]=float(np.nanmean(pooled))
        rows.append(row)
    return pd.DataFrame(rows)

def paired_intervals(frame, repeats=2000, seed=20261013):
    """Resample curves as clusters after averaging repeated seeds; no fake N=curves*seeds."""
    rows=[]
    main=frame[frame.strength==0]
    keys=["state","protocol","budget"]
    rng=np.random.default_rng(seed)
    for key,part in main.groupby(keys):
        wide=part.groupby(["task_id","method"]).tail_test_rmse_pct_peak.mean().unstack("method")
        if "diffusion" not in wide: continue
        for baseline in wide.columns:
            if baseline=="diffusion": continue
            delta=(wide.diffusion-wide[baseline]).dropna().to_numpy()
            if not len(delta): continue
            means=np.mean(delta[rng.integers(0,len(delta),size=(repeats,len(delta)))],axis=1)
            rows.append(dict(zip(keys,key),baseline=baseline,metric="tail_test_rmse_pct_peak",
                             diffusion_minus_baseline=float(delta.mean()),ci95_low=float(np.quantile(means,.025)),
                             ci95_high=float(np.quantile(means,.975)),paired_curves=len(delta),bootstrap_repeats=repeats,
                             uncertainty_scope="curve sampling conditional on these models/seeds; not new-device uncertainty"))
    return pd.DataFrame(rows)

def paired_wordline_intervals(frame,repeats=2000,seed=20261013):
    """Resample WL clusters, preserving paired curves and repeated conditions within WL."""
    rows=[];rng=np.random.default_rng(seed)
    main=frame[frame.strength==0]
    for key,part in main.groupby(["state","protocol","budget"]):
        wide=part.groupby(["task_id","WL","method"]).tail_test_rmse_pct_peak.mean().unstack("method")
        if "diffusion" not in wide:continue
        for method in wide.columns:
            if method=="diffusion":continue
            delta=(wide.diffusion-wide[method]).dropna()
            if not len(delta):continue
            groups=delta.groupby(level="WL").agg(["sum","count"])
            indices=rng.integers(0,len(groups),size=(repeats,len(groups)))
            numerator=groups["sum"].to_numpy()[indices].sum(axis=1)
            denominator=groups["count"].to_numpy()[indices].sum(axis=1)
            distribution=numerator/denominator
            rows.append(dict(state=key[0],protocol=key[1],budget=key[2],baseline=method,
                diffusion_minus_baseline=float(delta.mean()),ci95_low=float(np.quantile(distribution,.025)),
                ci95_high=float(np.quantile(distribution,.975)),paired_curves=len(delta),wordline_clusters=len(groups),
                bootstrap_repeats=repeats,uncertainty_scope="WL clusters within archive; unknown device/block correlation remains"))
    return pd.DataFrame(rows)
