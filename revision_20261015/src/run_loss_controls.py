"""Separate seven-point controls for fitting objective and simple function-space priors."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erf
from sklearn.linear_model import Ridge
from common import local_path,write_json
from data_io import load_archive,nearest_indices,condition_matrix
from protocols import fixed_test_indices,nested_order
from metrics import score
from revision_stats import paired_intervals

def log_gaussian(x,y,grid,skew=False):
    span=max(float(np.ptp(x)),1);top=max(float(y.max()),1)
    initial=[top,float(np.average(x,weights=np.maximum(y,1))),max(span/6,.3),max(float(y.min()),0)]
    lower=[0,x.min()-span,.1,0];upper=[top*5+1,x.max()+span,span*3,top*2+1]
    if skew:initial.append(0);lower.append(-10);upper.append(10)
    def curve(xx,v):
        z=(xx-v[1])/v[2]
        shape=np.exp(-.5*z*z)
        if skew:shape=shape*(1+erf(v[4]*z/np.sqrt(2)))
        return v[3]+v[0]*shape
    solution=least_squares(lambda v:np.log1p(curve(x,v))-np.log1p(y),initial,bounds=(lower,upper),x_scale="jac",max_nfev=5000)
    return np.maximum(curve(grid,solution.x),0),not solution.success

def rbf_features(x):
    x=np.asarray(x,float).reshape(-1,1)
    centers=np.linspace(0,1,16)
    return np.column_stack([np.ones(len(x)),x,x*x,np.exp(-.5*((x-centers)/(.075))**2)])

def source_function_prior(source,target,scale):
    c=condition_matrix(source);ct=condition_matrix(target)
    cm,cs=c.mean(axis=0),np.maximum(c.std(axis=0),1)
    c=np.column_stack([np.ones(len(c)),(c-cm)/cs])
    ct=np.column_stack([np.ones(len(ct)),(ct-cm)/cs])
    features=[];labels=[]
    for condition,task in zip(c,source):
        phi=rbf_features(scale(task["steps"]))
        features.append((condition[None,:,None]*phi[:,None,:]).reshape(len(phi),-1))
        labels.append(np.log1p(task["freqs"]))
    model=Ridge(alpha=.01,fit_intercept=False).fit(np.concatenate(features),np.concatenate(labels))
    return ct@model.coef_.reshape(4,19)

def main():
    folder=local_path("results/loss_controls");folder.mkdir(parents=True,exist_ok=True)
    config=json.loads(local_path("configs/main.json").read_text(encoding="utf-8"))
    assert local_path("results/main/completion.json").exists(),"Main experiment must complete first"
    rows=[]
    for model in config["models"]:
        source,target,_,_,scale=load_archive(model)
        selected=np.sort(np.random.default_rng(config["curve_seed"]).choice(len(target),min(config["target_limit"],len(target)),replace=False))
        target=[target[i] for i in selected]
        prior=source_function_prior(source,target,scale)
        nearest=nearest_indices(source,target)
        for protocol in config["protocols"]:
            for i,task in enumerate(target):
                steps=np.asarray(task["steps"],float);truth=np.asarray(task["freqs"],float)
                test=fixed_test_indices(len(steps),max(config["budgets"]))
                peak=None if protocol=="uniform_nested" else source[nearest[i]]["steps"][np.argmax(source[nearest[i]]["freqs"])]
                anchors=nested_order(steps,test,peak)[:7]
                x,y=steps[anchors],truth[anchors]
                for method in ("log_gaussian","log_skew_gaussian","rbf_conditional_prior","rbf_scratch_ridge"):
                    fallback=False
                    if method.startswith("log_"):
                        pred,fallback=log_gaussian(x,y,steps,"skew" in method)
                    else:
                        design=rbf_features(scale(x));all_features=rbf_features(scale(steps))
                        initial=prior[i] if method=="rbf_conditional_prior" else np.zeros(19)
                        residual=np.log1p(y)-design@initial
                        vector=initial+design.T@np.linalg.solve(design@design.T+.01*np.eye(7),residual)
                        pred=np.maximum(np.expm1(np.clip(all_features@vector,-20,12)),0)
                    rows.append(dict(state=f"P{model-1}",protocol=protocol,budget=7,method=method,strength=0,seed=42,
                         task_id=task["task_id"],WL=task["WL"],Retention=task["Retention"],PEC=task["PEC"],
                         validation_n=0,classical_fallback=fallback,**score(truth,pred,steps,test,anchors)))
        print(f"P{model-1}: log-space and function-prior controls complete",flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(folder/"curve_metrics.csv",index=False)
    summary=frame.groupby(["state","protocol","method"],as_index=False).agg(test_r2_mean=("test_r2","mean"),
              tail_test_mean=("tail_test_rmse_pct_peak","mean"),test_nrmse_mean=("test_nrmse_pct_peak","mean"))
    summary.to_csv(folder/"summary.csv",index=False)
    original=pd.read_csv(local_path("results/main/curve_metrics.csv"))
    paired_intervals(pd.concat([original[(original.budget==7)&(original.method=="diffusion")&(original.strength==0)],frame],ignore_index=True),config["bootstrap_repeats"]).to_csv(folder/"paired_tail_ci.csv",index=False)
    write_json(folder/"completion.json",dict(status="complete",rows=len(frame),budget=7,source_ridge_alpha=.01,
          target_ridge_alpha=.01,rbf_centers=16,rbf_width=.075,scope="separate fixed controls; no target hyperparameter tuning",
          motivation="separate log-space fitting benefit and simple source-conditioned function prior from diffusion generation"))

if __name__=="__main__":main()
