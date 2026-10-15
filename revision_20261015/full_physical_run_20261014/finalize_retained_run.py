"""Full-cohort retained-review audit. No cancelled extension is a dependency."""
from pathlib import Path
import json,sys,time,sqlite3
import numpy as np
import pandas as pd
from full_common import HERE,config,read_json,atomic_json,hash_file,load_data,parquet_write,selections
from queue_scope import PROFILE,required_job
from finalize_full_run import METRICS,read_parts,check_coverage,summarize,bootstrap,delivery

MAIN=['pooled_source','conditional_ridge','nearest_source','scratch','diffusion','selected_policy']
CLASSICAL=['gaussian','log_gaussian','skew_gaussian','log_skew_gaussian','spline','pchip']
PROTOCOLS={('nested',5),('nested',7),('nested',9),('default',0)}

def lambda_methods():return [f'lambda_{s:g}' for s in config()['lambda_values'] if s!=0]

def validate(frame,expected,seeds,method,protocols,tag):
    assert set(frame['method'])=={method},tag+' method mismatch'
    assert set(zip(frame.protocol,frame.budget))==protocols,tag+' protocol mismatch'
    return check_coverage(frame,expected,seeds,tag)

def describe(frame,key):
    rows=summarize(frame,key)
    seedrows=[]
    for row in rows:
        g=frame[(frame.method==row['method'])&(frame.protocol==row['protocol'])&(frame.budget==row['budget'])]
        for name in METRICS:
            v=g[name].to_numpy(float);v=v[np.isfinite(v)]
            for q in [.05,.25,.5,.75,.95,.99]:row[f'{name}_p{int(100*q):02}']=float(np.quantile(v,q)) if len(v) else None
            means=g.groupby('seed')[name].mean()
            row[name+'_seed_mean_std_ddof1']=float(means.std(ddof=1)) if len(means)>1 else None
        for seed,s in g.groupby('seed'):
            sr=dict(case=key,method=row['method'],protocol=row['protocol'],budget=row['budget'],seed=int(seed),curves=len(s),negative_r2_fraction=float((s.test_r2<0).mean()))
            for name in METRICS:sr[name+'_mean']=float(s[name].mean())
            seedrows.append(sr)
        if 'penalty_loss' in g and 'data_loss' in g:
            a=g.penalty_loss.to_numpy(float);b=g.data_loss.to_numpy(float);ok=np.isfinite(a)&np.isfinite(b)&(b>0)
            ratio=a[ok]/b[ok];row['penalty_data_ratio_defined_rows']=int(ok.sum())
            row['penalty_data_ratio_undefined_rows']=int((~ok).sum())
            for label,q in [('median',.5),('p95',.95)]:row['penalty_data_ratio_'+label]=float(np.quantile(ratio,q)) if len(ratio) else None
            row['penalty_loss_mean']=float(np.nanmean(a));row['data_loss_mean']=float(np.nanmean(b))
    return rows,seedrows

def paired(diff,base,key):
    result=[]
    for (method,protocol,B),g in base.groupby(['method','protocol','budget']):
        d=diff[(diff.protocol==protocol)&(diff.budget==B)]
        if not len(d):continue
        for metric in ['tail_test_rmse_pct_peak','test_nrmse_pct_peak']:
            a=d.groupby('task_id').agg(value=(metric,'mean'),WL=('WL','first'),PEC=('PEC','first'))
            b=g.groupby('task_id')[metric].mean().rename('reference');p=a.join(b,how='inner')
            assert len(p)==len(a)==g.task_id.nunique()
            delta=p.value-p.reference
            r=dict(case=key,method='diffusion',reference=method,protocol=protocol,budget=int(B),metric=metric,curves=len(p),
                   paired_win_fraction=float((delta<0).mean()),interpretation='negative delta favours diffusion; average seeds within curve before resampling')
            r.update(bootstrap(delta,p.WL,p.PEC,20261014));result.append(r)
    return result

def failure_records(frame,key,out,d):
    from sklearn.cluster import KMeans
    q=frame.copy();sst=q.sum_y2-q.sum_y**2/q.test_n
    q['test_sst']=sst;q['test_variance']=sst/q.test_n
    peaks=dict(zip(d['task_id'][d['target']],d['frequency'][d['target']].max(1)))
    peak=q.task_id.map(peaks).to_numpy(float)
    q['test_std_pct_peak']=100*np.sqrt(np.maximum(q.test_variance,0))/np.maximum(peak,1e-9)
    q['negative_r2']=q.test_r2<0
    q['category']=np.select([~np.isfinite(q.test_r2),q.negative_r2&(q.peak_error_steps>2),q.negative_r2],
        ['undefined_r2','negative_r2_peak_displaced','negative_r2_other'],default='nonnegative_r2')
    q['cluster']=-1
    for (protocol,B,seed),g in q.groupby(['protocol','budget','seed']):
        bad=g[g.negative_r2];cols=['test_nrmse_pct_peak','tail_test_rmse_pct_peak','peak_error_steps','tail_slope_rmse']
        finite=np.isfinite(bad[cols]).all(1);v=np.log1p(np.maximum(bad.loc[finite,cols].to_numpy(float),0))
        if len(v)>=3:
            v=(v-np.median(v,0))/np.maximum(np.percentile(v,75,axis=0)-np.percentile(v,25,axis=0),1e-9)
            k=min(3,len(np.unique(v,axis=0)))
            if k>1:q.loc[bad.index[finite],'cluster']=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(v)
    parquet_write(out/'failure_records.parquet',q)
    q.groupby(['protocol','budget','seed','category']).size().rename('curves').reset_index().to_csv(out/'failure_categories.csv',index=False)
    # Illustrative examples only; no population is removed from aggregate results.
    examples=[]
    for (protocol,B,seed),g in q.groupby(['protocol','budget','seed']):
        for reason,sortcol,ascending in [('worst_r2','test_r2',True),('worst_nrmse','test_nrmse_pct_peak',False),('small_test_variation','test_std_pct_peak',True)]:
            subset=g[g.negative_r2].sort_values(sortcol,ascending=ascending).head(1)
            for _,r in subset.iterrows():examples.append(dict(protocol=protocol,budget=int(B),seed=int(seed),task_id=int(r.task_id),selection=reason))
    atomic_json(out/'failure_examples.json',dict(examples=examples,scope='illustrative extremes; categories and clusters are numeric symptoms, not verified physical causes'))
    # Save the actual observed/test coordinates and predictions for each chosen example.
    index={int(d['task_id'][i]):(pos,int(i)) for pos,i in enumerate(d['target'])}
    for e in examples:
        pos,i=index[e['task_id']];start=pos//config()['gpu_batch']*config()['gpu_batch'];local=pos-start
        f=HERE/'results'/key/f"seed{e['seed']}/predictions/diffusion/{e['protocol']}_B{e['budget']}_{start:07}.npz"
        with np.load(f) as z:
            assert int(z['task_id'][local])==e['task_id'];pred=z['pred'][local]
        fit,test,_=selections(d,[i],e['protocol'],e['budget']);n=int(d['length'][i])
        pd.DataFrame(dict(step=d['steps'][i,:n],truth=d['frequency'][i,:n],prediction=pred[:n],
                         observed=np.isin(np.arange(n),fit[0]),test=np.isin(np.arange(n),test[0]))).to_csv(
            out/f"example_{e['protocol']}_B{e['budget']}_seed{e['seed']}_{e['task_id']}_{e['selection']}.csv",index=False)
    return q.groupby('category').size().to_dict()

def audit_case(rec):
    key=f"{rec['chip']}_{rec['experiment']}_P{rec['state']}";out=HERE/'analysis_retained'/key;out.mkdir(parents=True,exist_ok=True)
    marker=out/'complete.json'
    prior=read_json(marker)
    if prior and prior.get('scope')==PROFILE:return prior
    d=load_data(key);expected=d['task_id'][d['target']];seeds=config()['seeds'];coverage=[];summaries=[];seedrows=[];pairs=[]
    ref=read_parts(HERE/'results'/key/'classical')
    assert set(ref.method)==set(CLASSICAL)
    for m in CLASSICAL:coverage+=validate(ref[ref.method==m],expected,[0],m,PROTOCOLS,key+' classical '+m)
    r,s=describe(ref,key);summaries+=r;seedrows+=s
    for seed in seeds:
        folder=HERE/'results'/key/f'seed{seed}';g=read_json(folder/'generated/main.json')
        assert g['rows']==len(expected) and g['source_vectors']==len(d['src']) and g['min_source_exposure']>=2
        assert hash_file(folder/'generated/main.npy')==g['checksum']
        mark=read_json(folder/'regularization_complete.json');assert mark and mark['full_targets']==len(expected) and mark['lambda_values']==config()['lambda_values']
    diff=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/metrics/diffusion') for s in seeds],ignore_index=True)
    pairs+=paired(diff,ref,key)
    for m in MAIN+lambda_methods():
        q=diff if m=='diffusion' else pd.concat([read_parts(HERE/'results'/key/f'seed{s}/metrics/{m}') for s in seeds],ignore_index=True)
        protocols=PROTOCOLS if m in MAIN[:5] else {('default',0)} if m=='selected_policy' else {('nested',7)}
        coverage+=validate(q,expected,seeds,m,protocols,key+' '+m)
        r,s=describe(q,key);summaries+=r;seedrows+=s
        if m!='diffusion':pairs+=paired(diff,q,key)
    failures=failure_records(diff,key,out,d)
    for name,rows in [('summary',summaries),('seed_summary',seedrows),('paired_comparisons',pairs),('coverage',coverage)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    result=dict(status='complete',scope=PROFILE,case=key,target_curves=len(expected),source_curves=len(d['src']),seeds=seeds,
                coverage_groups=len(coverage),no_target_subsample=True,failure_categories={k:int(v) for k,v in failures.items()})
    atomic_json(marker,result);return result

def analysis():
    with sqlite3.connect(HERE/'tasks.sqlite') as db:jobs=db.execute('select * from jobs').fetchall()
    missing=[j[0] for j in jobs if required_job(j) and j[1] not in ('analysis','delivery') and j[4]!='complete']
    assert not missing,f'Required work incomplete: {missing[:10]}'
    assert read_json(HERE/'cost_complete.json',{}).get('scope')==PROFILE
    assert read_json(HERE/'runtime_validation.json',{}).get('status')=='passed'
    out=HERE/'analysis_retained';out.mkdir(exist_ok=True);records=[];manifest=read_json(HERE/'data/manifest.json')
    for rec in manifest['records']:
        r=audit_case(rec);records.append(r);atomic_json(out/'progress.json',dict(scope=PROFILE,completed_cohorts=len(records),total=28,updated=time.time()))
        print('Audited',r['case'],flush=True)
    for name in ['summary','seed_summary','paired_comparisons','coverage','failure_categories']:
        pd.concat([pd.read_csv(out/r['case']/f'{name}.csv').assign(case=r['case']) for r in records],ignore_index=True).to_csv(out/f'all_{name}.csv',index=False,encoding='utf-8-sig')
    inventory=[]
    for pattern in ['*.py','*.json','data/*.npz']:
        for f in HERE.glob(pattern):inventory.append(dict(path=str(f.relative_to(HERE)),bytes=f.stat().st_size,sha256=hash_file(f)))
    atomic_json(out/'reproduction_inventory.json',dict(scope=PROFILE,files=inventory))
    atomic_json(out/'audit.json',dict(scope=PROFILE,cohorts=records,all_retained_requirements_verified=True,
        cancelled_tasks_not_claimed_complete=[j[0] for j in jobs if not required_job(j) and j[4]!='complete'],
        optional_completed_archives=[j[0] for j in jobs if not required_job(j) and j[4]=='complete'],
        limits=['PEC and block confounded; intervals are within-cohort, not independent-device population uncertainty',
                'two source passes do not establish convergence','batch L-BFGS shares curvature and line search',
                'known positive support; no BER/ECC or firmware latency/energy claim','lambda grid is descriptive, not target-test model selection',
                'removed generator families, condition/T/source sweeps and rigorous bias-variance are not claimed completed']))
    atomic_json(HERE/'analysis_complete.json',dict(status='complete',scope=PROFILE,verified_cohorts=28,
        target_curves=manifest['target_curves'],source_curves=manifest['source_curves'],delivery_complete=False,updated=time.time()))

if __name__=='__main__':
    if sys.argv[1]=='analysis':analysis()
    elif sys.argv[1]=='delivery':
        assert read_json(HERE/'analysis_complete.json',{}).get('scope')==PROFILE
        delivery()
    else:raise ValueError(sys.argv[1])
