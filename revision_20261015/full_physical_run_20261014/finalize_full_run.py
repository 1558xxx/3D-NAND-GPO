"""Full-cohort audit, paired statistics and a guarded final-delivery stage.

Never creates a delivery-complete marker from numerical completion alone.
The final manuscript/reply PDFs must be rendered, inspected and listed by the
chat agent in delivery_release.json with verified hashes before final release.
"""
from pathlib import Path
import sys,time,json,sqlite3
import numpy as np
import pandas as pd
from full_common import HERE,ROOT,config,load_data,read_json,atomic_json,hash_file,parquet_write,selections

METRICS=['test_r2','test_nrmse_pct_peak','test_smape_pct','tail_test_rmse_pct_peak',
    'tail_full_rmse_pct_peak','tail_test_symmetric_kl','peak_error_steps','tail_slope_rmse']


def bootstrap(delta,wl,pec,seed):
    delta=np.asarray(delta,float);good=np.isfinite(delta);delta=delta[good];wl=np.asarray(wl)[good];pec=np.asarray(pec)[good]
    if not len(delta):return dict(paired_finite_n=0,curve_low=None,curve_high=None,WL_low=None,WL_high=None)
    rng=np.random.default_rng(seed);reps=config()['bootstrap'];curve=[]
    # True paired curve resampling, bounded temporary memory.
    block=max(1,min(32,1000000//len(delta)))
    for start in range(0,reps,block):
        indices=rng.integers(0,len(delta),size=(min(block,reps-start),len(delta)))
        curve.extend(delta[indices].mean(1))
    def clustered(labels):
        groups=pd.DataFrame(dict(label=labels,delta=delta)).groupby('label')['delta'].agg(['sum','count'])
        idx=rng.integers(0,len(groups),size=(reps,len(groups)))
        return groups['sum'].to_numpy()[idx].sum(1)/groups['count'].to_numpy()[idx].sum(1)
    bounds=lambda x:[float(v) for v in np.quantile(x,[.025,.975])]
    c=bounds(curve);w=bounds(clustered(wl));p=bounds(clustered(pec))
    return dict(paired_finite_n=len(delta),mean_delta=float(delta.mean()),curve_low=c[0],curve_high=c[1],
        WL_low=w[0],WL_high=w[1],PEC_block_low=p[0],PEC_block_high=p[1],bootstrap_repeats=reps,
        scope='within archived cohort; PEC and block confounded; no independent chip-population CI')


def read_parts(folder):
    files=sorted(Path(folder).glob('*.parquet'))
    if not files:raise RuntimeError(f'Missing metric parts: {folder}')
    return pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)


def check_coverage(frame,expected,seeds,tag):
    groups=['method','protocol','budget','seed']
    records=[]
    for ident,q in frame.groupby(groups,dropna=False,sort=False):
        actual=q['task_id'].to_numpy();assert len(actual)==len(expected),f'{tag} {ident}: count {len(actual)} != {len(expected)}'
        assert len(np.unique(actual))==len(expected),f'{tag} {ident}: duplicate task IDs'
        assert np.array_equal(np.sort(actual),np.sort(expected)),f'{tag} {ident}: full-cohort mismatch'
        records.append(dict(tag=tag,method=ident[0],protocol=ident[1],budget=int(ident[2]),seed=int(ident[3]),curves=len(actual)))
    found=set(frame['seed'].unique());assert found==set(seeds),f'{tag}: seed coverage {found} != {seeds}'
    patterns=set(zip(frame['method'],frame['protocol'],frame['budget']))
    identities=set(zip(frame['method'],frame['protocol'],frame['budget'],frame['seed']))
    assert identities=={(*p,s) for p in patterns for s in seeds},f'{tag}: missing a method/protocol/seed cell'
    return records


def summarize(frame,key,reference=None,metric=METRICS[3]):
    result=[]
    for (method,protocol,B),q in frame.groupby(['method','protocol','budget'],sort=False):
        row=dict(case=key,method=method,protocol=protocol,budget=int(B),rows=len(q),curves=q['task_id'].nunique(),seeds=q['seed'].nunique())
        for name in METRICS+['extended_tail_rmse_pct_peak','exterior_zero_rmse_pct_peak','positive_tail_rmse_pct_peak']:
            if name not in q:continue
            # Undefined or nonfinite metrics retain their denominators explicitly.
            a=q[name].to_numpy(float);finite=a[np.isfinite(a)]
            row[name+'_mean']=float(finite.mean()) if len(finite) else None
            row[name+'_median']=float(np.median(finite)) if len(finite) else None
            row[name+'_undefined_rows']=int((~np.isfinite(a)).sum())
        if 'test_r2' in q:row['negative_r2_fraction']=float((q['test_r2']<0).mean())
        if 'nonfinite' in q:row['optimizer_nonfinite_fraction']=float(q['nonfinite'].fillna(False).mean())
        if 'fallback' in q:row['classical_fallback_fraction']=float(q['fallback'].fillna(False).mean())
        if 'feasible' in q:row['infeasible_rows']=int((~q['feasible']).sum())
        if 'sse' in q:
            n=float(q['test_n'].sum());sst=float(q['sum_y2'].sum()-q['sum_y'].sum()**2/n)
            row['pooled_point_r2']=1-float(q['sse'].sum())/sst if sst>0 else None
        if metric in q:
            seedmeans=q.groupby('seed')[metric].mean()
            row['seed_mean_std_ddof1']=float(seedmeans.std(ddof=1)) if len(seedmeans)>1 else None
        if reference is not None and metric in q:
            base=reference[(reference['protocol']==protocol)&(reference['budget']==B)]
            if len(base):
                # Average the three repeated fits per curve before sampling curves/WLs.
                a=q.groupby('task_id').agg(value=(metric,'mean'),WL=('WL','first'),PEC=('PEC','first'))
                b=base.groupby('task_id')[metric].mean().rename('reference');paired=a.join(b,how='inner')
                assert len(paired)==q['task_id'].nunique()
                row.update(bootstrap(paired['value']-paired['reference'],paired['WL'],paired['PEC'],20261014))
        result.append(row)
    return result


def variance_analysis(d,key,out):
    """Finite-seed squared-error identity, not physical/irreducible noise bias."""
    records=[];target=d['target'];cfg=config()
    main_methods=['diffusion','scratch','pooled_source','nearest_source','conditional_ridge']
    for method in main_methods+['cvae','flow','dropout','ensemble','conditional_mean']:
      for protocol,B in ([('nested',b) for b in cfg['budgets']]+[('default',0)] if method in main_methods else [('nested',7)]):
        for start in range(0,len(target),cfg['gpu_batch']):
            parts=[HERE/'results'/key/f'seed{s}/predictions/{method}/{protocol}_B{B}_{start:07}.npz' for s in cfg['seeds']]
            if not all(p.exists() for p in parts):raise RuntimeError(f'Missing prediction repeats: {parts}')
            ids=target[start:start+cfg['gpu_batch']];pred=[]
            for p in parts:
                with np.load(p) as z:
                    assert np.array_equal(z['task_id'],d['task_id'][ids]);pred.append(z['pred'].astype(np.float64))
            arr=np.stack(pred);mean=arr.mean(0);var=arr.var(0,ddof=0);truth=d['frequency'][ids].astype(np.float64)
            _,tests,_=selections(d,ids,protocol,B)
            rows=[]
            for j,(i,t) in enumerate(zip(ids,tests)):
                n=d['length'][i];k=max(2,int(np.ceil(.2*n)));tail=t[(t<k)|(t>=n-k)]
                peak=max(float(truth[j].max()),1e-9)
                for scope,a in [('test',t),('tail_test',tail)]:
                    mean_error=float(((mean[j,a]-truth[j,a])**2).mean())/peak**2
                    variance=float(var[j,a].mean())/peak**2
                    mse=float(((arr[:,j,a]-truth[j,a])**2).mean())/peak**2
                    assert np.isclose(mean_error+variance,mse,rtol=1e-7,atol=1e-12)
                    rows.append(dict(case=key,method=method,protocol=protocol,budget=B,task_id=int(d['task_id'][i]),scope=scope,
                        mean_prediction_squared_error=mean_error,finite_seed_prediction_variance=variance,mean_seed_mse=mse,seeds=len(parts)))
            parquet_write(out/f'variance/{method}_{protocol}_B{B}_{start:07}.parquet',pd.DataFrame(rows))
            records.extend(pd.DataFrame(rows).groupby('scope')[['mean_prediction_squared_error','finite_seed_prediction_variance','mean_seed_mse']].mean().reset_index().assign(case=key,method=method,protocol=protocol,budget=B,curves=len(ids)).to_dict('records'))
    return records


def failure_analysis(frame,key,out):
    from sklearn.cluster import KMeans
    q=frame[(frame['method']=='diffusion')&(frame['protocol']=='nested')&(frame['budget']==7)]
    q=q.groupby('task_id').agg(test_r2=('test_r2','mean'),tail=('tail_test_rmse_pct_peak','mean'),
        peak=('peak_error_steps','mean'),WL=('WL','first'),PEC=('PEC','first'),label=('label','first'))
    q['failure']=q['test_r2']<0;q['undefined_r2']=~np.isfinite(q['test_r2'])
    q['category']=np.select([q['undefined_r2'],q['failure']&(q['peak']>2),q['failure']],['undefined_R2','negative_R2_peak_displaced','negative_R2_other'],default='nonnegative_R2')
    q['cluster']=-1;failed=q[q['failure']];cols=['test_r2','tail','peak']
    finite=np.isfinite(failed[cols]).all(1);v=failed.loc[finite,cols].to_numpy()
    if len(v)>=3:
        v=(v-np.median(v,0))/np.maximum(np.percentile(v,75,axis=0)-np.percentile(v,25,axis=0),1e-9)
        labels=KMeans(n_clusters=min(3,len(np.unique(v,axis=0))),random_state=42,n_init=10).fit_predict(v)
        q.loc[failed.index[finite],'cluster']=labels
    parquet_write(out/'failures.parquet',q.reset_index().assign(case=key))
    return q['category'].value_counts().to_dict()


def audit_case(record):
    key=f'{record["chip"]}_{record["experiment"]}_P{record["state"]}';d=load_data(key);expected=d['task_id'][d['target']]
    out=HERE/'analysis'/key;out.mkdir(parents=True,exist_ok=True)
    marker=out/'complete.json'
    if marker.exists():return read_json(marker)
    cfg=config();coverage=[];summaries=[];references=read_parts(HERE/'results'/key/'classical')
    for seed in cfg['seeds']:
        folder=HERE/'results'/key/f'seed{seed}'
        g=read_json(folder/'generated/main.json')
        assert g['rows']==len(expected) and g['source_vectors']==len(d['src']) and g['min_source_exposure']>=2
        assert hash_file(folder/'generated/main.npy')==g['checksum']
        for number in [50,150,len(d['src'])//2]:
            r=read_json(folder/f'generated/source{number}_matched_updates.json')
            assert r['rows']==len(expected) and r['source_vectors']==number
            assert r['updates']==cfg['generator_epochs']*int(np.ceil(len(d['src'])/cfg['generator_batch']))
    coverage.extend(check_coverage(references,expected,[0],key+' classical'))
    assert set(references['method'])=={'gaussian','log_gaussian','skew_gaussian','log_skew_gaussian','spline','pchip'}
    assert set(zip(references['method'],references['protocol'],references['budget']))=={(m,p,b) for m in references['method'].unique() for p,b in [('nested',5),('nested',7),('nested',9),('default',0)]}
    ref=references[references['method']=='gaussian'];summaries.extend(summarize(references,key,ref))
    methods=['pooled_source','conditional_ridge','nearest_source','scratch','diffusion','selected_policy',
        'lambda_1e-06','lambda_0.0001','lambda_0.01','T50_native','T50_matched','noWL','noStress','noPEC','unconditional','cvae','flow','dropout',
        'source50','source150',f'source{len(d["src"])//2}','shuffle_WL','shuffle_Stress','shuffle_PEC',
        'diffusion_adam250','nearest_source_adam250','pooled_source_adam250','ensemble',
        'direct_random_forest','direct_knn','direct_xgboost','direct_tradaboost_original','direct_tradaboost_improved',
        'function_rbf_conditional','function_rbf_zero_prior','uniform_sparsity','direct_function_mlp',
        'source50_matched_updates','source150_matched_updates',f'source{len(d["src"])//2}_matched_updates','conditional_mean']
    failures=None
    for method in methods:
        frame=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/metrics/{method}') for s in cfg['seeds']],ignore_index=True)
        coverage.extend(check_coverage(frame,expected,cfg['seeds'],key+' '+method))
        groups=set(zip(frame['protocol'],frame['budget']))
        required={('nested',5),('nested',7),('nested',9),('default',0)} if method in methods[:5] or method.startswith('function_rbf') else {('default',0)} if method=='selected_policy' else {('uniform_full_range',B) for B in [3,5,7,9,11,13]} if method=='uniform_sparsity' else {('nested',7)}
        assert groups==required,f'{key} {method}: incomplete protocol coverage {groups}'
        summaries.extend(summarize(frame,key,ref if method!='uniform_sparsity' else None))
        if method=='diffusion':failures=failure_analysis(frame,key,out)
    uniform=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/uniform_classical') for s in cfg['seeds']],ignore_index=True)
    coverage.extend(check_coverage(uniform,expected,cfg['seeds'],key+' uniform classical'))
    assert set(zip(uniform['method'],uniform['budget']))=={(m,b) for m in references['method'].unique() for b in [3,5,7,9,11,13]}
    uref=uniform[uniform['method']=='gaussian'];summaries.extend(summarize(uniform,key,uref))
    uf=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/metrics/uniform_sparsity') for s in cfg['seeds']],ignore_index=True)
    summaries.extend(summarize(uf,key,uref))
    zclass=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/zero_extension/classical') for s in cfg['seeds']],ignore_index=True)
    assert set(zclass['method'])==set(references['method'])
    coverage.extend(check_coverage(zclass,expected,cfg['seeds'],key+' zero classical'));zref=zclass[zclass['method']=='gaussian']
    zs=summarize(zclass,key,zref,metric='extended_tail_rmse_pct_peak')
    for method in methods[:5]:
        z=pd.concat([read_parts(HERE/'results'/key/f'seed{s}/zero_extension/{method}') for s in cfg['seeds']],ignore_index=True)
        coverage.extend(check_coverage(z,expected,cfg['seeds'],key+' zero '+method));zs.extend(summarize(z,key,zref,metric='extended_tail_rmse_pct_peak'))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(zs).to_csv(out/'zero_extension_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False,encoding='utf-8-sig')
    vr=variance_analysis(d,key,out);pd.DataFrame(vr).to_csv(out/'variance_summary_parts.csv',index=False,encoding='utf-8-sig')
    result=dict(status='complete',case=key,target_curves=len(expected),source_curves=len(d['src']),failure_categories=failures,
        coverage_groups=len(coverage),no_curve_subsample=True,ci_scope='curve, WL and PEC-block resampling within cohort',
        variance_scope='empirical decomposition across three training seeds, not true bias/noise separation')
    atomic_json(marker,result);return result


def analysis():
    assert read_json(HERE/'sensitivity_validation.json',{}).get('status')=='passed'
    assert read_json(HERE/'runtime_validation.json',{}).get('status')=='passed'
    with sqlite3.connect(HERE/'tasks.sqlite') as db:
        unfinished=db.execute('select id,status from jobs where kind not in ("analysis","delivery") and status != "complete"').fetchall()
    assert not unfinished,f'Numerical tasks unfinished: {unfinished[:10]}'
    manifest=read_json(HERE/'data/manifest.json');out=HERE/'analysis';out.mkdir(exist_ok=True);records=[]
    for record in manifest['records']:
        records.append(audit_case(record));atomic_json(out/'progress.json',dict(completed_cohorts=len(records),total=28,updated=time.time()))
    pd.concat([pd.read_csv(out/r['case']/'summary.csv') for r in records],ignore_index=True).to_csv(out/'all_cohort_summaries.csv',index=False,encoding='utf-8-sig')
    pd.concat([pd.read_csv(out/r['case']/'zero_extension_summary.csv') for r in records],ignore_index=True).to_csv(out/'all_zero_extension_summaries.csv',index=False,encoding='utf-8-sig')
    costs=[]
    for case in records:
      for seed in config()['seeds']:
        folder=HERE/'results'/case['case']/f'seed{seed}'
        for file in (folder/'generated').glob('*.json'):
            r=read_json(file);r.update(case=case['case'],training_seed=seed,file=str(file.relative_to(HERE)));costs.append(r)
    pd.DataFrame(costs).to_csv(out/'generation_costs.csv',index=False,encoding='utf-8-sig')
    atomic_json(out/'audit.json',dict(cohorts=records,source_curves=sum(r['source_curves'] for r in records),
        target_curves=sum(r['target_curves'] for r in records),all_full_scope_verified=True,
        unresolved_noncomputational_records=['CT voltage calibration','unique chip/Die identifiers','independent block/PEC separation',
            'historical lambda selection and unrecorded seeds','algorithm BER/ECC validation','hardware acquisition latency and energy'],
        limitations=['two full source passes do not establish convergence','batch L-BFGS shares line search and curvature history',
            'second chip is a separately trained replication, not zero-shot transfer','known positive support is archival information',
            'direct transfer regressors use all target anchors jointly; per-curve MLP adaptation uses only that curve anchors',
            'zero-extension combined RMSE can be diluted by zero-bin count; both components are reported']))
    atomic_json(HERE/'analysis_complete.json',dict(status='complete',verified_cohorts=28,source_curves=manifest['source_curves'],
        target_curves=manifest['target_curves'],delivery_complete=False,updated=time.time()))


def delivery():
    assert (HERE/'analysis_complete.json').exists()
    required=['manuscript_pdf','chinese_reviewer_reply_pdf','reproduction_inventory']
    # Waiting for the chat agent to finish the manuscript and visual PDF verification
    # is an explicit final stage, never interpreted as numerical task completion.
    while True:
        release=read_json(HERE/'delivery_release.json',{})
        if release.get('agent_reviewed') and release.get('pdf_visual_qa_passed') and all(k in release for k in required):break
        atomic_json(HERE/'delivery_progress.json',dict(status='awaiting_document_completion_and_visual_review',
            numerical_audit_complete=True,not_finished=True,updated=time.time()))
        time.sleep(30)
    for key in required:
        item=release[key];path=Path(item['path']).resolve();path.relative_to(ROOT.resolve())
        assert path.exists() and hash_file(path)==item['sha256'],f'Unverified final artifact {key}'
    atomic_json(HERE/'delivery_complete.json',dict(status='complete',artifacts=release,updated=time.time()))


if __name__=='__main__':
    from queue_scope import retained_scope
    if retained_scope():
        from finalize_retained_run import analysis as retained_analysis
        if sys.argv[1]=='analysis':retained_analysis()
        elif sys.argv[1]=='delivery':
            assert read_json(HERE/'analysis_complete.json',{}).get('scope')=='retained_reviews_20261014'
            delivery()
        else:raise ValueError(sys.argv[1])
        sys.exit(0)
    if sys.argv[1]=='analysis':analysis()
    elif sys.argv[1]=='delivery':delivery()
    else:raise ValueError(sys.argv[1])
