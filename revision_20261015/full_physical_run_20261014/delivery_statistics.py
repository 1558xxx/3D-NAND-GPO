"""Produce publication summaries from every audited retained target curve.

No fitting, target tuning, experimental resampling, or changes to score records.
Bootstrap resampling is used only for explicitly labelled uncertainty intervals.
"""
from pathlib import Path
import gc,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from full_common import HERE,read_json,atomic_json,load_data,selections
from finalize_retained_run import PROFILE,bootstrap,METRICS

OUT=HERE/'delivery_final';DATA=OUT/'tables';FIG=OUT/'source/figures'
METHODS=['gaussian','log_gaussian','skew_gaussian','log_skew_gaussian','spline','pchip',
    'scratch','pooled_source','nearest_source','conditional_ridge','diffusion']
FAMILIES=[('TH58TFT1T23BA8H','Bake'),('TH58TFT1T23BA8H','Read'),('MT29F1T08EEHAF','Bake'),('MT29F1T08EEHAF','Read')]
NAMES={'gaussian':'Gaussian','log_gaussian':'Log-space Gaussian','skew_gaussian':'Skew Gaussian',
    'log_skew_gaussian':'Log-space skew Gaussian','spline':'Natural spline','pchip':'PCHIP','scratch':'Scratch MLP',
    'pooled_source':'Pooled-source MLP','nearest_source':'Nearest-source MLP','conditional_ridge':'Conditional parameter ridge',
    'diffusion':'Diffusion initialization','selected_policy':'Validation-selected policy',
    'lambda_1e-06':'lambda=1e-6','lambda_0.0001':'lambda=1e-4','lambda_0.01':'lambda=1e-2'}


def get_frame(chip,group,method,protocol,B):
    frames=[];classical=method in METHODS[:6]
    for state in range(1,8):
        key=f'{chip}_{group}_P{state}';base=HERE/'results'/key
        for seed in ([0] if classical else [42,43,44]):
            folder=base/'classical' if classical else base/f'seed{seed}/metrics/{method}'
            files=sorted(folder.glob(f'{protocol}_B{B}_*.parquet'));assert files,(key,method,protocol,B)
            for file in files:
                schema=pq.ParquetFile(file).schema.names
                wanted=['task_id','WL','PEC','label','physical','seed','method','protocol','budget','test_n','tail_test_n','sse','sum_y','sum_y2']+METRICS+['penalty_loss','data_loss','nonfinite','fallback']
                q=pd.read_parquet(file,columns=[c for c in wanted if c in schema])
                if classical:q=q[q.method==method].copy()
                q['state']=state;q['case']=key;frames.append(q)
    return pd.concat(frames,ignore_index=True)


def summarize(q,chip,group,method,protocol,B):
    r=dict(chip=chip,group=group,method=method,protocol=protocol,budget=B,
        unique_curves=len(q[['state','task_id']].drop_duplicates()),curve_seed_records=len(q),training_seeds=q.seed.nunique())
    for metric in METRICS:
        a=q[metric].to_numpy(float);a=a[np.isfinite(a)]
        r[metric+'_mean']=float(a.mean()) if len(a) else None
        r[metric+'_undefined']=int(len(q)-len(a))
        means=q.groupby('seed')[metric].mean()
        r[metric+'_seed_std']=float(means.std(ddof=1)) if len(means)>1 else None
        for quant in [.05,.25,.5,.75,.95,.99]:r[f'{metric}_p{int(quant*100):02}']=float(np.quantile(a,quant)) if len(a) else None
    r['negative_r2_fraction']=float((q.test_r2<0).mean())
    r['tail_test_n_min']=int(q.tail_test_n.min());r['tail_test_n_max']=int(q.tail_test_n.max())
    n=float(q.test_n.sum());sst=float(q.sum_y2.sum()-q.sum_y.sum()**2/n)
    r['point_pooled_r2']=1-float(q.sse.sum())/sst if sst>0 else None
    if 'nonfinite' in q:r['optimizer_nonfinite_fraction']=float(q.nonfinite.fillna(False).mean())
    if 'fallback' in q:r['fit_fallback_fraction']=float(q.fallback.fillna(False).mean())
    if 'penalty_loss' in q:
        a=q.penalty_loss.to_numpy(float);b=q.data_loss.to_numpy(float);ok=np.isfinite(a)&np.isfinite(b)&(b>0)
        ratio=a[ok]/b[ok];r['penalty_data_ratio_defined']=int(ok.sum());r['penalty_data_ratio_undefined']=int((~ok).sum())
        r['penalty_data_ratio_median']=float(np.median(ratio)) if len(ratio) else None
        r['penalty_data_ratio_p95']=float(np.quantile(ratio,.95)) if len(ratio) else None
    return r


def curve_means(q):
    return q.groupby(['state','task_id']).agg(tail=('tail_test_rmse_pct_peak','mean'),WL=('WL','first'),PEC=('PEC','first'))


def plot_figures(summary,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'diffusion':'#255bb0','gaussian':'#c07a18','log_gaussian':'#2b856e','pooled_source':'#8e57a7'}
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,(chip,group) in zip(axes.flat,FAMILIES):
        for method in ['gaussian','log_gaussian','pooled_source','diffusion']:
            q=summary[(summary.chip==chip)&(summary.group==group)&(summary.method==method)&(summary.protocol=='nested')].sort_values('budget')
            y=q.tail_test_rmse_pct_peak_mean.to_numpy();sd=q.tail_test_rmse_pct_peak_seed_std.fillna(0).to_numpy()
            ax.errorbar(q.budget,y,yerr=sd,marker='o',capsize=3,label=NAMES[method],color=colors[method],lw=1.5)
        ax.set_title(('CT' if chip.startswith('TH') else 'FG')+' / '+group);ax.set_xticks([5,7,9]);ax.set_xlabel('Observed coordinates B')
        ax.set_ylabel('Test-tail RMSE (% of peak)');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(FIG/'full_budget.pdf',bbox_inches='tight');fig.savefig(OUT/'qa/full_budget.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,(chip,group) in zip(axes.flat,FAMILIES):
        q=pairs[(pairs.case.str.startswith(chip+'_'+group+'_'))&(pairs.reference=='gaussian')&(pairs.protocol=='nested')&(pairs.budget==7)&(pairs.metric=='tail_test_rmse_pct_peak')].copy()
        q['state']=q.case.str.extract(r'_P(\d+)$').astype(int);q=q.sort_values('state')
        for _,r in q.iterrows():ax.plot([r.WL_low,r.WL_high],[r.state,r.state],color='#426e8c',lw=2);ax.plot(r.mean_delta,r.state,'o',color='#255bb0')
        ax.axvline(0,color='#555',ls='--',lw=1);ax.set_yticks(range(1,8),[f'P{i}' for i in range(1,8)])
        ax.set_title(('CT' if chip.startswith('TH') else 'FG')+' / '+group);ax.set_xlabel('Diffusion - Gaussian (percentage points)');ax.grid(axis='x',alpha=.2)
    fig.savefig(FIG/'full_state_intervals.pdf',bbox_inches='tight');fig.savefig(OUT/'qa/full_state_intervals.png',dpi=160,bbox_inches='tight');plt.close(fig)


def failure_examples():
    import matplotlib.pyplot as plt
    chosen=[];fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for ax,(chip,group) in zip(axes.flat,FAMILIES):
        candidates=[]
        for state in range(1,8):
            key=f'{chip}_{group}_P{state}'
            q=pd.read_parquet(HERE/'analysis_retained'/key/'failure_records.parquet',columns=['task_id','seed','protocol','budget','test_r2','test_nrmse_pct_peak'])
            q=q[(q.protocol=='nested')&(q.budget==7)&np.isfinite(q.test_r2)]
            if len(q):r=q.loc[q.test_r2.idxmin()].to_dict();r['case']=key;candidates.append(r)
        r=min(candidates,key=lambda a:a['test_r2']);key=r['case'];seed=int(r['seed']);task=int(r['task_id']);d=load_data(key)
        pos=int(np.flatnonzero(d['task_id'][d['target']]==task)[0]);i=int(d['target'][pos]);start=pos//4096*4096
        with np.load(HERE/'results'/key/f'seed{seed}/predictions/diffusion/nested_B7_{start:07}.npz') as z:
            assert int(z['task_id'][pos-start])==task;pred=z['pred'][pos-start]
        n=int(d['length'][i]);x=d['steps'][i,:n];y=d['frequency'][i,:n];a,t,_=selections(d,[i],'nested',7);a=a[0];t=t[0]
        ax.plot(x,y+1,'-',color='#6d747c',label='Reference');ax.plot(x,pred[:n]+1,'-',color='#255bb0',label='Diffusion')
        ax.scatter(x[a],y[a]+1,s=22,color='#d68127',label='Observed');ax.scatter(x[t],y[t]+1,s=22,facecolors='none',edgecolors='#333',marker='s',label='Test')
        ax.set_yscale('log');ax.set_xlabel('Worksheet scan index');ax.set_ylabel('Frequency + 1 (log scale)');ax.grid(alpha=.18)
        ax.set_title(('CT' if chip.startswith('TH') else 'FG')+f' / {group} / P{key[-1]}\n'+f'WL {int(d["WL"][i])}, PEC {int(d["PEC"][i])}, seed {seed}')
        ax.text(.02,.04,f'Test R²={r["test_r2"]:.3g}; NRMSE={r["test_nrmse_pct_peak"]:.3g}%',transform=ax.transAxes,fontsize=8)
        rows=pd.DataFrame(dict(step=x,truth=y,prediction=pred[:n],observed=np.isin(np.arange(n),a),test=np.isin(np.arange(n),t)))
        rows.to_csv(DATA/f'failure_example_{chip}_{group}.csv',index=False);chosen.append(r)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(FIG/'full_failure_examples.pdf',bbox_inches='tight');fig.savefig(OUT/'qa/full_failure_examples.png',dpi=160,bbox_inches='tight');plt.close(fig)
    atomic_json(DATA/'failure_examples_selection.json',dict(rule='minimum finite B7 test R2 over all states, target curves and three seeds within each family; extremes, not typical curves',examples=chosen))


def main():
    assert read_json(HERE/'analysis_complete.json',{}).get('scope')==PROFILE,'Full retained analysis must finish first'
    for p in [DATA,FIG,OUT/'qa']:p.mkdir(parents=True,exist_ok=True)
    allrows=[];pairrows=[]
    for chip,group in FAMILIES:
        comparators={}
        for method in METHODS+['lambda_1e-06','lambda_0.0001','lambda_0.01']:
            budgets=[7] if method.startswith('lambda_') or method not in ['gaussian','log_gaussian','pooled_source','diffusion'] else [5,7,9]
            for B in budgets:
                q=get_frame(chip,group,method,'nested',B);allrows.append(summarize(q,chip,group,method,'nested',B))
                if B==7 and method in ['gaussian','log_gaussian','pooled_source','diffusion']:comparators[method]=curve_means(q)
                del q;gc.collect()
        for method in ['gaussian','log_gaussian','pooled_source','scratch','diffusion','selected_policy']:
            q=get_frame(chip,group,method,'default',0);allrows.append(summarize(q,chip,group,method,'default',0));del q;gc.collect()
        a=comparators['diffusion']
        for method in ['gaussian','log_gaussian','pooled_source']:
            b=comparators[method];assert a.index.equals(b.index)
            r=dict(chip=chip,group=group,reference=method,budget=7,protocol='nested',curves=len(a),paired_win_fraction=float((a['tail']<b['tail']).mean()))
            r.update(bootstrap(a['tail']-b['tail'],a.WL,a.PEC,20261015));pairrows.append(r)
        print('Publication family summarized',chip,group,flush=True)
    summary=pd.DataFrame(allrows);summary.to_csv(DATA/'family_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairrows).to_csv(DATA/'family_paired_tail_intervals.csv',index=False,encoding='utf-8-sig')
    for name in ['all_summary','all_seed_summary','all_paired_comparisons','all_coverage','all_failure_categories']:
        frame=pd.read_csv(HERE/'analysis_retained'/f'{name}.csv');frame.to_csv(DATA/f'{name}.csv',index=False,encoding='utf-8-sig')
    plot_figures(summary,pd.read_csv(DATA/'all_paired_comparisons.csv'));failure_examples()
    costs=[]
    import torch
    for path in sorted((HERE/'results').glob('*/seed*/generator_main.pt')):
        r=torch.load(path,map_location='cpu',weights_only=False)
        costs.append(dict(case=path.parent.parent.name,seed=r['seed'],source_vectors=r['source_vectors'],epochs=r['epochs'],
            updates=r['optimizer_updates'],training_seconds=r['training_seconds'],parameters=r['parameters']))
    assert len(costs)==84;pd.DataFrame(costs).to_csv(DATA/'generator_training_costs.csv',index=False,encoding='utf-8-sig')
    atomic_json(OUT/'publication_statistics_complete.json',dict(scope=PROFILE,unique_targets=384780,seeds=[42,43,44],
        family_count=4,generator_checkpoints=len(costs),family_CI='paired curve, WL and PEC-block bootstrap after averaging seeds per curve',
        quantile_scope='all curve-seed scores; deterministic classical scores entered once',sampling='no target curves removed'))


if __name__=='__main__':main()
