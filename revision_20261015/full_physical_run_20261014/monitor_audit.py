"""Read-only coverage validation for completed main/classical/CPU supplement jobs."""
import sqlite3,time
import numpy as np
from full_common import HERE,load_data,atomic_json,hash_file,read_json
from finalize_full_run import read_parts,check_coverage
from queue_scope import retained_scope,scope_description


def main():
    with sqlite3.connect(HERE/'tasks.sqlite') as db:
        jobs=db.execute('select id,kind,case_key,seed from jobs where status="complete" and kind in ("main","classical","extra_cpu","regularization")').fetchall()
    previous={r['job']:r for r in read_json(HERE/'monitor_coverage_audit.json',{}).get('audited_completed_jobs',[])}
    records=[]
    for job,kind,key,seed in jobs:
        done=HERE/'results'/key/('classical/complete.json' if kind=='classical' else f'seed{seed}/{kind}_complete.json')
        done_hash=hash_file(done)
        prior=previous.get(job)
        if prior and prior.get('completion_marker_sha256')==done_hash:
            records.append(prior);continue
        d=load_data(key);expected=d['task_id'][d['target']];root=HERE/'results'/key;groups=[]
        if kind=='classical':
            f=read_parts(root/'classical');groups=check_coverage(f,expected,[0],job)
            assert len(groups)==24
        elif kind=='main':
            for method in ['diffusion','scratch','pooled_source','nearest_source','conditional_ridge','selected_policy']:
                f=read_parts(root/f'seed{seed}/metrics/{method}');r=check_coverage(f,expected,[seed],job+' '+method)
                assert len(r)==(1 if method=='selected_policy' else 4);groups.extend(r)
            params=np.load(root/f'seed{seed}/source_params.npy',mmap_mode='r');assert params.shape==(len(d['src']),609)
            assert np.isfinite(params).all()
            generated=read_json(root/f'seed{seed}/generated/main.json');assert generated['min_source_exposure']>=2
            assert generated['source_vectors']==len(d['src']) and generated['rows']==len(expected)
        elif kind=='regularization':
            folder=root/f'seed{seed}'
            for strength in read_json(HERE/'config.json')['lambda_values']:
                method='diffusion' if strength==0 else f'lambda_{strength:g}'
                f=read_parts(folder/f'metrics/{method}');f=f[(f.protocol=='nested')&(f.budget==7)]
                r=check_coverage(f,expected,[seed],job+' '+method)
                assert len(r)==1;groups.extend(r)
            mark=read_json(done);assert mark['full_targets']==len(expected) and mark['all_target_curves']
        else:
            folder=root/f'seed{seed}'
            for method in ['direct_random_forest','direct_knn','direct_xgboost','direct_tradaboost_original','direct_tradaboost_improved','function_rbf_conditional','function_rbf_zero_prior']:
                f=read_parts(folder/f'metrics/{method}');r=check_coverage(f,expected,[seed],job+' '+method)
                assert len(r)==(4 if method.startswith('function_rbf') else 1);groups.extend(r)
            for tag,number in [('uniform_classical',36),('zero_extension/classical',6)]:
                r=check_coverage(read_parts(folder/tag),expected,[seed],job+' '+tag)
                assert len(r)==number;groups.extend(r)
            shap_ids=[]
            for file in sorted((folder/'shap_full').glob('*.npz')):
                with np.load(file) as z:
                    shap_ids.extend(z['task_id'].tolist())
                    assert z['shap_values'].shape==z['features'].shape
                    assert z['features'].shape[1]==4
                    assert z['shap_values'].shape[0]==int(z['test_counts'].sum())
            assert len(shap_ids)==len(expected) and np.array_equal(np.sort(shap_ids),np.sort(expected)),job+' SHAP coverage mismatch'
            marker=read_json(done)
            assert marker['full_targets']==len(expected) and marker['zero_count_tail_extension'] and marker['uniform_sparsity_classical']
        records.append(dict(job=job,status='coverage_verified',target_curves=len(expected),metric_groups=len(groups),completion_marker_sha256=done_hash))
        print(job,'verified',len(expected),flush=True)
    atomic_json(HERE/'monitor_coverage_audit.json',dict(updated=time.time(),audited_completed_jobs=records,
        scope='completed jobs only; remaining jobs are not called completed',all_experiments_complete=False))
    manifest=read_json(HERE/'run_manifest.json');manifest.update(status='registered_running',
        supplemental_scope=['all exterior zero-count bins scored separately on full target cohorts',
            'uniform budgets3/5/7/9/11/13 for diffusion and all six classical methods',
            'source50/150/half with matched full-source generator update counts',
            'conditional mean, CVAE, flow, ensemble and dropout prediction variance on every target curve'],
        final_stage_gate='all numerical cohorts audited plus manuscript and Chinese reply PDFs visually checked',
        code_snapshot={p.name:hash_file(p) for p in HERE.glob('*.py')},last_supervision_audit=time.time())
    if retained_scope():
        manifest.update(scope_profile=scope_description()['profile'],supplemental_scope=scope_description()['required'],
            cancelled_scope=scope_description()['cancelled'],main_and_all_ablations_full_targets=False,
            full_main_and_lambda_targets=True,scope_confirmation=scope_description()['authorization'])
    atomic_json(HERE/'run_manifest.json',manifest)


if __name__=='__main__':main()
