"""Incremental full-cohort CPU analysis of completed retained main jobs."""
import os,time,sqlite3,gc,traceback
import pandas as pd
import psutil
from full_common import HERE,load_data,atomic_json,read_json,hash_file
from finalize_retained_run import MAIN,PROTOCOLS,read_parts,validate,describe,failure_records
from queue_scope import PROFILE,retained_scope

def main():
    lock=(HERE/'cpu_retained_preanalysis.lock').open('a+b')
    import msvcrt
    lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    root=HERE/'analysis_cpu_progress';root.mkdir(exist_ok=True)
    atomic_json(root/'process.json',dict(pid=os.getpid(),scope=PROFILE,started=time.time()))
    while retained_scope():
        with sqlite3.connect(HERE/'tasks.sqlite') as db:
            jobs=db.execute("select case_key,seed from jobs where kind='main' and status='complete' order by rowid").fetchall()
        completed=0;worked=False
        for key,seed in jobs:
            out=root/key/f'seed{seed}';marker=out/'complete.json'
            source=HERE/'results'/key/f'seed{seed}/main_complete.json'
            fingerprint=hash_file(source)
            prior=read_json(marker,{})
            if prior.get('source_marker_sha256')==fingerprint:
                completed+=1;continue
            if psutil.virtual_memory().available<1.5e9:break
            atomic_json(root/'status.json',dict(status='running',case=key,seed=seed,completed=completed,available_completed_main=len(jobs),total=84,updated=time.time()))
            out.mkdir(parents=True,exist_ok=True)
            d=load_data(key);expected=d['task_id'][d['target']];rows=[];seedrows=[];coverage=[]
            for method in MAIN:
                f=read_parts(HERE/'results'/key/f'seed{seed}/metrics/{method}')
                protocols={('default',0)} if method=='selected_policy' else PROTOCOLS
                coverage+=validate(f,expected,[seed],method,protocols,key)
                r,s=describe(f,key);rows+=r;seedrows+=s
                if method=='diffusion':failure_records(f,key,out,d)
                del f
            for name,data in [('summary',rows),('seed_summary',seedrows),('coverage',coverage)]:
                pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
            atomic_json(marker,dict(scope=PROFILE,source_marker_sha256=fingerprint,target_curves=len(expected),seed=seed,full_cohort=True,final_analysis_complete=False,updated=time.time()))
            print('Completed CPU analysis',key,seed,len(expected),flush=True)
            del d,rows,seedrows,coverage;gc.collect();completed+=1;worked=True
        atomic_json(root/'status.json',dict(status='complete' if completed==84 else 'waiting_for_main_or_memory',completed=completed,total=84,available_completed_main=len(jobs),updated=time.time(),final_analysis_complete=False))
        if completed==84:break
        if not worked:time.sleep(30)

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(HERE/'analysis_cpu_progress/error.json',dict(error=traceback.format_exc(),updated=time.time()))
        raise
