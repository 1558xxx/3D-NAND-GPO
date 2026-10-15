"""Durable scheduler. The workload is complete only after numerical, audit and delivery stages."""
from pathlib import Path
import os,sys,json,time,sqlite3,subprocess,ctypes
import psutil
from full_common import HERE,ROOT,config,read_json,atomic_json
from queue_scope import retained_scope,required_job,scope_description
DB=HERE/'tasks.sqlite';LOGS=HERE/'logs';LOGS.mkdir(exist_ok=True)

def init_db():
    c=sqlite3.connect(DB);c.execute('pragma journal_mode=WAL')
    c.execute('create table if not exists jobs(id text primary key,kind text,case_key text,seed integer,status text,pid integer,attempt integer,started real,finished real,error text)')
    data=read_json(HERE/'data/manifest.json');cfg=config()
    if data is None or data.get('status')!='complete':raise RuntimeError('Raw dataset preparation not complete')
    records=data['records'];keys=[f'{r["chip"]}_{r["experiment"]}_P{r["state"]}' for r in records]
    order=[2,4,7,1,3,5,6]
    keys.sort(key=lambda x:(order.index(int(x.rsplit('P',1)[1])),0 if x.startswith('TH') else 1,0 if '_Bake_' in x else 1))
    jobs=[]
    for key in keys:jobs.append((f'classical:{key}','classical',key,0))
    for seed in cfg['seeds']:
      for key in keys:jobs.append((f'main:{key}:{seed}','main',key,seed))
    if retained_scope():
      for seed in cfg['seeds']:
        for key in keys:jobs.append((f'regularization:{key}:{seed}','regularization',key,seed))
    else:
      for seed in cfg['seeds']:
        for key in keys:jobs.append((f'ablation:{key}:{seed}','ablation',key,seed))
      for seed in cfg['seeds']:
        for key in keys:
          jobs.extend([(f'extra_gpu:{key}:{seed}','extra_gpu',key,seed),(f'extra_cpu:{key}:{seed}','extra_cpu',key,seed)])
    if retained_scope():jobs.append(('cost:all','cost','all',0))
    jobs.extend([('analysis:all','analysis','all',0),('delivery:all','delivery','all',0)])
    for id,kind,key,seed in jobs:c.execute('insert or ignore into jobs values(?,?,?,?,?,?,?,?,?,?)',(id,kind,key,seed,'pending',None,0,None,None,None))
    c.commit()
    manifest=read_json(HERE/'run_manifest.json',{});manifest.update(registered_tasks=len(jobs),registered_by_kind={k:sum(j[1]==k for j in jobs) for k in set(j[1] for j in jobs)},
         task_database=str(DB),last_registry_update=time.time(),required_unimplemented_stages_are_not_skipped=True)
    atomic_json(HERE/'run_manifest.json',manifest)
    return c

def marker(job):
    id,kind,key,seed,*_=job
    if kind=='classical':return HERE/'results'/key/'classical/complete.json'
    if kind in ('main','ablation','regularization'):return HERE/'results'/key/f'seed{seed}/{kind}_complete.json'
    return HERE/'results'/key/f'seed{seed}/{kind}_complete.json' if kind in ('extra_gpu','extra_cpu') else HERE/f'{kind}_complete.json'

def main():
    lock=(HERE/'supervisor.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            lock.close();raise RuntimeError('An experiment supervisor already holds the scheduler lock')
    c=init_db();atomic_json(HERE/'supervisor_process.json',dict(pid=os.getpid(),started=time.time(),command='supervisor.py'))
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    runtime_gpu=ROOT/'.venv_gpu/Scripts/python.exe';runtime_cpu=ROOT/'.venv/Scripts/python.exe'
    running={}
    for job in c.execute('select * from jobs where status="running"').fetchall():
        if not required_job(job):
            raise RuntimeError(f'Out-of-scope running job must be explicitly stopped before restart: {job[0]}')
        if marker(job).exists():c.execute('update jobs set status="complete",finished=? where id=?',(time.time(),job[0]))
        elif job[5] and psutil.pid_exists(job[5]):
            try:
                proc=psutil.Process(job[5]);cmd=' '.join(proc.cmdline())
                if str(HERE).lower() in cmd.lower():running[job[0]]=(proc,None,job[1]);continue
            except psutil.Error:pass
            c.execute('update jobs set status="pending",pid=null,error="abandoned worker, resuming" where id=?',(job[0],))
        else:c.execute('update jobs set status="pending",pid=null where id=?',(job[0],))
    c.commit();last_status=0
    try:
      while True:
        for id,(proc,handle,kind) in list(running.items()):
            alive=proc.poll() is None if isinstance(proc,subprocess.Popen) else proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
            if alive:continue
            job=c.execute('select * from jobs where id=?',(id,)).fetchone();ok=marker(job).exists()
            error=None
            if not ok:
                path=LOGS/(id.replace(':','_')+'.log');error=path.read_text(encoding='utf-8',errors='replace')[-4000:] if path.exists() else 'worker ended without completion marker'
            status='complete' if ok else ('pending' if job[6]<3 else 'error')
            c.execute('update jobs set status=?,finished=?,error=? where id=?',(status,time.time(),error,id))
            if handle:handle.close()
            del running[id];c.commit()
        historical_jobs=c.execute('select * from jobs order by rowid').fetchall()
        jobs=[j for j in historical_jobs if required_job(j)];statuses={j[0]:j[4] for j in jobs}
        pending=[j for j in jobs if j[4]=='pending']
        gpu_count=sum(k in ('main','ablation','extra_gpu','regularization','cost') for _,_,k in running.values())
        cpu_count=len(running)-gpu_count
        available=psutil.virtual_memory().available/1e9;cpu_max=config()['cpu_workers'] if available>2.5 else 1
        disk_free=psutil.disk_usage(str(HERE)).free/1e9
        for job in pending:
            id,kind,key,seed,*_=job
            gpu=kind in ('main','ablation','extra_gpu','regularization','cost')
            if disk_free<15:break
            if gpu and gpu_count>=config()['gpu_workers']:continue
            if not gpu and cpu_count>=cpu_max:continue
            if not gpu and available<1.25:continue
            # Tree ensembles retain large source/target point arrays; serialize these jobs.
            if kind=='extra_cpu' and any(k=='extra_cpu' for _,_,k in running.values()):continue
            if kind in ('ablation','extra_gpu','extra_cpu','regularization') and statuses.get(f'main:{key}:{seed}')!='complete':continue
            if kind=='regularization' and any(j[4]!='complete' for j in jobs if j[1]=='main'):continue
            if kind=='cost' and any(j[4]!='complete' for j in jobs if j[1]=='main'):continue
            if kind=='analysis' and any(j[4]!='complete' for j in jobs if j[1] not in ('analysis','delivery')):continue
            if kind=='delivery' and statuses.get('analysis:all')!='complete':continue
            if marker(job).exists():
                c.execute('update jobs set status="complete",finished=? where id=?',(time.time(),id));c.commit();continue
            script=HERE/('gpu_worker.py' if kind in ('main','ablation','regularization') else 'generator_cost_retained.py' if kind=='cost' else 'classical_worker.py' if kind=='classical' else 'extra_worker.py' if kind in ('extra_gpu','extra_cpu') else 'finalize_retained_run.py' if retained_scope() else 'finalize_full_run.py')
            # Missing implementation remains pending and attention-visible; it is never marked complete.
            if not script.exists():continue
            args=[str(runtime_gpu if gpu else runtime_cpu),str(script)]
            if kind=='classical':args+=[key]
            elif kind in ('main','ablation','regularization'):args+=[key,str(seed),kind]
            elif kind in ('extra_gpu','extra_cpu'):args+=[key,str(seed),kind]
            else:args+=[kind]
            log=LOGS/(id.replace(':','_')+'.log');handle=log.open('a',encoding='utf-8')
            env=os.environ.copy();env['OMP_NUM_THREADS']='1' if not gpu else '2';env['MKL_NUM_THREADS']=env['OMP_NUM_THREADS'];env['PYTHONUNBUFFERED']='1'
            proc=subprocess.Popen(args,cwd=HERE,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            c.execute('update jobs set status="running",pid=?,attempt=attempt+1,started=?,finished=null,error=null where id=?',(proc.pid,time.time(),id));c.commit()
            running[id]=(proc,handle,kind)
            if gpu:gpu_count+=1
            else:cpu_count+=1
        if time.time()-last_status>15:
            historical_jobs=c.execute('select * from jobs').fetchall();jobs=[j for j in historical_jobs if required_job(j)]
            counts={s:sum(j[4]==s for j in jobs) for s in ('pending','running','complete','error')}
            missing=sorted({j[1] for j in jobs if j[4]=='pending' and j[1] in ('extra_cpu','extra_gpu','analysis','delivery') and not (HERE/('extra_worker.py' if j[1].startswith('extra_') else 'finalize_full_run.py')).exists()})
            state=dict(status='complete' if counts['complete']==len(jobs) else 'running' if running else 'needs_attention',
                total=len(jobs),counts=counts,running=[dict(id=id,pid=proc.pid,kind=k) for id,(proc,_,k) in running.items()],
                updated=time.time(),source_curves=read_json(HERE/'data/manifest.json')['source_curves'],target_curves=read_json(HERE/'data/manifest.json')['target_curves'],
                scope_profile=scope_description().get('profile','original_full_scope'),all_target_main_scope=True,
                all_target_lambda_scope=retained_scope(),all_target_ablation_scope=not retained_scope(),
                historical_total=len(historical_jobs),cancelled_scope_tasks=sum(j[4]=='cancelled_scope' for j in historical_jobs),
                optional_completed_retained=sum(j[4]=='complete' and not required_job(j) for j in historical_jobs),
                missing_stage_implementations=missing,disk_available_GB=round(disk_free,1),ram_available_GB=round(available,1),
                error_tasks=[dict(id=j[0],error=j[9]) for j in jobs if j[4]=='error'],not_finished=counts['complete']!=len(jobs))
            atomic_json(HERE/'status.json',state);last_status=time.time()
            if state['status']=='complete':break
        time.sleep(5)
    finally:
        c.close()
        lock.close()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
