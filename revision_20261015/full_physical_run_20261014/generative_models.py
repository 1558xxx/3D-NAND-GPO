from pathlib import Path
import sys,time,importlib
import numpy as np
import torch
from torch import nn
from scipy.optimize import brentq
from full_common import ROOT,config
from gpu_ops import seed_all,ConditionalMean,CVAE,ConditionalFlow
sys.path.insert(0,str(ROOT/'paper_revision/source_reference/original_generator'))
from TimeTransformer import Transformer3
from denoising_diffusion_pytorch import GaussianDiffusion1D
importlib.import_module(GaussianDiffusion1D.__module__).tqdm=lambda iterable,*args,**kwargs:iterable

def diffusion_model(T,wordlines,matched=False):
    m=importlib.import_module(GaussianDiffusion1D.__module__);original=m.linear_beta_schedule
    if matched:
        baseline=float(torch.prod(1-original(10)));raw=original(T)
        factor=brentq(lambda f:float(torch.prod(1-f*raw))-baseline,1e-7,1.)
        m.linear_beta_schedule=lambda n:original(n)*factor
    try:
        net=Transformer3(d_input=1,d_model=16,d_output=1,num_wl=wordlines,N=4,layernum=609,dropout=.1,pe='original')
        d=GaussianDiffusion1D(net,seq_length=609,timesteps=T,loss_type='l2',objective='pred_v',auto_normalize=False,beta_schedule='linear')
    finally:m.linear_beta_schedule=original
    return d.to('cuda')

def changed_conditions(c,variant):
    c=c.copy()
    for name,column in [('noWL',0),('noStress',1),('noPEC',2)]:
        if variant==name:c[:,column]=0
    if variant=='unconditional':c[:]=0
    return c

def fit_generator(params,c,wordlines,seed,variant,path,force_updates=None):
    cfg=config();seed_all(seed);start=time.perf_counter()
    if path.exists():
        record=torch.load(path,map_location='cpu',weights_only=False)
    else:
        kind='ensemble' if variant.startswith('ensemble_member') else ('diffusion' if variant not in ('cvae','flow','ensemble','dropout') else variant)
        mean=params.mean(0);std=np.maximum(params.std(0),1e-3)
        cm=c.mean(0);cs=np.maximum(c.std(0),1e-3);cs[0]=max(wordlines-1,1)
        T=50 if variant.startswith('T50') else 10
        net=diffusion_model(T,wordlines,variant=='T50_matched') if kind=='diffusion' else (
            CVAE().cuda() if kind=='cvae' else ConditionalFlow().cuda() if kind=='flow' else ConditionalMean(.1 if kind=='dropout' else 0).cuda())
        cc=changed_conditions(c,variant) if kind=='diffusion' else (c-cm)/cs
        opt=torch.optim.Adam(net.parameters(),lr=cfg['generator_lr'])
        history=[];seen=np.zeros(len(params),np.int32);batch=cfg['generator_batch']
        epoch_count=cfg['generator_epochs'] if force_updates is None else int(np.ceil(force_updates/np.ceil(len(params)/batch)))
        for epoch in range(epoch_count):
            order=np.random.default_rng(seed+epoch).permutation(len(params))
            for a in range(0,len(params),batch):
                if force_updates is not None and len(history)>=force_updates:break
                idx=order[a:a+batch];seen[idx]+=1
                zz=params[idx] if kind=='diffusion' else (params[idx]-mean)/std
                z=torch.as_tensor(zz,dtype=torch.float32,device='cuda');cond=torch.as_tensor(cc[idx],dtype=torch.float32,device='cuda')
                if kind=='diffusion':loss=net(z[:,None,:],cond)
                elif kind=='cvae':pred,kl=net(z,cond);loss=(pred-z).square().mean()+.001*kl
                elif kind=='flow':
                    z=torch.cat([z,torch.randn((len(z),1),device='cuda')],1);noise,ld=net.transform(z,cond)
                    loss=(.5*noise.square().sum(1)-ld).mean()/610
                else:loss=(net(cond)-z).square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10);opt.step()
                value=float(loss.detach());history.append(value)
                if not np.isfinite(value):raise FloatingPointError('Generator training loss became nonfinite')
            print(f'{variant} source pass {epoch+1}/{epoch_count}: updates {len(history)}',flush=True)
        if force_updates is not None:assert len(history)==force_updates
        assert seen.min()>=cfg['generator_epochs']
        record=dict(kind=kind,variant=variant,T=T,wordlines=wordlines,seed=seed,source_vectors=len(params),epochs=float(seen.sum()/len(params)),
            configured_full_passes=cfg['generator_epochs'],forced_optimizer_updates=force_updates,
            batch=batch,optimizer_updates=len(history),min_source_exposure=int(seen.min()),max_source_exposure=int(seen.max()),
            history=history,mean=mean,std=std,condition_mean=cm,condition_std=cs,
            parameters=sum(p.numel() for p in net.parameters()),training_seconds=time.perf_counter()-start,
            model={k:v.detach().cpu() for k,v in net.state_dict().items()},torch=torch.__version__,device='RTX3060 CUDA12.6')
        temp=path.with_suffix('.tmp');torch.save(record,temp);temp.replace(path)
    kind=record['kind']
    net=diffusion_model(record['T'],wordlines,variant=='T50_matched') if kind=='diffusion' else (
        CVAE().cuda() if kind=='cvae' else ConditionalFlow().cuda() if kind=='flow' else ConditionalMean(.1 if kind=='dropout' else 0).cuda())
    net.load_state_dict(record['model']);net.eval()
    return net,record

def generate(net,record,c,seed,variant):
    cfg=config();seed_all(10000+seed);kind=record['kind'];cc=changed_conditions(c,variant)
    if kind!='diffusion':cc=(c-record['condition_mean'])/record['condition_std']
    values=[];start=time.perf_counter()
    with torch.no_grad():
        for a in range(0,len(c),cfg['sampling_batch']):
            cond=torch.tensor(cc[a:a+cfg['sampling_batch']],dtype=torch.float32,device='cuda')
            if kind=='diffusion':p=net.sample(cond,len(cond)).reshape(len(cond),609)
            elif kind in ('cvae','flow'):p=net.sample(cond)
            elif kind=='dropout':
                net.train();p=torch.stack([net(cond) for _ in range(8)]).mean(0);net.eval()
            else:p=net(cond)
            result=p.cpu().numpy()
            if kind!='diffusion':result=result*record['std']+record['mean']
            values.append(result.astype(np.float32))
    result=np.concatenate(values)
    if not np.isfinite(result).all():raise FloatingPointError('Generated initialization is nonfinite')
    return result,time.perf_counter()-start
