import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)

def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def new_vector(seed):
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(1,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1))
    return torch.nn.utils.parameters_to_vector(net.parameters()).detach().numpy().copy()

def forward(v,x):
    h=torch.tanh(x.unsqueeze(-1)*v[:,:32].reshape(-1,1,32)+v[:,32:64].reshape(-1,1,32))
    h=torch.tanh(torch.bmm(h,v[:,64:576].reshape(-1,16,32).transpose(1,2))+v[:,576:592].unsqueeze(1))
    return (h*v[:,592:608].unsqueeze(1)).sum(-1)+v[:,608].unsqueeze(1)

def predict(v,x,device='cuda'):
    with torch.no_grad():
        out=forward(torch.as_tensor(v,dtype=torch.float32,device=device),torch.as_tensor(x,dtype=torch.float32,device=device))
        return torch.expm1(out.clamp(-20,12)).clamp_min(0).cpu().numpy()

def adapt(initial,x,y,mask,steps=140,lr=.8,strength=0,optimizer='lbfgs',validation=None,device='cuda'):
    ref=torch.as_tensor(initial,dtype=torch.float32,device=device);v=nn.Parameter(ref.clone())
    x=torch.as_tensor(x,dtype=torch.float32,device=device);y=torch.log1p(torch.as_tensor(y,dtype=torch.float32,device=device))
    mask=torch.as_tensor(mask,dtype=torch.float32,device=device)
    def losses():
        data=((forward(v,x)-y).square()*mask).sum(1)/mask.sum(1)
        penalty=strength*(v-ref).square().mean(1)
        return data,penalty
    def monitor(data,penalty):
        if validation is None:return data+penalty
        vx,vy,vm=validation
        return ((forward(v,vx)-vy).square()*vm).sum(1)/vm.sum(1)
    if validation is not None:
        validation=tuple(torch.as_tensor(a,dtype=torch.float32,device=device) for a in validation)
        validation=(validation[0],torch.log1p(validation[1]),validation[2])
    opt=torch.optim.LBFGS([v],lr=lr,max_iter=5,history_size=20,line_search_fn='strong_wolfe') if optimizer=='lbfgs' else torch.optim.Adam([v],lr=lr)
    best=ref.clone();best_score=torch.full((len(v),),float('inf'),device=device);bad=torch.zeros(len(v),dtype=torch.bool,device=device)
    with torch.no_grad():
        a,b=losses();m=monitor(a,b);ok=torch.isfinite(m);best_score[ok]=m[ok]
    for k in range(steps):
        if optimizer=='lbfgs':
            def closure():
                opt.zero_grad();a,b=losses();total=(a+b).sum();total.backward();return total
            opt.step(closure)
        else:
            opt.zero_grad();a,b=losses();(a+b).sum().backward()
            v.grad.nan_to_num_(nan=0,posinf=0,neginf=0)
            norm=torch.linalg.vector_norm(v.grad,dim=1).clamp_min(1e-12)
            v.grad.mul_((10/norm).clamp_max(1).unsqueeze(1));opt.step()
        with torch.no_grad():
            a,b=losses();m=monitor(a,b);finite=torch.isfinite(m)&torch.isfinite(a)
            bad|=~finite;improved=finite&(m<best_score)
            best[improved]=v[improved];best_score[improved]=m[improved]
            if not finite.all():v.copy_(best);break
    with torch.no_grad():
        v.copy_(best);a,b=losses()
        info=dict(nonfinite=bad.cpu().numpy(),data_loss=a.cpu().numpy(),penalty_loss=b.cpu().numpy(),monitor=best_score.cpu().numpy())
    return best.cpu().numpy(),info

class ConditionalMean(nn.Module):
    def __init__(self,dropout=0):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Dropout(dropout),nn.Linear(64,64),nn.Tanh(),nn.Dropout(dropout),nn.Linear(64,609))
    def forward(self,c):return self.net(c)

class CVAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(612,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.mu=nn.Linear(64,16);self.logvar=nn.Linear(64,16);self.decoder=nn.Sequential(nn.Linear(19,64),nn.Tanh(),nn.Linear(64,128),nn.Tanh(),nn.Linear(128,609))
    def forward(self,z,c):
        h=self.encoder(torch.cat([z,c],1));mu=self.mu(h);lv=self.logvar(h).clamp(-8,8)
        q=mu+torch.randn_like(mu)*torch.exp(.5*lv);out=self.decoder(torch.cat([q,c],1))
        kl=-.5*(1+lv-mu.square()-lv.exp()).mean()
        return out,kl
    def sample(self,c):return self.decoder(torch.cat([torch.randn((len(c),16),device=c.device),c],1))

class ConditionalFlow(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([nn.Sequential(nn.Linear(308,64),nn.Tanh(),nn.Linear(64,610)) for _ in range(6)])
    def transform(self,z,c,reverse=False):
        logdet=torch.zeros(len(z),device=z.device);layers=list(enumerate(self.layers));layers=layers[::-1] if reverse else layers
        for k,net in layers:
            if k%2:z=torch.cat([z[:,305:],z[:,:305]],1)
            a,b=z[:,:305],z[:,305:];st=net(torch.cat([a,c],1));s=1.5*torch.tanh(st[:,:305]);t=st[:,305:]
            b=(b-t)*torch.exp(-s) if reverse else b*torch.exp(s)+t
            logdet+=-s.sum(1) if reverse else s.sum(1);z=torch.cat([a,b],1)
            if k%2:z=torch.cat([z[:,305:],z[:,:305]],1)
        return z,logdet
    def sample(self,c):return self.transform(torch.randn((len(c),610),device=c.device),c,True)[0][:,:609]
