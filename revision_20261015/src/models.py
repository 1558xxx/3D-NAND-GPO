"""Same 609-parameter 1-32-16-1 tanh curve model, without legacy imports."""
import time
import numpy as np
import torch
from torch import nn
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import erf

def new_vector(seed):
    torch.manual_seed(int(seed))
    model = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,16), nn.Tanh(), nn.Linear(16,1))
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()

def forward(vectors, x):
    w1 = vectors[:, :32].reshape(-1, 1, 32)
    b1 = vectors[:, 32:64].reshape(-1, 1, 32)
    h1 = torch.tanh(x.unsqueeze(-1)*w1+b1)
    w2 = vectors[:,64:576].reshape(-1,16,32).transpose(1,2)
    h2 = torch.tanh(torch.bmm(h1,w2)+vectors[:,576:592].unsqueeze(1))
    return torch.sum(h2*vectors[:,592:608].unsqueeze(1),dim=-1)+vectors[:,608].unsqueeze(1)

def predict(vector, x):
    with torch.no_grad():
        log_y = forward(torch.tensor(np.asarray(vector).reshape(1,-1),dtype=torch.float32),
                        torch.tensor(np.asarray(x).reshape(1,-1),dtype=torch.float32)).numpy()[0]
    return np.maximum(np.expm1(np.clip(log_y,-20,12)),0)

def adapt_batch(initial, x, y, epochs, lr, strength, mask=None, optimizer="adam"):
    # Summing per-curve losses gives each vector the same gradient scale as a separate fit.
    start = time.perf_counter()
    reference = torch.tensor(initial,dtype=torch.float32)
    v = nn.Parameter(reference.clone())
    tx, ty = torch.tensor(x,dtype=torch.float32), torch.log1p(torch.tensor(y,dtype=torch.float32))
    valid = torch.ones_like(tx) if mask is None else torch.tensor(mask,dtype=torch.float32)
    def data_losses(vec):
        return (((forward(vec,tx)-ty)**2)*valid).sum(dim=1)/valid.sum(dim=1)
    opt = (torch.optim.Adam([v],lr=lr) if optimizer=="adam" else
           torch.optim.LBFGS([v],lr=lr,max_iter=5,history_size=20,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7,tolerance_change=1e-9))
    best, best_loss = reference.clone(), torch.full((len(v),),float("inf"))
    bad = torch.zeros(len(v),dtype=torch.bool)
    for _ in range(epochs):
        if optimizer=="lbfgs":
            def closure():
                opt.zero_grad()
                losses=data_losses(v)+strength*((v-reference)**2).mean(dim=1)
                total=losses.sum()
                total.backward()
                return total
            opt.step(closure)
            with torch.no_grad():
                losses=data_losses(v)+strength*((v-reference)**2).mean(dim=1)
                finite=torch.isfinite(losses)
                bad |= ~finite
                improved=finite & (losses<best_loss)
                best[improved]=v[improved]
                best_loss[improved]=losses[improved]
                if not finite.all():
                    v.copy_(best)
                    break
            continue
        opt.zero_grad()
        data = data_losses(v)
        penalty = ((v-reference)**2).mean(dim=1)
        losses = data + strength*penalty
        finite = torch.isfinite(losses)
        bad |= ~finite
        if not finite.any():
            break
        torch.where(finite,losses,torch.zeros_like(losses)).sum().backward()
        if v.grad is not None:
            v.grad.nan_to_num_(nan=0.0,posinf=0.0,neginf=0.0)
            norms = torch.linalg.vector_norm(v.grad,dim=1).clamp_min(1e-12)
            v.grad.mul_((10/norms).clamp_max(1).unsqueeze(1))
        # Snapshot and loss refer to exactly the same pre-update parameter vector.
        with torch.no_grad():
            improved = finite & (losses < best_loss)
            best[improved] = v[improved]
            best_loss[improved] = losses[improved]
        opt.step()
    with torch.no_grad():
        data = data_losses(v)
        penalty = ((v-reference)**2).mean(dim=1)
        losses = data+strength*penalty
        improved = torch.isfinite(losses) & (losses < best_loss)
        best[improved] = v[improved]
        final_data = data_losses(best)
        final_penalty = strength*((best-reference)**2).mean(dim=1)
    return best.numpy(), dict(data_loss=final_data.numpy(), penalty_loss=final_penalty.numpy(),
                             nonfinite=bad.numpy(), seconds=(time.perf_counter()-start)/len(v))

def pooled_source_vector(tasks, scale, epochs, lr, seed, optimizer="adam"):
    x = np.concatenate([scale(t["steps"]) for t in tasks]).reshape(1,-1)
    y = np.concatenate([t["freqs"] for t in tasks]).reshape(1,-1)
    return adapt_batch(new_vector(seed)[None,:],x,y,epochs,lr,0,optimizer=optimizer)[0][0]

def classical(method, x, y, grid):
    order=np.argsort(x)
    x,y=np.asarray(x)[order],np.asarray(y)[order]
    if method in ("spline","pchip"):
        ctor=CubicSpline if method=="spline" else PchipInterpolator
        kwargs={"bc_type":"natural"} if method=="spline" else {}
        return np.maximum(ctor(x,y,extrapolate=True,**kwargs)(grid),0), False
    def gauss(xx,a,m,s,b):
        return b+a*np.exp(-0.5*((xx-m)/s)**2)
    def skew(xx,a,m,s,b,alpha):
        z=(xx-m)/s
        return b+a*np.exp(-0.5*z*z)*(1+erf(alpha*z/np.sqrt(2)))
    span=max(float(np.ptp(x)),1)
    ymax=max(float(y.max()),1)
    weights=np.maximum(y,1)
    initial=[max(float(y.max()-y.min()),1),float(np.average(x,weights=weights)),max(span/6,0.3),max(float(y.min()),0)]
    lower=[0,x.min()-span,0.1,0]
    upper=[ymax*5+1,x.max()+span,span*3,ymax*2+1]
    func=gauss
    if method=="skew_gaussian":
        initial.append(0)
        lower.append(-10)
        upper.append(10)
        func=skew
    try:
        pars,_=curve_fit(func,x,y,p0=initial,bounds=(lower,upper),maxfev=5000)
        return np.maximum(func(grid,*pars),0),False
    except (RuntimeError,ValueError,FloatingPointError):
        return np.maximum(func(grid,*initial),0),True
