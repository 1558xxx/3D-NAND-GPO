"""Measure original-generator single-vector latency, not hardware acquisition cost."""
import time
import numpy as np
import torch
from full_common import HERE,load_data,atomic_json,read_json
from generative_models import fit_generator

def main():
    rows=[]
    for chip in ['TH58TFT1T23BA8H','MT29F1T08EEHAF']:
        key=chip+'_Bake_P2';d=load_data(key);folder=HERE/'results'/key/'seed42'
        params=np.load(folder/'source_params.npy')
        net,record=fit_generator(params,d['condition'][d['src']],d['wordlines'],42,'main',folder/'generator_main.pt')
        cond=torch.tensor(d['condition'][d['target'][:1]],dtype=torch.float32,device='cuda')
        durations=[]
        with torch.no_grad():
            for _ in range(3):net.sample(cond,1)
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter();net.sample(cond,1);torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True,record_shapes=True) as prof:
                net.sample(cond,1);torch.cuda.synchronize()
        rows.append(dict(chip=chip,checkpoint=str(folder/'generator_main.pt'),T=record['T'],seed=42,
            trainable_parameters=sum(p.numel() for p in net.parameters()),device=torch.cuda.get_device_name(),
            repeats=20,warmups=3,single_vector_seconds_median=float(np.median(durations)),
            single_vector_seconds_p05=float(np.quantile(durations,.05)),single_vector_seconds_p95=float(np.quantile(durations,.95)),
            profiler_supported_flops=int(sum(e.flops or 0 for e in prof.key_averages())),
            flops_scope='Profiler-supported arithmetic only; unsupported attention/elementwise/RNG operations may be omitted. Not a complete hardware-operation count.',
            latency_scope='eval-mode T10 generator sample of one 609-vector, CUDA synchronized; excludes loading, source training, target adaptation, measurement and firmware costs'))
        del net;torch.cuda.empty_cache()
    atomic_json(HERE/'generation_single_vector_costs.json',dict(scope='retained_reviews_20261014',records=rows))
    atomic_json(HERE/'cost_complete.json',dict(status='complete',scope='retained_reviews_20261014',chips=2,updated=time.time()))

if __name__=='__main__':main()
