import time, resource, json, torch
from neuralop.models import FNO
torch.manual_seed(0); torch.set_num_threads(4)
def rel_l2(p,y): return ((p-y).flatten(1).norm(dim=1)/y.flatten(1).norm(dim=1)).mean()
res=[]
for name,(R,W,M,L,B) in {"A 64^2 w32 m12 L4 b16":(64,32,12,4,16),
                       "B 128^2 w32 m12 L4 b16":(128,32,12,4,16),
                       "C 64^2 w64 m16 L4 b16":(64,64,16,4,16),
                       "D 32^2 w32 m12 L4 b16":(32,32,12,4,16)}.items():
    m=FNO(n_modes=(M,M),hidden_channels=W,in_channels=1,out_channels=1,n_layers=L)
    opt=torch.optim.AdamW(m.parameters(),1e-3)
    x=torch.randn(B,1,R,R); y=torch.randn(B,1,R,R)
    for i in range(2):
        opt.zero_grad(); rel_l2(m(x),y).backward(); opt.step()
    n=5; t=time.perf_counter()
    for i in range(n):
        opt.zero_grad(); rel_l2(m(x),y).backward(); opt.step()
    step=(time.perf_counter()-t)/n
    m.eval()
    with torch.no_grad():
        x1=torch.randn(1,1,R,R); [m(x1) for _ in range(3)]
        t=time.perf_counter(); [m(x1) for _ in range(20)]; inf=(time.perf_counter()-t)/20
    r=dict(cfg=name,params=sum(p.numel() for p in m.parameters()),train_step_s=round(step,4),per_sample_ms=round(1000*step/B,2),infer_b1_ms=round(1000*inf,2),peak_rss_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(json.dumps(r)); res.append(r)
json.dump(res,open("cpu_step_benchmark_results.json","w"),indent=1)
