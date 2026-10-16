#!/usr/bin/env python3
"""Small CPU teaching experiments, not reproductions of foundation-model results."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def run():
    rng=np.random.default_rng(336)
    out={}
    length=np.array([256,512,1024,2048,4096,8192])
    out['00']={'kind':'analytic budget','x':length.tolist(),'series':{'MHA KV MiB':(2*32*length*32*128*2/2**20).tolist(),'GQA-8 KV MiB':(2*32*length*8*128*2/2**20).tolist()},'xlabel':'Cached tokens','ylabel':'KV cache (MiB)','assumptions':'B=1, L=32, head_dim=128, BF16; excludes weights, activations, allocator and fragmentation.'}
    n=np.array([2,4,8,16,32,64])
    out['01']={'kind':'analytic loss baseline','x':n.tolist(),'series':{'CLIP one-direction':np.log(n).tolist(),'SigLIP / N':(n*np.log(2)).tolist()},'xlabel':'Batch size N','ylabel':'Loss at zero logits','assumptions':'Zero logits and bias; different normalizations, so magnitudes cannot rank model quality.'}
    t=np.arange(4096)/4096
    signal=.7*np.sin(2*np.pi*7*t)+.2*np.sin(2*np.pi*19*t)
    residual=signal.copy(); errors=[]
    for q in range(1,9):
        step=2**(-q)
        code=np.clip(np.round(residual/step),-2,2)*step
        residual-=code
        errors.append(float(np.mean(residual**2)))
    out['02']={'kind':'executed scalar residual quantizer','x':list(range(1,9)),'series':{'Residual MSE':errors},'xlabel':'Residual quantization stages','ylabel':'Mean squared error','assumptions':'Fixed scalar grids and deterministic two-tone signal. Not trained RVQ, EnCodec or perceptual audio quality.'}
    frames=np.array([8,16,32,64,128,256]); duration=60.; event=.2
    phase=rng.random(100000)
    measured=[float(np.mean(phase*(duration/n)<event)) for n in frames]
    out['03']={'kind':'executed random-phase event sampling','x':frames.tolist(),'series':{'Monte Carlo capture':measured,'Exact random-phase model':np.minimum(1,frames*event/duration).tolist()},'xlabel':'Frames per 60-second clip','ylabel':'Event capture probability','assumptions':'Uniform grid with random phase, 0.2 s event, perfect detection if sampled. Not VLM accuracy.'}
    arrivals=np.arange(200)*.02; streams={}
    for service in [.015,.025]:
        finish=0.; delay=[]
        for arrival in arrivals:
            finish=max(finish,arrival)+service; delay.append(float(finish-arrival))
        streams[f'service {service*1000:.0f} ms']=delay
    out['04']={'kind':'executed deterministic queue','x':arrivals.tolist(),'series':streams,'xlabel':'Input time (seconds)','ylabel':'Completion delay (seconds)','assumptions':'One FIFO server; 20 ms arrivals, no batching or network. Illustrates queue instability, not Omni latency.'}
    alpha=np.linspace(.02,.98,40)
    out['05']={'kind':'analytic noise-error amplification','x':alpha.tolist(),'series':{'x0 error / epsilon error':np.sqrt((1-alpha)/alpha).tolist()},'xlabel':'Cumulative alpha','ylabel':'Error amplification','assumptions':'Fixed additive error in epsilon prediction; x0 reconstruction formula, not measured image generation.'}
    curves={}
    for chunk in [1,4,16]:
        x=0.; action=0.; trajectory=[]
        for k in range(80):
            if k==30:x-=.4
            if k%chunk==0:action=2*(1-x)
            x+=.05*action;trajectory.append(x)
        curves[f'chunk {chunk}']=trajectory
    out['06']={'kind':'executed scalar feedback controller','x':((np.arange(80)+1)*.05).tolist(),'series':curves,'xlabel':'Time (seconds)','ylabel':'Position (target = 1)','assumptions':'Integrator, proportional gain 2, dt=50 ms, disturbance -0.4 just before the update starting at 1.5 s; plotted states are at update ends. Not robot-policy success.'}
    steps=np.arange(1,41); p=.95
    out['07']={'kind':'analytic independent-step failure model','x':steps.tolist(),'series':{'one attempt':(p**steps).tolist(),'one local retry':((1-(1-p)**2)**steps).tolist()},'xlabel':'Task length (steps)','ylabel':'Full-task success probability','assumptions':'Independent failures, perfect verifier, safe idempotent retries; correlated failures violate the model.'}
    out['08']={'kind':'analytic roofline bound','x':list(range(1,65)),'series':{'Weight-only intensity':list(range(1,65))},'xlabel':'Batch size B','ylabel':'FLOP / byte (BF16 weights)','assumptions':'Dense matmul ~2BP FLOPs and ~2P weight bytes; ignores KV/activation traffic. No GPU timing claim.'}
    gap=np.linspace(-6,6,81)
    out['09']={'kind':'analytic DPO derivative','x':gap.tolist(),'series':{f'beta={b}':(-b/(1+np.exp(b*gap))).tolist() for b in [.1,.5,1.]},'xlabel':'Policy minus reference preference gap','ylabel':'Derivative of loss w.r.t. gap','assumptions':'Single preferred pair, sequence-summed log probabilities; no claim about visual grounding.'}
    outcomes=np.array([0]*80+[-1]*10+[1]*20+[0]*90)
    bootstrap=rng.choice(outcomes,(4000,200),replace=True).mean(axis=1)
    ci=np.quantile(bootstrap,[.025,.975])
    out['10']={'kind':'executed paired bootstrap on constructed outcomes','x':list(range(len(bootstrap))),'series':{'Paired accuracy difference':bootstrap.tolist()},'xlabel':'Bootstrap resample','ylabel':'B minus A accuracy','estimate':float(outcomes.mean()),'ci95':ci.tolist(),'assumptions':'Constructed 200 independent items: both correct 80, A-only 10, B-only 20, both wrong 90. Seed 336; 4000 item-level bootstrap draws. Not a model benchmark.'}
    mult=np.arange(1,11); fractions=[]
    for m in mult:
        family=np.repeat(np.arange(1000),m)
        train=rng.random(len(family))<.8
        seen=set(family[train]); test=family[~train]
        fractions.append(float(np.mean([f in seen for f in test])))
    out['11']={'kind':'executed duplicate-family split simulation','x':mult.tolist(),'series':{'Record random split leakage':fractions,'Family-disjoint split':[0.]*len(mult)},'xlabel':'Copies per family','ylabel':'Test records with family in train','assumptions':'1000 families, independent 80/20 record split vs family-disjoint partition; duplicate membership, not downstream accuracy.'}
    conf=rng.normal(size=10000); noise=rng.normal(size=10000)
    action=conf+.3*noise; y=.2*action+.8*conf+.1*rng.normal(size=10000)
    intervened=rng.normal(size=10000); yi=.2*intervened+.8*conf+.1*rng.normal(size=10000)
    slopes=[float(np.cov(a,b,ddof=0)[0,1]/np.var(a)) for a,b in [(action,y),(intervened,yi)]]
    out['12']={'kind':'executed confounded linear-system simulation','x':[0,1,2],'series':{'Estimated effect':[slopes[0],slopes[1],.2]},'labels':['observational OLS','randomized action OLS','true action coefficient'],'xlabel':'Action identification setting','ylabel':'Estimated coefficient','assumptions':'Constructed dynamics y=.2*a+.8*c+noise; observational a=c+.3*noise, randomized a independent of c. Not a trained world model.'}
    assert np.isclose(out['00']['series']['GQA-8 KV MiB'][4],512)
    assert errors[-1]<errors[0]
    assert slopes[0]>.8 and abs(slopes[1]-.2)<.04
    return out

if __name__=='__main__':
    target=ROOT/'sources/probes.json'
    target.write_text(json.dumps({'seed':336,'scope':'CPU teaching experiments and analytic calculations; never foundation-model reproductions','probes':run()},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(target)
