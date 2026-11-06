"""Development selection, independent validation, then gated frozen evaluation."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from paired_campaign import write, digest
from accuracy_harness import build, CONFIGS

ROOT = Path(__file__).resolve().parent
os.environ.update(SDL_VIDEODRIVER='dummy', MPLBACKEND='Agg', OMP_NUM_THREADS='4')
for key in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','CODEX_API_KEY','GEMINI_API_KEY']:
    os.environ.pop(key,None)


def status(state, **kw):
    d = dict(state=state,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/'accuracy-status.json',d)
    print(json.dumps(d),flush=True)


def summary(rows):
    n=len(rows)
    out=dict(pairs=n,arms={})
    for arm in ['authors','my_harness']:
        rs=[r[arm] for r in rows]
        out['arms'][arm]=dict(episodes=n,successes=sum(r['paper_success'] for r in rs),
            release_successes=sum(r['release_success'] for r in rs),
            collisions=sum(r['collision'] for r in rs),
            mean_final_reward=sum(r['final_reward'] for r in rs)/n if n else None,
            episode_wall_seconds=sum(r['episode_wall_seconds'] for r in rs),
            simulator_seconds=sum(r['episode']['simulator_seconds'] for r in rs),
            reference_wall_seconds=sum(r['reference']['wall_seconds'] for r in rs),
            model_rows=sum(sum(v['rows'] for k,v in r['model_calls'].items() if k.startswith('episode/')) for r in rs),
            horizon_violations=sum(r['horizon_violations'] for r in rs))
    out['my_only_wins']=sum(r['my_harness']['paper_success'] and not r['authors']['paper_success'] for r in rows)
    out['authors_only_wins']=sum(r['authors']['paper_success'] and not r['my_harness']['paper_success'] for r in rows)
    out['success_rate_delta']=((out['arms']['my_harness']['successes']-out['arms']['authors']['successes'])/n) if n else None
    return out


class Runner:
    def __init__(self):
        sys.path[:0]=[str(ROOT/'upstream/Push-T'),str(ROOT/'upstream')]
        import numpy as np
        import torch
        self.np,self.torch=np,torch
        torch.set_num_threads(4)
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        source=ROOT/'upstream/Push-T/inference/dynamic/inference_steg_dynamic.py'
        spec=importlib.util.spec_from_file_location('released_dynamic_accuracy',source)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.m=m;self.record={};self.calls={};self.phase='episode'
        original=m.run_guided_inference_visualized
        self.functions={key:build(m,original,key) for key in CONFIGS}
        for net in [m.ema_si_velocity_net,m.ema_si_denoiser_net]:
            net.eval()
            for p in net.parameters():p.requires_grad_(False)
        self.original_step=m.env.step;self.original_reset=m.env.reset
        self.original_reference=m.run_clean_baseline
        m.env.step=self.step;m.env.reset=self.reset;m.run_clean_baseline=self.reference
        self.original_render=m.env.render;m.env.render=self.render
        self.handles=[net.register_forward_pre_hook(self.hook(name)) for net,name in
            [(m.ema_si_velocity_net,'velocity'),(m.ema_si_denoiser_net,'denoiser')]]

    def reset(self,*a,**kw):
        out=self.original_reset(*a,**kw)
        state=self.np.asarray(self.m.env._get_obs(),dtype=self.np.float64)
        self.record[self.phase]=dict(steps=0,simulator_seconds=0.,rewards=[],actions=[],observations=[],
            initial_state=state.tolist(),initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        return out

    def step(self,action):
        r=self.record[self.phase];limit=200 if self.phase=='reference' else 250
        assert r['steps']<limit,'Horizon violation'
        assert self.np.isfinite(action).all(),'Non-finite action'
        t=time.perf_counter();out=self.original_step(action)
        r['simulator_seconds']+=time.perf_counter()-t;r['steps']+=1
        r['rewards'].append(float(out[1]));r['actions'].append(self.np.asarray(action).tolist())
        obs=out[0][0] if isinstance(out[0],tuple) else out[0]
        r['observations'].append(self.np.asarray(obs).tolist())
        return out

    def reference(self,*a,**kw):
        self.phase='reference';self.torch.cuda.synchronize();t=time.perf_counter()
        trajectory=self.original_reference(*a,**kw)
        self.torch.cuda.synchronize()
        self.record['reference']['wall_seconds']=time.perf_counter()-t
        self.record['reference']['trajectory_sha256']=hashlib.sha256(trajectory.tobytes()).hexdigest()
        self.phase='episode';return trajectory

    def render(self,*a,**kw):
        t=time.perf_counter();out=self.original_render(*a,**kw)
        self.record['render_seconds']=self.record.get('render_seconds',0.)+time.perf_counter()-t
        return out

    def hook(self,name):
        def count(net,args):
            c=self.calls.setdefault(self.phase+'/'+name,dict(calls=0,rows=0))
            c['calls']+=1;c['rows']+=int(args[0].shape[0])
        return count

    def run(self,split,variant,seed):
        p=ROOT/'runs-accuracy'/split/variant/f'{seed}.json'
        if p.exists():return json.loads(p.read_text())
        status(split,variant=variant,seed=seed)
        self.phase='episode';self.record.clear();self.calls.clear()
        self.m.run_guided_inference_visualized=self.functions[variant]
        self.torch.cuda.reset_peak_memory_stats();self.torch.cuda.synchronize();start=time.perf_counter()
        imgs,metrics=self.m.run_guided_inference_visualized(obstacle_mode=1,start_obs_pos_norm=[.5,.5],max_guidance_scale=7.,seed=seed)
        self.torch.cuda.synchronize();wall=time.perf_counter()-start
        e=self.record['episode'];ref=self.record['reference']
        final=e['rewards'][-1] if e['rewards'] else 0.
        assert e['steps']<=250 and ref['steps']<=200
        assert self.np.isclose(max(e['rewards'],default=0.),metrics['reward'])
        r=dict(split=split,variant=variant,seed=seed,horizon=250,steps=e['steps'],horizon_violations=0,
            paper_success=bool(not metrics['collision'] and final>.85),release_success=bool(metrics['success']),
            collision=bool(metrics['collision']),final_reward=final,max_reward=float(metrics['reward']),
            min_distance=float(metrics['min_dist']),wall_seconds=wall,episode_wall_seconds=wall-ref['wall_seconds'],
            episode=e,reference=ref,model_calls=dict(self.calls),render_seconds=self.record.get('render_seconds',0.),
            gpu_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),external_api_calls=0,external_api_tokens=0)
        write(p,r);del imgs
        return r

    def pair(self,split,variant,seed,index):
        rs={}
        order=['authors',variant] if index%2==0 else [variant,'authors']
        for key in order:rs[key]=self.run(split,key,seed)
        b,i=rs['authors'],rs[variant]
        assert b['episode']['initial_state_sha256']==i['episode']['initial_state_sha256'],'State mismatch'
        assert b['reference']['trajectory_sha256']==i['reference']['trajectory_sha256'],'Reference mismatch'
        return dict(authors=b,my_harness=i)

    def close(self):
        for h in self.handles:h.remove()
        self.m.env.close()


def main():
    lock=(ROOT/'accuracy.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='5'
    names=['settle_only']
    files=[ROOT/'accuracy_campaign.py',ROOT/'accuracy_harness.py',ROOT/'paired_campaign.py',
        ROOT/'ACCURACY_PLAN.md',ROOT/'upstream/Push-T/inference/dynamic/inference_steg_dynamic.py',
        ROOT/'upstream/Push-T/checkpoints/pusht_state_1000ep_si.ckpt',
        ROOT/'upstream/Push-T/data/pusht_cchi_v7_replay.zarr.zip']
    protocol=dict(candidates=names,configs=CONFIGS,selection_seeds=list(range(900301,900307)),
        validation_seeds=list(range(900501,900521)),released_eval_seeds=list(range(500,550)),
        fresh_robustness_seeds=list(range(550,600)),horizon=250,mode=1,physical_gpu=5,
        selection='max final-reward successes, fewer collisions, mean final reward, simplicity',
        validation_gate='strict success improvement in selection AND independent validation',
        primary_metric='no collision and final reward>.85; original termination unchanged',
        prior_eval_aggregate_known=True,exact_paper_seed_manifest_unknown=True,
        hashes={str(p.relative_to(ROOT)):digest(p) for p in files},external_api_calls=0)
    pp=ROOT/'accuracy-protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Protocol changed'
    else:write(pp,protocol)
    status('loading_models')
    runner=Runner()
    try:
        candidates={}
        # Interleave candidate variants per seed to reduce ordering effects.
        for j,seed in enumerate(protocol['selection_seeds']):
            order=names[j%len(names):]+names[:j%len(names)]
            for k,name in enumerate(order):
                candidates.setdefault(name,[]).append(runner.pair('selection',name,seed,j+k))
                write(ROOT/'accuracy-selection.json',{n:summary(rs) for n,rs in candidates.items()})
        def score(name):
            d=summary(candidates[name])['arms']['my_harness']
            return (d['successes'],-d['collisions'],d['mean_final_reward'],-names.index(name))
        selected=max(names,key=score)
        result=summary(candidates[selected]);b,i=[result['arms'][a]['successes'] for a in ['authors','my_harness']]
        write(ROOT/'accuracy-selected.json',dict(variant=selected,selection=result))
        if i<=b:
            status('selection_failed',variant=selected,comparison=result,reason='No development success improvement; no final evaluation started')
            return
        validation=[]
        for j,seed in enumerate(protocol['validation_seeds']):
            validation.append(runner.pair('validation',selected,seed,j))
            write(ROOT/'accuracy-validation.json',summary(validation))
        result=summary(validation);b,i=[result['arms'][a]['successes'] for a in ['authors','my_harness']]
        if i<=b:
            status('validation_failed',variant=selected,comparison=result,reason='Independent validation did not improve success; no final evaluation started')
            return
        frozen=dict(variant=selected,protocol_sha256=digest(pp),source_hashes=protocol['hashes'],
            validation=result,selection=summary(candidates[selected]),no_further_tuning=True)
        fp=ROOT/'accuracy-frozen.json'
        if fp.exists():assert json.loads(fp.read_text())==frozen,'Frozen candidate changed'
        else:write(fp,frozen)
        for split,seeds in [('released_eval',protocol['released_eval_seeds']),('fresh_robustness',protocol['fresh_robustness_seeds'])]:
            rows=[]
            for j,seed in enumerate(seeds):
                rows.append(runner.pair(split,selected,seed,j))
                write(ROOT/(split+'-comparison.json'),summary(rows))
        status('complete',variant=selected)
    finally:runner.close()


if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
