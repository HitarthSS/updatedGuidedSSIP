"""Audited, resumable Oscillate reproduction with disjoint development checks."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.update(SDL_VIDEODRIVER='dummy', MPLBACKEND='Agg', OMP_NUM_THREADS='4')
for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'CODEX_API_KEY', 'GEMINI_API_KEY']:
    os.environ.pop(key, None)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def status(state, **kw):
    out = dict(state=state, time=time.time(), pid=os.getpid(), **kw)
    write(ROOT/'oscillate-status.json', out)
    print(json.dumps(out), flush=True)


def main():
    lock = (ROOT/'oscillate.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '5', 'GPU 5 required'
    sys.path[:0] = [str(ROOT/'upstream/Push-T'), str(ROOT/'upstream')]
    import numpy as np
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    source = ROOT/'upstream/Push-T/inference/dynamic/inference_steg_dynamic.py'
    ckpt = ROOT/'upstream/Push-T/checkpoints/pusht_state_1000ep_si.ckpt'
    data = ROOT/'upstream/Push-T/data/pusht_cchi_v7_replay.zarr.zip'
    manifest = dict(
        benchmark='PushT dynamic Oscillate only, paper Table II', mode=1,
        paper_reported_success_rate=0.56, horizon=250, multiplier=1,
        development_seeds=[900301, 900302, 900303], evaluation_seeds=list(range(500, 550)),
        arms=['authors', 'my_harness'], scale=7, N=64, K=3,
        primary_metric='no collision and final reward > 0.85 (paper)',
        secondary_metric='no collision and max reward > 0.85 (release)',
        initialization='released reset(seed), clean reference computed separately for obstacle placement',
        reference_horizon=200, reference_is_benchmark_setup_not_retry=True,
        limitations=['paper seed manifest unverified; seeds are released script values',
                    'paper final vs release maximum reward mismatch; both recorded',
                    'Oscillate only; Chase and static headline not reproduced',
                    'shared GPU 5 timing; A6000 differs from paper A5000'],
        source_commit=subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
        hashes={str(p.relative_to(ROOT)): digest(p) for p in [source, ckpt, data, ROOT/'my_harness.py', Path(__file__).resolve()]},
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
        physical_gpu=5, external_api_calls=0)
    frozen_path = ROOT/'oscillate-frozen.json'
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == manifest, 'Frozen manifest changed'
    else:
        write(frozen_path, manifest)
    status('loading_models')
    spec = importlib.util.spec_from_file_location('released_dynamic_steg', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    nets = [m.ema_si_velocity_net, m.ema_si_denoiser_net]
    for net in nets:
        net.eval()
        for param in net.parameters():
            param.requires_grad_(False)
    from my_harness import install
    original_guided = m.run_guided_inference_visualized
    change = install(m)
    improved_guided = m.run_guided_inference_visualized
    write(ROOT/'my-harness-design.json', change)
    record = {}
    phase = 'episode'
    original_step, original_reset = m.env.step, m.env.reset
    original_reference = m.run_clean_baseline

    def reset(*a, **kw):
        result = original_reset(*a, **kw)
        state = np.asarray(m.env._get_obs(), dtype=np.float64)
        record[phase] = dict(steps=0, simulator_seconds=0., rewards=[],
                             actions=[], initial_state=state.tolist(),
                             initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        return result

    def step(action):
        r = record[phase]
        limit = 200 if phase == 'reference' else 250
        assert r['steps'] < limit, 'Horizon violation'
        action = np.asarray(action)
        assert np.isfinite(action).all(), 'Non-finite action'
        start = time.perf_counter()
        result = original_step(action)
        r['simulator_seconds'] += time.perf_counter()-start
        r['steps'] += 1
        r['rewards'].append(float(result[1]))
        r['actions'].append(action.tolist())
        return result

    def reference(*a, **kw):
        nonlocal phase
        phase = 'reference'
        torch.cuda.synchronize()
        start = time.perf_counter()
        trajectory = original_reference(*a, **kw)
        torch.cuda.synchronize()
        record['reference']['wall_seconds'] = time.perf_counter()-start
        record['reference']['trajectory_sha256'] = hashlib.sha256(trajectory.tobytes()).hexdigest()
        phase = 'episode'
        return trajectory

    m.env.reset, m.env.step, m.run_clean_baseline = reset, step, reference
    calls = {}
    def hook(name):
        def count(net, args):
            key = phase + '/' + name
            c = calls.setdefault(key, dict(calls=0, rows=0))
            c['calls'] += 1
            c['rows'] += int(args[0].shape[0])
        return count
    handles = [net.register_forward_pre_hook(hook(name)) for net, name in zip(nets, ['velocity', 'denoiser'])]
    # Rendering remains identical in both arms and is separately timed.
    render = m.env.render
    def timed_render(*a, **kw):
        start = time.perf_counter()
        out = render(*a, **kw)
        record['render_seconds'] = record.get('render_seconds', 0.) + time.perf_counter()-start
        return out
    m.env.render = timed_render

    def run(split, arm, seed):
        nonlocal phase
        output = ROOT/'runs-oscillate'/split/arm/f'{seed}.json'
        if output.exists():
            return json.loads(output.read_text())
        phase = 'episode'; record.clear(); calls.clear()
        m.run_guided_inference_visualized = original_guided if arm == 'authors' else improved_guided
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); start = time.perf_counter()
        imgs, metrics = m.run_guided_inference_visualized(
            obstacle_mode=1, start_obs_pos_norm=[0.5, 0.5], max_guidance_scale=7., seed=seed)
        torch.cuda.synchronize(); wall = time.perf_counter()-start
        e = record['episode']; ref = record['reference']
        final = e['rewards'][-1] if e['rewards'] else 0.
        assert e['steps'] <= 250 and ref['steps'] <= 200
        assert np.isclose(max(e['rewards'], default=0.), metrics['reward'])
        result = dict(split=split, arm=arm, seed=seed, mode=1, horizon=250,
                      steps=e['steps'], horizon_violations=0,
                      paper_success=bool(not metrics['collision'] and final > .85),
                      release_success=bool(metrics['success']), collision=bool(metrics['collision']),
                      final_reward=final, max_reward=float(metrics['reward']),
                      min_distance=float(metrics['min_dist']),
                      wall_seconds=wall, episode_wall_seconds=wall-ref['wall_seconds'],
                      reference=ref, episode=e, render_seconds=record.get('render_seconds', 0.),
                      model_calls=dict(calls), gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      actions_sha256=hashlib.sha256(np.array(e['actions'], dtype=np.float64).tobytes()).hexdigest(),
                      final_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                      external_api_calls=0, external_api_tokens=0)
        write(output, result)
        del imgs
        return result

    def matched(b, i, exact=False):
        assert b['episode']['initial_state_sha256'] == i['episode']['initial_state_sha256'], 'Initial-state mismatch'
        assert b['reference']['trajectory_sha256'] == i['reference']['trajectory_sha256'], 'Obstacle reference mismatch'
        if exact:
            assert b['actions_sha256'] == i['actions_sha256'], 'Development action equality failed'
            assert b['final_rng_sha256'] == i['final_rng_sha256'], 'Development RNG equality failed'
            assert b['paper_success'] == i['paper_success'] and b['collision'] == i['collision']

    try:
        dev = []
        for j, seed in enumerate(manifest['development_seeds']):
            rows = {}
            for arm in (manifest['arms'] if j % 2 == 0 else list(reversed(manifest['arms']))):
                status('development', arm=arm, seed=seed)
                rows[arm] = run('dev', arm, seed)
            matched(rows['authors'], rows['my_harness'], exact=True)
            dev.append(rows)
        write(ROOT/'development-validation.json', dict(action_exact=True, rng_exact=True, pairs=len(dev)))
        status('development_passed', pairs=len(dev), action_exact=True, rng_exact=True)
        results = []
        for j, seed in enumerate(manifest['evaluation_seeds']):
            rows = {}
            for arm in (manifest['arms'] if j % 2 == 0 else list(reversed(manifest['arms']))):
                status('eval', arm=arm, seed=seed, completed_pairs=len(results), target_pairs=50)
                rows[arm] = run('eval', arm, seed)
            matched(rows['authors'], rows['my_harness'])
            results.append(rows)
            summary = dict(pairs=len(results), target_pairs=50, horizon_violations=0,
                           matched_action_hashes=sum(x['authors']['actions_sha256'] == x['my_harness']['actions_sha256'] for x in results),
                           arms={arm: dict(episodes=len(results), paper_successes=sum(x[arm]['paper_success'] for x in results),
                                           release_successes=sum(x[arm]['release_success'] for x in results),
                                           collisions=sum(x[arm]['collision'] for x in results),
                                           episode_wall_seconds=sum(x[arm]['episode_wall_seconds'] for x in results),
                                           reference_wall_seconds=sum(x[arm]['reference']['wall_seconds'] for x in results),
                                           model_rows=sum(sum(c['rows'] for k,c in x[arm]['model_calls'].items() if k.startswith('episode/')) for x in results)) for arm in manifest['arms']},
                           external_api_calls=0, external_api_tokens=0,
                           timing_note='shared GPU 5, alternating arms, simulator and reference separately recorded')
            write(ROOT/'oscillate-comparison.json', summary)
        status('complete', comparison=summary)
    finally:
        for handle in handles:
            handle.remove()
        m.env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status('failed', error=repr(e))
        raise
