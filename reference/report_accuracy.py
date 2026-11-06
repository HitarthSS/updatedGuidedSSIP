"""Read-only matched outcomes, paired uncertainty and protocol audit."""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parent


def wilson(s,n):
    if not n:return None
    z=1.95996398454;p=s/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def paired(split,variant):
    lookup={}
    for arm in ['authors',variant]:
        lookup[arm]={r['seed']:r for p in (ROOT/'runs-accuracy'/split/arm).glob('*.json')
                     for r in [json.loads(p.read_text())]}
    seeds=sorted(lookup['authors'].keys()&lookup[variant].keys())
    pairs=[]
    for seed in seeds:
        b,i=lookup['authors'][seed],lookup[variant][seed]
        assert b['episode']['initial_state_sha256']==i['episode']['initial_state_sha256']
        assert b['reference']['trajectory_sha256']==i['reference']['trajectory_sha256']
        assert b['steps']<=250 and i['steps']<=250
        pairs.append((b,i))
    n=len(pairs);out=dict(pairs=n,seeds=seeds,arms={})
    for arm,index in [('authors',0),('my_harness',1)]:
        rs=[p[index] for p in pairs];s=sum(r['paper_success'] for r in rs)
        out['arms'][arm]=dict(successes=s,episodes=n,success_rate=s/n if n else None,
            wilson_95=wilson(s,n),release_successes=sum(r['release_success'] for r in rs),
            collisions=sum(r['collision'] for r in rs),
            episode_wall_seconds=sum(r['episode_wall_seconds'] for r in rs),
            simulator_seconds=sum(r['episode']['simulator_seconds'] for r in rs),
            render_seconds=sum(r['render_seconds'] for r in rs),
            reference_wall_seconds=sum(r['reference']['wall_seconds'] for r in rs),
            model_rows=sum(sum(v['rows'] for k,v in r['model_calls'].items() if k.startswith('episode/')) for r in rs),
            horizon_violations=sum(r['steps']>250 for r in rs))
    wins=sum(i['paper_success'] and not b['paper_success'] for b,i in pairs)
    losses=sum(b['paper_success'] and not i['paper_success'] for b,i in pairs)
    out.update(my_only_wins=wins,authors_only_wins=losses)
    if n:
        deltas=[int(i['paper_success'])-int(b['paper_success']) for b,i in pairs]
        rng=random.Random(731)
        bs=sorted(sum(rng.choice(deltas) for _ in range(n))/n for _ in range(10000))
        discordant=wins+losses
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
        out.update(success_rate_delta=(wins-losses)/n,paired_bootstrap_95=[bs[249],bs[9749]],
                   exact_mcnemar_two_sided_p=p)
    return out


selected_path=ROOT/'accuracy-selected.json'
selected=json.loads(selected_path.read_text())['variant'] if selected_path.exists() else None
out=dict(selected=selected)
for name in ['accuracy-status.json','accuracy-selection.json','accuracy-validation.json']:
    p=ROOT/name
    if p.exists():out[name]=json.loads(p.read_text())
if selected:
    for split in ['validation','released_eval','fresh_robustness']:
        out[split]=paired(split,selected)
out['caveats']=['Oscillate condition only; exact paper seed manifest unknown',
    'Released evaluation is repeated evaluation with its earlier aggregate known',
    'Fresh seeds550-599 are a separately labelled robustness extension, not released evaluation seeds',
    'Primary final reward/no collision; secondary maximum-reward score retained',
    'Shared GPU5 timing; episode wall includes rendering and simulator, reference separate',
    'Model rows measure computation; pure inference latency not isolated']
out['external_api_calls']=0
(ROOT/'accuracy-report.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
