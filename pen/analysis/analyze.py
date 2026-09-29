"""Measure candidate rewards on existing, scored development rollouts only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rewards import evaluate_trial, WEIGHTS, A
from synthetic import evaluate_scenarios


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    if not len(values):
        return None
    return dict(mean=float(values.mean()), min=float(values.min()), max=float(values.max()),
                p10=float(np.quantile(values,.1)), median=float(np.median(values)), p90=float(np.quantile(values,.9)))


def auc(values, labels):
    x, y = np.asarray(values)[labels], np.asarray(values)[~labels]
    if not len(x) or not len(y):
        return None
    return float(((x[:,None]>y).sum()+.5*(x[:,None]==y).sum())/(len(x)*len(y)))


def analyze(run, out):
    meta = json.loads((run/'meta.json').read_text())
    scores = json.loads((run/'score.json').read_text())
    n = meta['trials']
    seeds = list(range(meta['seed0'], meta['seed0']+n))
    if any(41100000 <= s <= 41100099 for s in seeds):
        raise ValueError('Reserved final-evaluation seeds must not be used for reward development.')
    if len(scores['trials']) != n or any(t['seed'] != seeds[i] for i,t in enumerate(scores['trials'])):
        raise ValueError('Score/trajectory seed mismatch')
    if not np.isclose(meta['control_dt'],1/A['control_hz']):
        raise ValueError('Unexpected control timestep')
    z = np.load(run/'trajectories.npz')
    if z['q'].shape[:2] != (n,meta['steps']):
        raise ValueError('Trajectory dimensions do not match metadata')
    common = {'q_lo','q_hi'}
    keys = ('q','q0','qd','pen_axis','pen_angvel','pen_pos','ref_pos','finger_force','action',
            'physx_penetration','heading0','q_lo','q_hi','unsettled')
    traces = {}; rows=[]; diagnostics=[]
    for i in range(n):
        rec={k:z[k] if k in common else z[k][i] for k in keys}
        if any(not np.isfinite(v).all() for v in rec.values()):
            raise ValueError(f'Non-finite input in trial {i}')
        components, diagnostic = evaluate_trial(rec,meta['joints'])
        diagnostics.append(diagnostic)
        row=dict(run=run.name,trial=i,seed=seeds[i],strict_pass=scores['trials'][i]['passed'],
                 **diagnostic)
        for k,v in components.items():
            traces.setdefault(k,[]).append(v)
            row[k]=float(v.sum())
        total=sum(components.values())
        row['total']=float(total.sum())
        row['without_completion']=row['total']-row['completion_proxy']
        for gamma in WEIGHTS['discount_factors']:
            row[f'discounted_{gamma}']=float(total @ (gamma**np.arange(len(total))))
        row['gate_failures']=';'.join(k for k,v in scores['trials'][i].items() if k.endswith('_ok') and not v)
        if scores['trials'][i]['drop']:row['gate_failures']+=';drop'
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'trials.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'components.npz',**{k:np.stack(v) for k,v in traces.items()},seeds=seeds)
    labels=np.array([r['strict_pass'] for r in rows],bool)
    summary=dict(run=str(run.resolve()),trials=n,unique_seeds=n,strict_passes=int(labels.sum()),
                 components={},ranking={},proxy_disagreement={},component_gate_disagreement={})
    for k in traces:
        values=np.array([r[k] for r in rows])
        summary['components'][k]=dict(all=stats(values),strict_pass=stats(values[labels]),strict_fail=stats(values[~labels]))
    for k in ('total','without_completion',*[f'discounted_{g}' for g in WEIGHTS['discount_factors']]):
        values=np.array([r[k] for r in rows]);order=np.argsort(-values,kind='stable')
        summary['ranking'][k]=dict(auc=auc(values,labels),all=stats(values),
            strict_pass=stats(values[labels]),strict_fail=stats(values[~labels]),
            top20_strict_passes=int(labels[order[:20]].sum()),
            highest_reward_failure=next((dict(seed=rows[j]['seed'],reward=float(values[j]),failures=rows[j]['gate_failures']) for j in order if not labels[j]),None))
    proxy=np.array([r['proxy_success'] for r in rows],bool)
    summary['proxy_disagreement']=dict(false_positive=int((proxy&~labels).sum()),false_negative=int((~proxy&labels).sum()),
        false_positive_trials=[dict(seed=r['seed'],gate_failures=r['gate_failures']) for r in rows if r['proxy_success'] and not r['strict_pass']])
    for component,target in [('hold_bonus','hold_ok'),('thumb_bonus','thumb_release_ok')]:
        predicted=np.array([r[component]>0 for r in rows]);actual=np.array([s[target] for s in scores['trials']])
        summary['component_gate_disagreement'][component]=int((predicted!=actual).sum())
    summary['input_hashes']={name:sha(run/name) for name in ('trajectories.npz','meta.json','score.json')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary,rows,traces


def plot(out, rows, traces, synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(traces);passed=np.array([r['strict_pass'] for r in rows],bool)
    means=np.array([[r[k] for k in keys] for r in rows])
    fig,axes=plt.subplots(2,1,figsize=(12,10),layout='constrained')
    x=np.arange(len(keys))
    for offset,mask,label,color in [(-.19,passed,'Strict pass','#2b9c77'),(.19,~passed,'Strict fail','#d87947')]:
        axes[0].bar(x+offset,means[mask].mean(axis=0),.38,label=label,color=color)
    axes[0].set_xticks(x,keys,rotation=40,ha='right');axes[0].set_ylabel('Mean undiscounted component return')
    axes[0].legend();axes[0].set_title('Candidate reward v0 — 256 saved rollouts (no training)')
    axes[1].hist([np.array([r['without_completion'] for r in rows])[passed],np.array([r['without_completion'] for r in rows])[~passed]],
                 bins=25,label=['Strict pass','Strict fail'],color=['#2b9c77','#d87947'],alpha=.75)
    axes[1].set_xlabel('Total return excluding the +40 proxy completion bonus');axes[1].set_ylabel('Trials');axes[1].legend()
    fig.savefig(out/'components.png',dpi=160);plt.close(fig)
    success=next(i for i,r in enumerate(rows) if r['strict_pass'])
    failure=max((i for i,r in enumerate(rows) if not r['strict_pass']),key=lambda i:rows[i]['total'])
    fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for ax,idx in zip(axs,[success,failure]):
        for k in ['rotation','timely_turn','hold_shaping','hold_bonus','thumb_shaping','thumb_bonus','participation','completion_proxy']:
            ax.plot(np.arange(1,len(traces[k][idx])+1)/60,np.cumsum(traces[k][idx]),label=k)
        ax.set_title(f"Seed {rows[idx]['seed']} · {'strict pass' if rows[idx]['strict_pass'] else rows[idx]['gate_failures']}")
        ax.set_ylabel('Cumulative component reward');ax.set_xlabel('Recorded time (s)');ax.legend(ncol=4,fontsize=8)
    fig.savefig(out/'example-traces.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    all_results=[];largest=None
    for run in args.runs:
        result=analyze(run,args.out/run.name);all_results.append(result[0])
        if largest is None or len(result[1])>len(largest[1]):largest=result
        print(f"Scored {run}: {result[0]['trials']} trajectories",flush=True)
    synthetic=evaluate_scenarios()
    (args.out/'synthetic.json').write_text(json.dumps(synthetic,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    provenance=dict(weights=WEIGHTS,acceptance_sha256=sha(Path(__file__).resolve().parents[1]/'policy/acceptance.json'),
                    analysis_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                    note='Offline candidate v0; no training, no simulator mutation; synthetic signals are not physically validated.')
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    plot(args.out,largest[1],largest[2],synthetic)
    primary=largest[0]
    lines=['# Offline pen reward analysis — candidate v0','',
        'No training was performed. The frozen acceptance gate and policy weights were not changed. All inputs are existing development rollouts. The 64/128/256 batches contain 448 trajectory realizations but only 256 distinct seeds; they are not 448 independent seeds. The largest batch is the primary analysis; smaller batches are cross-checks.',
        '',f"Primary batch: {primary['strict_passes']}/{primary['trials']} strict passes.",'',
        '| Component | Mean: strict pass | Mean: strict fail | Observed range |',
        '|---|---:|---:|---:|']
    for k,v in primary['components'].items():
        lines.append(f"| {k} | {v['strict_pass']['mean']:.4f} | {v['strict_fail']['mean']:.4f} | {v['all']['min']:.4f} … {v['all']['max']:.4f} |")
    lines+=['','## Ranking','', '| Return | Pass mean | Fail mean | AUC | Strict passes among top 20 |','|---|---:|---:|---:|---:|']
    for k,v in primary['ranking'].items():
        lines.append(f"| {k} | {v['strict_pass']['mean']:.4f} | {v['strict_fail']['mean']:.4f} | {v['auc']:.4f} | {v['top20_strict_passes']} |")
    lines+=['', 'AUC is the probability that a randomly selected strict pass scores above a strict fail (ties count as half). It is descriptive, not evidence of learning improvement. The no-completion row prevents the large success bonus from hiding weaknesses in the other terms.', '',
        '## PhysX completion proxy','', '```json', json.dumps(primary['proxy_disagreement'],indent=2), '```', '',
        '## Constructed signal checks','', '| Scenario | Total | Discounted (0.99) | Thumb shaping, discounted (0.99) |','|---|---:|---:|---:|']
    for k,v in synthetic.items():lines.append(f"| {k} | {v['total']:.4f} | {v['discounted_099']:.4f} | {v['thumb_shaping_discounted_099']:.6f} |")
    lines+=['','## Cross-check batches','',
        '| Environments | Strict passes | AUC without completion | Proxy false positives | Hold / thumb disagreements with gate |',
        '|---|---:|---:|---:|---:|']
    for result in all_results:
        disagreement=result['component_gate_disagreement']
        lines.append(f"| {result['trials']} | {result['strict_passes']} | {result['ranking']['without_completion']['auc']:.4f} | {result['proxy_disagreement']['false_positive']} | {disagreement['hold_bonus']} / {disagreement['thumb_bonus']} |")
    lines+=['','## Findings','',
        '- In these existing rollouts, rotation, turn timing, and participation are largely saturated in both passing and failing trials. Thumb release provides the clearest separation; hold contributes less. This does not mean rotation shaping would be unnecessary for a new policy.',
        '- The low observed joint-limit, instability, and penetration penalties do not establish that their weights are adequate. The existing controller explores a narrow behavior distribution.',
        '- Before PPO, address discounted partial-attempt credit and the penetration proxy mismatch, and strengthen participation credit against tiny-motion jitter. Keep this v0 as the measured baseline.',
        '', '## Interpretation and limitations','',
        '- Plain potential differences can pay for repeated partial attempts under discounting: positive credit arrives before its negative rollback. A discount-aware potential term with correct terminal handling is a candidate correction, not implemented in this v0.',
        '- Tiny positive pen motion plus joint jitter can earn participation credit because the frozen gate tests forward motion as a boolean. Capping credit bounds the exploit; it does not establish useful manipulation.',
        '- PhysX penetration is not the independent MuJoCo penetration measurement. The recorded initial PhysX penetration is unavailable and is assumed zero here. Existing strict score.json labels are kept independent of reward computation.',
        '- Reward traces end at the first drop or first valid hold, unlike the source evaluator which records the full 12 seconds. Completed bonuses latch; incomplete thumb/hold potentials are rolled back at the terminal step. This is counterfactual episode accounting, not a rerun of the policy.',
        '- Smoothness uses zero as the pre-first-action reference. No force/energy penalty was added: none was concretely specified in the proposal.',
        '- Penalties and the 4+6 hold / 2+6 thumb budget splits are explicit initial hypotheses in reward_config.json. These measurements do not tune or validate them.',
        '- Discount factors are illustrative sensitivity settings, not a recovered original PPO configuration. One control step is 1/60 second.',
        '- Synthetic traces test mathematical loopholes but are not dynamically consistent simulations. No final evaluation seeds were read.',
        '', 'Artifacts: per-batch trials.csv, components.npz (trial × control step per component), summary.json, input/code hashes in provenance.json, and synthetic.json.',
        '', '![Component means and distributions](components.png)', '', '![Example reward traces](example-traces.png)']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report:',args.out/'REPORT.md',flush=True)

if __name__=='__main__':main()
