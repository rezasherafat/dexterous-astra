"""Replay saved state arrays through reward v1, without running physics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from reward import PenReward


def replay(records, joints, device='cpu'):
    n = len(records)
    model = PenReward(n, device, joints)
    def stack(key):return torch.as_tensor(np.stack([r[key] for r in records]),device=device,dtype=torch.float32)
    fields = {k:stack(k) for k in ('pen_pos','ref_pos','pen_axis','pen_angvel','finger_force','qd','q','q_lo','q_hi','action','physx_penetration')}
    progress = []
    for r in records:
        heading = np.unwrap(np.r_[float(r['heading0']),np.arctan2(r['pen_axis'][:,1].astype(float),r['pen_axis'][:,0].astype(float))])
        progress.append(heading[1:]-heading[0])
    progress = torch.tensor(np.array(progress),dtype=torch.float32,device=device)
    traces = {};stop = torch.zeros(n,device=device,dtype=torch.long)
    for t in range(progress.shape[1]):
        timeout = torch.full((n,),t==progress.shape[1]-1,device=device,dtype=torch.bool)
        _, done, components = model.step(progress=progress[:,t],relative_pos=fields['pen_pos'][:,t]-fields['ref_pos'],
             axis=fields['pen_axis'][:,t],omega_z=fields['pen_angvel'][:,t,2],finger_force=fields['finger_force'][:,t],
             qd=fields['qd'][:,t],q=fields['q'][:,t],q_lo=fields['q_lo'],q_hi=fields['q_hi'],
             action=fields['action'][:,t],penetration=fields['physx_penetration'][:,t],timeout=timeout)
        stop = torch.where((stop==0)&(done|timeout),t+1,stop)
        for k,v in components.items():traces.setdefault(k,[]).append(v)
    return {k:torch.stack(v,1).cpu().numpy() for k,v in traces.items()},stop.cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    meta=json.loads((args.run/'meta.json').read_text())
    if any(41100000<=meta['seed0']+i<=41100099 for i in range(meta['trials'])):raise ValueError('Reserved seeds')
    with np.load(args.run/'trajectories.npz') as z:
        keys=('pen_pos','ref_pos','pen_axis','pen_angvel','finger_force','qd','q','q_lo','q_hi','action','physx_penetration','heading0')
        data={k:z[k] for k in keys}
    records=[{k:v if k in ('q_lo','q_hi') else v[i] for k,v in data.items()} for i in range(meta['trials'])]
    c,stop=replay(records,meta['joints'],args.device)
    scores=json.loads((args.run/'score.json').read_text())['trials'];labels=np.array([s['passed'] for s in scores])
    summary=dict(device=args.device,trials=len(records),components={k:dict(pass_mean=float(v.sum(1)[labels].mean()),fail_mean=float(v.sum(1)[~labels].mean())) for k,v in c.items()},
                 hold_gate_disagreement=int(np.count_nonzero((c['hold'].sum(1)>0)!=[s['hold_ok'] for s in scores])),
                 thumb_gate_disagreement=int(np.count_nonzero((c['thumb'].sum(1)>0)!=[s['thumb_release_ok'] for s in scores])))
    args.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out/'components.npz',**c,stop=stop)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
