# Offline pen reward audit

This scores existing development trajectories using a proposed reward, without
training, changing the policy, or changing the frozen acceptance gate. These are
new reward hypotheses, not the original training reward. See the generated
`runs/pen-reward-analysis-v0/REPORT.md` for results.

From the repository root:

```bash
runs/pen-runtime/venv/bin/python pen/analysis/analyze.py \
  runs/pen-scaling/64 runs/pen-scaling/128 runs/pen-scaling/256 \
  --out runs/pen-reward-analysis-v0
.venv/bin/python -m unittest discover -s pen/tests -v
```

Analysis needs NumPy and Matplotlib; the tests need only NumPy. The installed pen
runtime already has both analysis dependencies. No GPU or simulator is needed.

## Candidate v0

At each control step, reward is the sum of the following components. Thresholds
come from `pen/policy/acceptance.json`; weights are explicit in
`reward_config.json`. One step is dt = 1/60 second.

| Component | Definition / budget |
|---|---|
| Rotation | 20 × change in forward high-water heading, capped at 3 turns, divided by 6π. Retracing old progress earns nothing. |
| Turn timing | +3 at each first turn crossing if that revolution took at most 3 s; maximum +9. |
| Hold shaping | 4 × change in best eligible partial hold fraction, with angular speed, heading range, contact support, and drop checks. |
| Hold bonus | +6 once a valid 1 s hold completes within the gate's allowed start window after turn three. |
| Thumb shaping | 2 × change in partial thumb-free interval potential: capped duration fraction × capped net rotation fraction, gated by support and no drop. |
| Thumb bonus | +6 for the first qualifying maximal thumb-free interval, committed at recontact or the third-turn boundary. |
| Participation | 6 × change in bounded quota satisfaction. Average of (a) best two middle/ring/pinky quotas per revolution and (b) best two revolution quotas for each non-thumb finger. Each quota caps active contact at 0.3 s. |
| Completion proxy | +40 once all timing, hold, thumb, participation, joint, and PhysX penetration conditions pass, without drop or unsettled start. |
| Drop | −40 once; stops the episode. |
| Time | −0.1 × dt. |
| Instability | −0.2 × dt × mean squared proximity to drop thresholds, ramping from half to the full threshold for downward displacement, lateral displacement, and tilt. |
| Braking | After turn three: −0.2 × dt × clip(abs(omega_z)/(30 deg/s) − 1, 0, 2)². |
| Joint limits | −1 × dt × clip(max joint violation / 0.02 rad, 0, 4)². |
| Penetration proxy | −0.5 × dt × clip(PhysX penetration / 0.002 m − 0.5, 0, 2)². |
| Action change | −0.1 × dt × mean squared action change; initial previous action is zero. |

Rewards stop at the first drop, first valid hold, or end of the saved trajectory.
This is counterfactual termination accounting; the source policy ran for 12 s.
Incomplete hold/thumb potentials roll back to zero at termination; completed
potentials latch. Plain potential differences can still reward unsuccessful
cycles under discounting. This known flaw is tested and reported, not corrected
in v0. Participation also permits tiny positive motion plus joint jitter.

The completion proxy does not read strict score labels. Those labels come from
the independent evaluator, including MuJoCo penetration replay. Recorded PhysX
penetration cannot replace that independent measurement; the initial PhysX sample
was not recorded and is assumed zero here. Hold and thumb calculations include
the gate's synthetic initial sample.

## Outputs and interpretation

Each batch gets `trials.csv`, `components.npz` (one trial × control-step array per
reward term plus seeds), and `summary.json`. The top-level report includes pass
versus fail means, ranking with and without completion bonus, discount sensitivity,
proxy disagreements, constructed failure patterns, and figures. Input and code
hashes are recorded in the output for provenance.

The 64/128/256 batches overlap in seeds: 448 realizations, 256 distinct seeds.
Treat the 256 batch as primary and the others as cross-checks. Final evaluation
seeds 41100000–41100099 are rejected. Ranking on this frozen policy's rollouts
cannot establish whether a reward will train a better policy. Synthetic fixtures
check reward accounting but are not physically validated trajectories.
