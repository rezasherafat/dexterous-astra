"""Import completed TensorBoard training metrics into W&B without retraining."""
import argparse
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--project', default='dexterous-astra')
    parser.add_argument('--entity')
    parser.add_argument('--mode', choices=['online','offline'], default='online')
    args = parser.parse_args()
    summary = json.loads((args.run/'summary.json').read_text())
    config = json.loads((args.run/'config.json').read_text())
    events = EventAccumulator(str(args.run), size_guidance={'scalars':0}).Reload()
    steps = {}
    for tag in events.Tags()['scalars']:
        if tag.endswith('/time'):continue  # Old wall-time axes are not iteration indices.
        for event in events.Scalars(tag):
            steps.setdefault(event.step,{})[tag] = event.value
    if not steps:raise ValueError('No scalar metrics found')
    with wandb.init(project=args.project, entity=args.entity, mode=args.mode,
                    name=args.run.name, dir=str(args.run.resolve()),
                    config={**config, 'imported_from_tensorboard':True}) as run:
        for step,values in sorted(steps.items()):run.log(values,step=step)
        run.summary.update(summary)
        info=dict(id=run.id,url=run.url,mode=args.mode,project=args.project,
                  metric_steps=len(steps),scalar_tags=events.Tags()['scalars'],directory=run.dir)
        (args.run/'wandb-run.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info,indent=2))

if __name__=='__main__':main()
