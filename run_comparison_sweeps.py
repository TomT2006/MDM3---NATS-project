"""Re-run the supplied stand/fleet comparisons with separated random streams."""
from dataclasses import replace
from pathlib import Path
import json
from overload import run_group
from model import example_config
from cost_optimiser_integrated import write_csv


def run(folder,workers=2):
    folder=Path(folder);base=example_config();mapping={}
    for parameter,levels in [('stands_per_port',range(2,9)),('fleet',range(4,13))]:
        rows=[];mapping[parameter]=[]
        for level in levels:
            name='cost_baseline' if (parameter=='fleet' and level==8) or (parameter=='stands_per_port' and level==4) else f'cost_{parameter}_{level}'
            c=replace(base,**{parameter:level})
            group=run_group(c,range(20001,20031),folder/name,workers=workers)
            mapping[parameter].append(dict(level=level,folder=name))
            for r in group['rows']:
                rows.append(dict(parameter=parameter,parameter_value=level,cost=r['score'],**r))
        write_csv(rows,folder.parent/f'{parameter}_sweep_corrected.csv')
    (folder/'cost_plan.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print('Corrected comparison sweeps complete.',flush=True)


if __name__=='__main__':run(Path(__file__).with_name('overload_data'))
