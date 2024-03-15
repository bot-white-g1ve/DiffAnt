# #!/bin/bash
 
# #PBS -q gpuvolta
# #PBS -P jp09
# #PBS -l walltime=24:00:00
# #PBS -l mem=16GB
# #PBS -l ncpus=12
# #PBS -l ngpus=1
# #PBS -l jobfs=50GB
# #PBS -l wd
# #PBS -l storage=scratch/jp09
# #PBS -M daochang.liu@sydney.edu.au

# module load use.own
# module load pytorch/1.10.0
# module load ASDiffusion

# python3 -u main.py --config configs/baseline.json > baseline.log

import os
import pdb
import numpy as np

prefix = 'RDV-T45AnticipationTry'

program_file = ['main.py', 'test.py'][0]

configs = [i for i in os.listdir('configs') if prefix in i]

print(program_file)

print(len(configs))

configs.sort()
configs_groups = np.array_split(np.array(configs), np.ceil(len(configs) / 6)) # How many programs on each GPU

print(len(configs_groups))

for g_id, group in enumerate(configs_groups):
    
    script_file = f'./{prefix}-Group{g_id}.sh'

    script = [
        '#!/bin/bash',
         
        '#PBS -q gpuvolta',
        '#PBS -P zg12',
        '#PBS -l walltime=10:00:00',
        '#PBS -l mem=50GB',
        '#PBS -l ncpus=12',
        '#PBS -l ngpus=1',
        '#PBS -l jobfs=64GB',
        '#PBS -l storage=gdata/zg12',
        '#PBS -l wd',
        '#PBS -M daochang.liu@sydney.edu.au',

        'module load use.own',
        'module load pytorch/1.10.0',
        'module load ASDiffusion',
    ]

    for c_id, config in enumerate(group):
        if c_id != len(group) - 1:
            cmd = f'python3 -u {program_file} --config configs/{config} --device -1 > ./{config}.log &'
        else:
            cmd = f'python3 -u {program_file} --config configs/{config} --device -1 > ./{config}.log'
        script.append(cmd)

    with open(script_file, 'w') as file:
        for line in script:
            file.write(line + '\n')

    os.system('qsub {}'.format(script_file))
