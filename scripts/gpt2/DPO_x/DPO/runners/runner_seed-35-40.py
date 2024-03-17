import os
import sys

port_base = 2050
seeds = [35, 40]
ports = [port_base + i for i in range(2)]

if False:
    print(seeds)
    print(ports)
    sys.exit(0)

for port, seed in zip(ports, seeds):
    os.system(f'bash scripts/gpt2/DPO_x/DPO/seed/seqkd-DPO_base-seed_input.sh {port} --seed {seed}')