"""Minimal GPU/process/collective check before allocating simulation environments."""
import json
import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
world = int(os.environ['WORLD_SIZE'])
assert world == 4 and torch.cuda.device_count() == 4
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=120))
value = torch.ones(1, device=f'cuda:{rank}')
dist.all_reduce(value)
assert value.item() == world
print(json.dumps(dict(rank=rank, gpu=torch.cuda.get_device_name(rank),
                      memory_bytes=torch.cuda.get_device_properties(rank).total_memory,
                      nccl='PASS')), flush=True)
dist.destroy_process_group()
