import torch
import numpy as np
import re, subprocess
from sys import platform

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def model_memory_in_bytes(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs # in bytes

def get_apple_hardware():
    "Get apple hardware info"
    cpu_info = subprocess.run(["system_profiler","SPHardwareDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    gpu_info = subprocess.run(["system_profiler","SPDisplaysDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8") 
    system_info = dict(
        cpu = re.search(r'Chip:\s+(.+)', cpu_info).group(1),
        cpu_cores = re.search(r'Number of Cores:\s+(\d+)', cpu_info).group(1),
        memory = re.search(r'Memory:\s+(\d+)\s+GB', cpu_info).group(1),
        gpu = re.search(r'Chipset Model:\s+(.+)', gpu_info).group(1),
        gpu_cores = re.search(r'Total Number of Cores:\s+(\d+)', gpu_info).group(1),
        )
    return system_info

def get_gpu_name():
    if platform == "darwin":
        system_info = get_apple_hardware()
        return f"{system_info['gpu']} GPU {system_info['gpu_cores']} Cores"
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return None
