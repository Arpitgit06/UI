"""Sequential local model policy for an 8 GiB laptop GPU."""
SUPPORTED_MODEL='Qwen/Qwen2-VL-7B-Instruct'
CODER_MODEL='Qwen/Qwen2.5-Coder-7B-Instruct'


def configure_recognition(torch,model_name,device_index=0):
    if model_name!=SUPPORTED_MODEL:
        raise ValueError(f'Recognition requires {SUPPORTED_MODEL}; no automatic model substitution')
    result=configure_cuda_stage(torch,'recognition',device_index)
    result.update({'model':model_name,'quantization':'nf4-double-quant'})
    return result


def configure_coder(torch,model_name,device_index=0):
    if model_name!=CODER_MODEL:
        raise ValueError(f'Code generation requires {CODER_MODEL}')
    result=configure_cuda_stage(torch,'code-generation',device_index)
    result.update({'model':model_name,'quantization':'nf4-double-quant'})
    return result


def configure_cuda_stage(torch,stage,device_index=0):
    if not torch.cuda.is_available():
        raise RuntimeError(f'The local {stage} stage requires CUDA; CPU OCR remains available')
    free,total=torch.cuda.mem_get_info(device_index)
    budget=min(7*1024**3,int(total*.875),free-512*1024**2)
    if budget<3*1024**3:
        raise RuntimeError(f'Insufficient free GPU memory for {stage}; close other GPU applications')
    torch.cuda.set_device(device_index)
    torch.cuda.set_per_process_memory_fraction(budget/total,device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    return {'profile':'local-8gib','stage':stage,'model_count':1,
            'allocator_budget_bytes':budget,'device_total_bytes':total,'free_before_bytes':free,
            'note':'PyTorch allocator budget; driver and non-PyTorch allocations are outside this limit'}
