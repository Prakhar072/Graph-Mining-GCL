import torch, sys, traceback
print('torch_version=', torch.__version__)
print('cuda_version=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
    except Exception as e:
        print('memory_summary_exception:', e)
        traceback.print_exc(file=sys.stdout)
else:
    print('no CUDA available')
