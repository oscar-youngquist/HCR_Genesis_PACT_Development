"""Select one CUDA-visible device before the worker imports Torch or Genesis."""
import os


def selected_cuda_device(gpu):
    if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
        raise ValueError('GPU must be a nonnegative CUDA device index')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return str(gpu)
    devices = [value.strip() for value in visible.split(',')]
    if gpu >= len(devices) or not devices[gpu] or devices[gpu] == '-1':
        raise ValueError(f'GPU {gpu} is unavailable under CUDA_VISIBLE_DEVICES={visible!r}')
    return devices[gpu]


def worker_device_environment(plan, environment):
    env = dict(environment)
    if 'cuda_visible_device' in plan:
        # Isolate both frameworks to the selected GPU, exposed locally as cuda:0.
        env['CUDA_VISIBLE_DEVICES'] = plan['cuda_visible_device']
    return env
