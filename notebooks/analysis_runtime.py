"""Bounded, ordered CPU work and local numeric caches for offline analyses."""
from collections import deque
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
import zipfile
import warnings

import numpy as np


def optional_thread_limits(limits, user_api=None):
    """Thread limiting is an optimization, not a requirement for offline metrics."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        warnings.warn(
            'threadpoolctl is unavailable; continuing without runtime CPU thread limits. '
            'Install it in the notebook kernel with %pip install threadpoolctl to avoid oversubscription.',
            RuntimeWarning, stacklevel=2)
        return nullcontext()
    return threadpool_limits(limits=limits, user_api=user_api)


def worker_init():
    # Spawn avoids inheriting CUDA state; small NumPy operations need one BLAS thread.
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'
    global _limits
    # Environment hints cannot reliably resize BLAS pools already initialized.
    _limits = optional_thread_limits(limits=1)


def ordered_work(function, jobs, workers=1):
    """At most 2*workers jobs in flight; preserve input order and bound RAM."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('num_workers must be a positive integer')
    if workers == 1:
        for job in jobs: yield job, function(job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool:
        pending = deque(); iterator = iter(jobs)
        try:
            for _ in range(2*workers):
                job = next(iterator, None)
                if job is None: break
                pending.append((job, pool.submit(function, job)))
            while pending:
                job, future = pending.popleft()
                yield job, future.result()
                following = next(iterator, None)
                if following is not None: pending.append((following, pool.submit(function, following)))
        finally:
            for _, future in pending: future.cancel()


def data_signature(runs):
    """Recordings are immutable: content metadata plus chunk size/mtime invalidate reuse."""
    signature=[]
    for run in runs:
        paths=[run['path']/'metadata.json',run['path']/'records.json']
        paths += [run['path']/c['file'] for c in run['records']['chunks']]
        signature.append([(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths])
    return signature


@lru_cache(None)
def code_signature():
    root=Path(__file__).parent
    files=['analysis_runtime.py','exp3_film_oscillation.py','exp3_representation.py',
           'exp3_oscillation_physical.py','exp3_support.py','exp1_analysis.py','exp2_analysis.py',
           '../legged_gym/dynamics/bard_b1z1_dynamics.py',
           '../legged_gym/dynamics/whole_body_dynamics.py',
           '../rsl_rl/modules/actor_critic_b1z1_pact.py',
           '../rsl_rl/modules/actor_critic_b1z1_pact_pos.py',
           '../rsl_rl/modules/hard_pact_control.py',
           '../rsl_rl/modules/actor_critic_unifp.py',
           '../rsl_rl/modules/actor_critic_unifp_original.py']
    import scipy
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in files)).hexdigest(),np.__version__,scipy.__version__


def numerical_config(cfg):
    ignored={'output_dir','cache_dir','cache_enabled','num_workers','progress','feature_bases',
             'max_plot_windows','max_scatter_plots','comparison_plot_windows','max_plot_relationships',
             'pca','pca_samples','pca_color','tsne','tsne_perplexity','plot_action_references','trace_components'}
    return {k:v for k,v in cfg.items() if k not in ignored and not k.endswith('_font_size')
            and not k.startswith(('oscillation_','paired_pca_'))}


class NumericCache:
    """Atomic NPZ+JSON caches. Never load pickled objects from disk."""
    def __init__(self,cfg):
        self.enabled=cfg.get('cache_enabled',False)
        self.root=Path(cfg.get('cache_dir') or (Path(cfg['output_dir']).parent/'.exp3_numeric_cache'))
        self.hits=self.misses=0

    def lookup(self,stage,inputs):
        if not self.enabled: return None,None
        # Hash values rather than object identities/aliasing, including after a
        # cached basis has been reloaded from JSON instead of fitted in memory.
        hasher=hashlib.sha256()
        def digest_value(value):
            if isinstance(value,np.generic): value=value.item()
            if isinstance(value,Path): value=str(value)
            if isinstance(value,np.ndarray):
                digest_value(('array',value.dtype.str,value.shape))
                hasher.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value,dict):
                digest_value(('dict',len(value)))
                for k in sorted(value,key=lambda x:(type(x).__name__,repr(x))):
                    digest_value(k); digest_value(value[k])
            elif isinstance(value,(tuple,list)):
                hasher.update((type(value).__name__+str(len(value))+':').encode())
                for item in value: digest_value(item)
            else:
                encoded=json.dumps(value,sort_keys=True).encode()
                hasher.update(str(len(encoded)).encode()+b':'+encoded)
        digest_value((code_signature(),stage,inputs))
        digest=hasher.hexdigest()
        path=self.root/stage/(digest+'.npz')
        if path.exists():
            try:
                with np.load(path,allow_pickle=False) as saved:
                    def decode(node):
                        kind,value=node
                        if kind=='array': return saved[value].copy()
                        if kind=='dict': return {decode(k):decode(v) for k,v in value}
                        if kind in ('list','tuple'):
                            values=[decode(v) for v in value]
                            return tuple(values) if kind=='tuple' else values
                        return value
                    result=decode(json.loads(str(saved['manifest'].item())))
                self.hits+=1
                return path,result
            except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile):
                pass  # Interrupted/corrupt optional caches are recomputed.
        self.misses+=1
        return path,None

    def save(self,path,value):
        if path is None: return
        arrays={}
        def encode(x):
            if isinstance(x,np.ndarray):
                key='array_'+str(len(arrays)); arrays[key]=x
                return ['array',key]
            if isinstance(x,dict): return ['dict',[[encode(k),encode(v)] for k,v in x.items()]]
            if isinstance(x,(list,tuple)): return ['tuple' if isinstance(x,tuple) else 'list',[encode(v) for v in x]]
            if isinstance(x,np.generic): x=x.item()
            return ['scalar',x]
        manifest=json.dumps(encode(value))
        path.parent.mkdir(parents=True,exist_ok=True)
        descriptor,name=tempfile.mkstemp(dir=path.parent,suffix='.npz')
        try:
            with os.fdopen(descriptor,'wb') as stream:
                np.savez_compressed(stream,manifest=np.asarray(manifest),**arrays)
            os.replace(name,path)
        finally:
            if os.path.exists(name): os.unlink(name)


def timed_read_chunk(job):
    filename,path,fields,specs,rows=job
    start=time.perf_counter()
    with np.load(path,allow_pickle=False) as saved: data={k:saved[k] for k in fields}
    for k,v in data.items():
        if v.shape!=(rows,*specs[k]['shape']) or str(v.dtype)!=specs[k]['dtype']:
            raise ValueError(f'Schema mismatch: {path}/{k}')
    return filename,data,time.perf_counter()-start
