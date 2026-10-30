"""Small CPU fixtures: ordered parallelism, numeric cache reuse and invalidation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
import analysis_runtime as runtime
import exp3_film_oscillation as film
import exp3_representation as rep
from test_exp3_film_oscillation import fixture as film_fixture, config as film_config
from test_exp3_representation import fixture as rep_fixture, config as rep_config, remove_fields


def test_film_and_worker_without_threadpoolctl(tmp_path,monkeypatch):
    film_fixture(tmp_path)
    cfg=film_config(tmp_path)
    cfg.update(advanced_diagnostics=False,unmodulated_comparison=False,physical_comparison=False,
               pca=False,max_plot_windows=0,max_scatter_plots=0,max_windows=1,
               cache_enabled=False,num_workers=1)
    expected=film.run_analysis(cfg)
    # Simulate the notebook environment that lacks this optional dependency,
    # regardless of whether the test interpreter happens to have it installed.
    monkeypatch.setitem(sys.modules,'threadpoolctl',None)
    with pytest.warns(RuntimeWarning,match='without runtime CPU thread limits'):
        actual=film.run_analysis(dict(cfg,output_dir=str(tmp_path/'without_dependency')))
    pd.testing.assert_frame_equal(pd.DataFrame(expected['summary']),pd.DataFrame(actual['summary']))
    monkeypatch.setenv('OMP_NUM_THREADS','1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','1')
    with pytest.warns(RuntimeWarning,match='threadpoolctl is unavailable'):
        runtime.worker_init()
    with runtime._limits:
        np.testing.assert_array_equal(np.eye(2)@np.ones(2),np.ones(2))


def test_optional_thread_limits_preserves_available_limiter(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    calls=[]
    @contextmanager
    def limiter(**kwargs):
        calls.append(kwargs)
        try: yield
        finally: calls.append('restored')
    monkeypatch.setitem(sys.modules,'threadpoolctl',SimpleNamespace(threadpool_limits=limiter))
    with runtime.optional_thread_limits(2,user_api='blas'):
        assert calls==[dict(limits=2,user_api='blas')]
    assert calls[-1]=='restored'


def test_numeric_cache_roundtrip_invalidation_and_corruption(tmp_path):
    cache=runtime.NumericCache(dict(output_dir=str(tmp_path/'out'),cache_enabled=True))
    key,value=cache.lookup('test',(np.arange(3),{'scale':1.}))
    assert value is None
    expected=({'x':np.arange(6).reshape(2,3)},[None,float('nan')],('a',2))
    cache.save(key,expected)
    _,actual=cache.lookup('test',(np.arange(3),{'scale':1.}))
    np.testing.assert_array_equal(actual[0]['x'],expected[0]['x'])
    assert actual[2]==expected[2] and np.isnan(actual[1][1])
    assert cache.lookup('test',(np.arange(3),{'scale':2.}))[1] is None
    key.write_bytes(b'interrupted cache')
    assert cache.lookup('test',(np.arange(3),{'scale':1.}))[1] is None
    assert not runtime.NumericCache(dict(output_dir=str(tmp_path))).enabled


def test_order_and_invalid_worker_count():
    assert [value for _,value in runtime.ordered_work(abs,[-3,-1,-2],2)]==[3,1,2]
    for count in (0,-1,True,1.5):
        with pytest.raises(ValueError,match='num_workers'):
            list(runtime.ordered_work(abs,[-1],count))


def test_film_parallel_and_cached_results_match_serial(tmp_path,monkeypatch):
    film_fixture(tmp_path)
    cfg=film_config(tmp_path)
    cfg.update(advanced_diagnostics=False,unmodulated_comparison=False,physical_comparison=False,
               pca=False,max_plot_windows=0,max_scatter_plots=0,max_windows=2,
               cache_enabled=True,cache_dir=str(tmp_path/'cache'),num_workers=1)
    first=film.run_analysis(cfg)
    parallel=film.run_analysis(dict(cfg,output_dir=str(tmp_path/'parallel'),num_workers=2,cache_enabled=False))
    pd.testing.assert_frame_equal(pd.DataFrame(first['summary']),pd.DataFrame(parallel['summary']),rtol=1e-10,atol=1e-12)
    monkeypatch.setattr(film,'analyze_window',lambda *a: pytest.fail('Cached windows recomputed'))
    monkeypatch.setattr(film,'fit_feature_bases',lambda *a: pytest.fail('Cached basis recomputed'))
    cached=film.run_analysis(dict(cfg,output_dir=str(tmp_path/'cached'),title_font_size=19))
    assert cached['timings']['basis_cache_hits']==1
    assert cached['timings']['window_cache_hits']==len(first['windows'])
    pd.testing.assert_frame_equal(pd.DataFrame(first['summary']),pd.DataFrame(cached['summary']))


def test_representation_parallel_preserves_manifest_and_metrics(tmp_path,monkeypatch):
    rep_fixture(tmp_path/'raw',seeds=(0,))
    cfg=rep_config(tmp_path)
    run=rep.discover(cfg)[0][0]
    cfg.update(recording_roots=[str(run['path'])],evaluation_seeds=[0],num_workers=1)
    monkeypatch.setattr(rep,'build_model',lambda *a: pytest.fail('Unexpected checkpoint'))
    monkeypatch.setattr(rep,'OfflineBard',lambda *a,**kw: pytest.fail('Unexpected BARD'))
    first=rep.run_analysis(cfg)
    parallel=rep.run_analysis(dict(cfg,num_workers=2,output_dir=str(tmp_path/'parallel')))
    assert first['selection']['transitions_sha256']==parallel['selection']['transitions_sha256']
    pd.testing.assert_frame_equal(pd.DataFrame(first['table_summary']),pd.DataFrame(parallel['table_summary']))


def test_bard_cache_bypasses_backend_and_preserves_diagnostics(tmp_path,monkeypatch):
    rep_fixture(tmp_path/'raw',seeds=(0,))
    cfg=rep_config(tmp_path)
    run=rep.discover(cfg)[0][0]
    cfg.update(recording_roots=[str(run['path'])],evaluation_seeds=[0],dynamics=True,
               cache_enabled=True,cache_dir=str(tmp_path/'cache'))
    class Backend:
        def __init__(self,*a,**kw): pass
        def terms_batch(self,rows):
            return [dict(mass_matrix=np.eye(8),bias=np.zeros(8),foot_jacobians=np.zeros((4,6,8)),
                         ee_jacobian=np.eye(6,8),base_jacobian=np.eye(6,8),ee_lever=np.zeros(3),
                         base_lever=np.zeros(3),approximation=['test passive approximation']) for row in rows]
    monkeypatch.setattr(rep,'OfflineBard',Backend)
    first=rep.run_analysis(cfg)
    assert first['prediction_sources'].get('offline:BARD_terms',0)>0
    monkeypatch.setattr(rep,'OfflineBard',lambda *a,**kw: pytest.fail('Cached BARD initialized'))
    second=rep.run_analysis(cfg)
    assert second['prediction_sources'].get('cached:BARD_terms',0)>0
    assert second['dynamics_approximations']==first['dynamics_approximations']
    pd.testing.assert_frame_equal(pd.DataFrame(first['table_summary']),pd.DataFrame(second['table_summary']))


def test_decoder_cache_preserves_stored_fields_without_model_reload(tmp_path,monkeypatch):
    rep_fixture(tmp_path/'raw',seeds=(0,))
    remove_fields(tmp_path/'raw',['prediction_grf'])
    cfg=rep_config(tmp_path)
    run=rep.discover(cfg)[0][0]
    cfg.update(recording_roots=[str(run['path'])],evaluation_seeds=[0],recorded_native_only=False,
               cache_enabled=True,cache_dir=str(tmp_path/'cache'))
    monkeypatch.setattr(rep,'build_model',lambda *a: object())
    def infer(model,rows,device):
        return dict(prediction_grf=np.zeros((len(rows),12)),prediction_ee_force=np.full((len(rows),3),1e10))
    monkeypatch.setattr(rep,'infer',infer)
    first=rep.run_analysis(cfg)
    assert first['prediction_sources']['fallback:prediction_grf']>0
    assert 'fallback:prediction_ee_force' not in first['prediction_sources']
    monkeypatch.setattr(rep,'build_model',lambda *a: pytest.fail('Cached decoder initialized'))
    monkeypatch.setattr(rep,'infer',lambda *a: pytest.fail('Cached inference recomputed'))
    second=rep.run_analysis(cfg)
    assert second['timings']['cache_hits']>0
    pd.testing.assert_frame_equal(pd.DataFrame(first['table_summary']),pd.DataFrame(second['table_summary']))
