import pytest
from liquid_collection.device import selected_cuda_device, worker_device_environment


def test_device_selection_and_worker_isolation(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    assert selected_cuda_device(2) == '2'
    parent = {'CUDA_VISIBLE_DEVICES': '3,7', 'OTHER': 'preserved'}
    env = worker_device_environment({'cuda_visible_device': '7'}, parent)
    assert env == {'CUDA_VISIBLE_DEVICES': '7', 'OTHER': 'preserved'}
    assert parent['CUDA_VISIBLE_DEVICES'] == '3,7'
    assert worker_device_environment({}, parent) == parent
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-abc,GPU-def')
    assert selected_cuda_device(1) == 'GPU-def'
    with pytest.raises(ValueError): selected_cuda_device(2)


@pytest.mark.parametrize('visibility', ['', '-1'])
def test_disabled_cuda_rejected(monkeypatch, visibility):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visibility)
    with pytest.raises(ValueError): selected_cuda_device(0)


def test_campaign_propagates_selection(monkeypatch):
    from liquid_collection.launch import campaign_plan
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,7')
    campaign = campaign_plan(['go1-dact', 'go2-dact'], 2, 1, 1, 6., gpu=1)
    assert campaign['gpu'] == 1 and campaign['cuda_visible_device'] == '7'
    for plan in campaign['plans'].values():
        assert plan['gpu'] == 1 and plan['cuda_visible_device'] == '7'
        assert plan['save_mode'] == '200hz'
    with pytest.raises(ValueError): campaign_plan(['go1-dact'], 2, 1, 1, 6., gpu=-1)
