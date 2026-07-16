"""
Tests for the VRAM lifecycle manager. These run without a GPU: when
CUDA isn't available, _snapshot() returns None and the budget check is
skipped, so we're only verifying load/unload *sequencing* discipline
here, not real VRAM numbers.
"""
import pytest

from app.core.vram_manager import GPUPipelineGuard, vram_scope


class FakeModel:
    def __init__(self):
        self.closed = False


def test_vram_scope_loads_and_unloads():
    load_calls = []
    unload_calls = []

    def loader():
        load_calls.append(1)
        return FakeModel()

    def unloader(model):
        model.closed = True
        unload_calls.append(1)

    with vram_scope("fake", loader=loader, unloader=unloader) as model:
        assert isinstance(model, FakeModel)
        assert model.closed is False

    assert len(load_calls) == 1
    assert len(unload_calls) == 1


def test_vram_scope_unloads_even_on_exception():
    unload_calls = []

    def loader():
        return FakeModel()

    def unloader(model):
        unload_calls.append(1)

    with pytest.raises(ValueError):
        with vram_scope("fake", loader=loader, unloader=unloader):
            raise ValueError("inference blew up")

    assert len(unload_calls) == 1, "model must be unloaded even when inference raises"


def test_gpu_pipeline_guard_sequential_stages_ok():
    guard = GPUPipelineGuard()

    def loader():
        return FakeModel()

    with guard.stage("stage-a", loader=loader):
        pass

    with guard.stage("stage-b", loader=loader):
        pass  # no RuntimeError => the guard can be reused for the next stage


def test_gpu_pipeline_guard_blocks_reentrant_stage():
    guard = GPUPipelineGuard()

    def loader():
        return FakeModel()

    with guard.stage("outer", loader=loader):
        with pytest.raises(RuntimeError):
            with guard.stage("inner", loader=loader):
                pass
