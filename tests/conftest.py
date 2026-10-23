"""Unit/API tests never download weights or run real inference."""
import pytest
from app.config import settings
from app.modules import module_b_spatial_vision as vision


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings,"storage_root",tmp_path)
    monkeypatch.setattr(settings,"uploads_dir",tmp_path/"uploads")
    monkeypatch.setattr(settings,"jobs_dir",tmp_path/"jobs")
    monkeypatch.setattr(settings,"outputs_dir",tmp_path/"outputs")
    monkeypatch.setattr(settings,"visual_verification",False)
    monkeypatch.setattr(settings,"require_visual_verification",False)
    monkeypatch.setattr(settings,"artwork_generator","preserve")
    monkeypatch.setattr(settings,"code_generator","deterministic")
    settings.ensure_directories()
    monkeypatch.setattr(vision,"_run_yolo_cpu",lambda states,*args:[[] for _ in states])
    monkeypatch.setattr(vision,"_run_vision_detector",lambda states,*args:[[] for _ in states])
    monkeypatch.setattr(vision,"_run_vision_verifier",lambda states,dets:(dets,[False for _ in states]))
    monkeypatch.setattr(vision._global_ocr_worker,"analyze",lambda paths,*args:[[] for _ in paths])
