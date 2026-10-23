import json
from tools import setup_project
from tools.runtime_preflight import weights_present
from app.modules.component_runtime import MEASURED_COMPONENTS
from app.modules.coder_runtime import validate_source
from app.core.model_budget import configure_coder,CODER_MODEL,SUPPORTED_MODEL
import pytest


def test_setup_downloads_current_stack_and_retains_other_configuration(tmp_path):
    path=tmp_path/'.env'
    path.write_text('OMNIUI_VISION_DETECTOR_MODEL=Qwen/Qwen2.5-VL-3B-Instruct\nOMNIUI_MAX_UPLOAD_MB=321\n')
    setup_project.migrate_config(tmp_path)
    content=path.read_text()
    assert SUPPORTED_MODEL in content and CODER_MODEL in content
    assert 'OMNIUI_MAX_UPLOAD_MB=321' in content
    assert '3B-Instruct' not in content and 'ltx-2b-fp16' in content
    setup_project.migrate_config(tmp_path)
    assert path.read_text()==content
    plan=setup_project.commands('project-python')
    assert any('setup_ltx.py' in ' '.join(command) and command[-1]=='fp16' for command in plan)
    assert not any('3B-Instruct' in ' '.join(command) for command in plan)


def test_partial_weight_snapshot_is_rejected(tmp_path):
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'first.safetensors','b':'second.safetensors'}}))
    (tmp_path/'first.safetensors').write_bytes(b'x'*2048)
    assert not weights_present(tmp_path)
    (tmp_path/'second.safetensors').write_bytes(b'x'*2048)
    assert weights_present(tmp_path)


def test_coder_rejects_backend_calls_before_building():
    with pytest.raises(ValueError,match='outside the local UI'):
        validate_source(MEASURED_COMPONENTS+'\nfetch("/scan");')


def test_coder_rejects_lost_native_submission():
    with pytest.raises(ValueError,match='contract'):
        validate_source(MEASURED_COMPONENTS.replace('omniui:submit','scan:start'))
