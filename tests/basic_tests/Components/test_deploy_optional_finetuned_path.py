import pytest

import lazyllm


@pytest.mark.parametrize('backend', ['lmdeploy', 'infinity', 'lightllm'])
@pytest.mark.parametrize('path_kind', ['none', 'empty', 'missing', 'weights'])
def test_command_builder_selects_optional_finetuned_path(tmp_path, backend, path_kind):
    base = tmp_path / 'base-model'
    base.mkdir()
    tuned = tmp_path / 'tuned-model'
    tuned.mkdir()
    (tuned / 'model.safetensors').write_bytes(b'path-selection-only')
    paths = {'none': None, 'empty': '', 'missing': str(tmp_path / 'missing'), 'weights': str(tuned)}
    deployment = getattr(lazyllm.deploy, backend)(launcher=lazyllm.launchers.empty(ngpus=0))
    command = deployment.cmd(finetuned_model=paths[path_kind], base_model=str(base))
    rendered = command.cmd()
    assert str(tuned if path_kind == 'weights' else base) in rendered


@pytest.mark.parametrize('backend', ['embedding', 'rerank'])
@pytest.mark.parametrize('path_kind', ['none', 'empty', 'missing', 'weights'])
def test_shared_embedding_model_path_selection(tmp_path, backend, path_kind):
    base = tmp_path / 'base-model'
    base.mkdir()
    tuned = tmp_path / 'tuned-model'
    tuned.mkdir()
    (tuned / 'model.safetensors').write_bytes(b'path-selection-only')
    paths = {'none': None, 'empty': '', 'missing': str(tmp_path / 'missing'), 'weights': str(tuned)}
    deployment = getattr(lazyllm.deploy, backend)(launcher=lazyllm.launchers.empty(ngpus=0))
    assert deployment._get_model_path(paths[path_kind], str(base)) == str(
        tuned if path_kind == 'weights' else base)


@pytest.mark.parametrize('path_kind', ['none', 'empty', 'missing', 'weights'])
def test_mindie_writes_selected_path_to_task_local_config(tmp_path, path_kind):
    import json
    from lazyllm.components.deploy.mindie import Mindie

    home = tmp_path / 'mindie-home'
    conf = home / 'mindie-service' / 'conf'
    conf.mkdir(parents=True)
    base, tuned = tmp_path / 'base-model', tmp_path / 'tuned-model'
    base.mkdir()
    tuned.mkdir()
    (tuned / 'model.safetensors').write_bytes(b'path-selection-only')
    paths = {'none': None, 'empty': '', 'missing': str(tmp_path / 'missing'), 'weights': str(tuned)}
    with lazyllm.config.temp('mindie_home', str(home)):
        deployment = lazyllm.deploy.mindie(launcher=lazyllm.launchers.empty(ngpus=0), port=30001)
        command = deployment.cmd(finetuned_model=paths[path_kind], base_model=str(base))
        config = json.loads((conf / 'config.json').read_text())
        model = config['BackendConfig']['ModelDeployConfig']['ModelConfig'][0]
        assert model['modelWeightPath'] == str(tuned if path_kind == 'weights' else base)
        assert command.cmd() == str(home / 'mindie-service/bin/mindieservice_daemon')
        assert isinstance(deployment, Mindie)
