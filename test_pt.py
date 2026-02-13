"""
Minimal usage examples for pt_op operators.
Run: python test.py
"""
import os
import tempfile

# Trigger pt_op registration
import lazyllm.tools.data.operators.pt_op  # noqa: F401


def _make_test_image(path, size=(300, 300)):
    from lazyllm.thirdparty import PIL
    img = PIL.Image.new('RGB', size, color='red')
    img.save(path, 'PNG')
    return path


# ---------------------------------------------------------------------------
# 1. resolution_filter
# ---------------------------------------------------------------------------
def test_resolution_filter():
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    _make_test_image(path)
    op = lazyllm.data.pt_mm.resolution_filter()
    data = {'image_path': path}
    res = op([data])
    os.unlink(path)
    print('resolution_filter:', 'OK' if res and res[0].get('image_path') else 'FAIL')
    print('  output:', res)


# ---------------------------------------------------------------------------
# 2. resolution_resize
# ---------------------------------------------------------------------------
def test_resolution_resize():
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    _make_test_image(path, (800, 600))
    op = lazyllm.data.pt_mm.resolution_resize(max_side=400)
    data = {'image_path': path}
    res = op([data])
    os.unlink(path)
    print('resolution_resize:', 'OK' if res and res[0].get('image_path') else 'FAIL')
    print('  output:', res)


# ---------------------------------------------------------------------------
# 3. integrity_check
# ---------------------------------------------------------------------------
def test_integrity_check():
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    _make_test_image(path)
    op = lazyllm.data.pt_mm.integrity_check()
    data = {'image_path': path}
    res = op([data])
    os.unlink(path)
    print('integrity_check:', 'OK' if res and res[0].get('image_path') else 'FAIL')
    print('  output:', res)


# ---------------------------------------------------------------------------
# 4. ImageDedup (batch)
# ---------------------------------------------------------------------------
def test_image_dedup():
    from lazyllm.tools.data.operators.pt_op import ImageDedup
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    _make_test_image(path)
    op = ImageDedup()
    batch = [
        {'image_path': path, 'id': 1},
        {'image_path': path, 'id': 2},  # duplicate
    ]
    res = op(batch)
    os.unlink(path)
    print('ImageDedup:', 'OK' if len(res) == 1 else 'FAIL')
    print('  output:', res)


# ---------------------------------------------------------------------------
# 5. GraphRetriever
# ---------------------------------------------------------------------------
def test_graph_retriever():
    from lazyllm.tools.data.operators.pt_op import GraphRetriever
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img_path = f.name
    _make_test_image(img_path)
    op = GraphRetriever(context_key='context', img_key='img')
    # Markdown format or plain path
    data = {'context': 'Test context with {braces}', 'img': f'![]({img_path})'}
    res = op([data])
    os.unlink(img_path)
    assert res and res[0]['context'] == 'Test context with {{braces}}'
    assert 'img' in res[0] and len(res[0]['img']) == 1
    assert os.path.isabs(res[0]['img'][0])
    print('GraphRetriever:', 'OK')
    print('  output:', res)


# ---------------------------------------------------------------------------
# 6–10. VLM operators (TextRelevanceFilter, VQAGenerator, Phi4QAGenerator,
#        VQAScorer, ContextQualFilter) require a real VLM.
#        Skip or set VLM_URL/VLM_API_KEY to run.
# ---------------------------------------------------------------------------
def test_vlm_operators():
    vlm = None
    try:
        vlm = lazyllm.OnlineChatModule(source='sensenova', model='SenseNova-V6-5-Turbo')
    except Exception:
        pass
    if vlm is None:
        print('VLM operators: SKIP (no VLM available, set API key to test)')
        print('  output: N/A')
        return

    from lazyllm.tools.data.operators.pt_op import (
        TextRelevanceFilter, VQAGenerator, Phi4QAGenerator, VQAScorer, ContextQualFilter
    )
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img_path = f.name
    _make_test_image(img_path)

    # TextRelevanceFilter
    op = TextRelevanceFilter(vlm, threshold=0.5)
    data = {'image_path': img_path, 'text': 'a red square'}
    res = op([data])
    print('TextRelevanceFilter:', 'OK' if res else 'SKIP/FILTERED')
    print('  output:', res)

    # VQAGenerator
    op = VQAGenerator(vlm, num_qa=2)
    data = {'image_path': img_path, 'context': 'A simple red image.'}
    res = op([data])
    print('VQAGenerator:', 'OK' if res and res[0].get('qa_pairs') else 'SKIP')
    print('  output:', res)

    # Phi4QAGenerator
    op = Phi4QAGenerator(vlm, num_qa=2)
    data = {'context': 'A red square.', 'image_path': img_path}
    res = op([data])
    print('Phi4QAGenerator:', 'OK' if res and res[0].get('qa_pairs') else 'SKIP')
    print('  output:', res)

    # VQAScorer
    op = VQAScorer(vlm)
    data = {'image_path': img_path}
    res = op([data])
    print('VQAScorer:', 'OK' if res and res[0].get('quality_score') else 'SKIP')
    print('  output:', res)

    # ContextQualFilter
    op = ContextQualFilter(vlm)
    data = {'context': 'A red image. What color is it?', 'image_path': img_path}
    res = op([data])
    print('ContextQualFilter:', 'OK' if res else 'SKIP/FILTERED')
    print('  output:', res)

    os.unlink(img_path)


if __name__ == '__main__':
    # Need to import lazyllm.data to trigger data module load
    import lazyllm.tools.data  # noqa: F401

    test_resolution_filter()
    test_resolution_resize()
    test_integrity_check()
    test_image_dedup()
    test_vlm_operators()
    test_graph_retriever()