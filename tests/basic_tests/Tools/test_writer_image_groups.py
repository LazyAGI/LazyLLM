from bs4 import BeautifulSoup
import pytest
from lazyllm.tools.writer.adapter.wechat import WeChatWriterAdapter
from lazyllm.tools.writer.data_models.revision import PatchSet
from lazyllm.tools.writer.data_models.multimodal import MediaAsset, MediaAssetLibrary
from lazyllm.tools.writer.tools.revision_tools import apply_patch_to_ir, WriterRevisionTools
from lazyllm.tools.writer.utils.conversion import writer_document_from_markdown, render_document_markdown
from lazyllm.tools.writer.utils.serialization import parse_document_markdown
from lazyllm.tools.writer.utils.image_groups import (
    image_group_markdown, sliding_image_groups, split_image_group_markdown,
)
from lazyllm.tools.writer.utils.export import export_writer_document


def test_image_groups_markdown_patch_and_wechat_round_trip():
    markdown = ('# City greenery\n\n## Park\n\n![Path](https://example.com/a.jpg)\n\n'
                '![Bench](https://example.com/b.jpg)\n\n![Lawn](https://example.com/c.jpg)\n\n'
                'Body\n\n![Single](https://example.com/d.jpg)')
    source = parse_document_markdown(markdown, 'gallery')
    assert export_writer_document(source, 'markdown').count('![') == 4
    images = [block for block in source.iter_blocks() if block.type == 'image']
    for image, name in zip(images, 'abcd'):
        image.references = [{'type': 'media_asset', 'id': name, 'url': f'https://example.com/{name}.jpg'}]
    groups = [{'images': [{'source': f'https://example.com/{name}.jpg', 'occurrence': 0}
                          for name in 'abc'], 'layout': 'sliding'}]
    selected, result = apply_patch_to_ir(source, PatchSet(target_doc_id=source.document_id, image_groups=groups))
    assert result.success
    assert selected.blocks == source.blocks
    assert sliding_image_groups(selected) == {images[0].node_id: [image.node_id for image in images[:3]]}
    serialized = image_group_markdown(markdown, selected)
    assert parse_document_markdown(serialized, 'gallery').metadata['image_groups'] == groups
    common = writer_document_from_markdown(serialized)
    assert common.metadata['image_groups'] == groups
    assert '<!-- writer-image-groups:' in render_document_markdown(common)
    exported = export_writer_document(common, 'markdown')
    assert 'writer-image-groups:' not in exported
    assert exported.count('![') == 4
    adapter = WeChatWriterAdapter()
    urls = {name: f'https://example.com/{name}.jpg' for name in 'abcd'}
    html = adapter.document_to_html(selected, urls)
    soup = BeautifulSoup(html, 'html.parser')
    assert [image['src'] for image in soup.find_all('img')] == list(urls.values())
    assert len(soup.select('section[style*="overflow-x:auto"]')) == 1
    gallery = soup.select_one('section[style*="overflow-x:auto"]')
    slides = gallery.find_all('section', recursive=False)
    assert len(slides) == 3
    assert all('width:auto' in slide['style'] for slide in slides)
    assert all('max-width:100%' in slide['style'] for slide in slides)
    assert 'margin-right:10px' in slides[0]['style']
    assert 'margin-right:0px' in slides[-1]['style']
    assert all('height:auto!important' in slide.img['style'] for slide in slides)
    assert all('width:auto!important' in slide.img['style'] for slide in slides)
    assert all('max-width:100%!important' in slide.img['style'] for slide in slides)
    assert all('object-fit:contain' in slide.img['style'] for slide in slides)
    assert all('border-radius:8px' in slide.img['style'] for slide in slides)
    assert all('font-size:13px' in slide.p['style'] for slide in slides)
    assert all(slide.img.find_next_sibling('p') == slide.p for slide in slides)
    assert all('width:0;min-width:100%' in slide.p['style'] for slide in slides)
    assert all('text-align:center' in slide.p['style'] for slide in slides)
    assert 'aspect-ratio' not in soup.find_all('img')[-1].get('style', '')
    hint = soup.select_one('.writer-image-group-hint').get_text()
    assert '3' in hint
    remote = adapter.html_to_ir(html, external_document_id='remote')
    assert len([block for block in remote.iter_blocks() if block.type == 'image']) == 4
    assert len(sliding_image_groups(remote)) == 1
    assert [block.content for block in remote.iter_blocks() if block.type == 'image'] == [
        'Path', 'Bench', 'Lawn', 'Single',
    ]
    assert all(hint not in block.content for block in remote.iter_blocks())
    assert adapter.document_to_html(remote) == html
    portable = export_writer_document(remote, 'markdown')
    assert portable.count('![') == 4
    assert [portable.index(url) for url in urls.values()] == sorted(portable.index(url) for url in urls.values())
    vertical = remote.model_copy(deep=True)
    vertical.metadata['image_groups'][0]['layout'] = 'vertical'
    restored = adapter.document_to_html(vertical)
    assert 'overflow-x:auto' not in restored
    assert 'aspect-ratio' not in restored
    assert 'writer-image-group-hint' not in restored
    assert len(BeautifulSoup(restored, 'html.parser').find_all('img')) == 4
    patch = WriterRevisionTools(llm=None)._diff_documents(source, selected)
    assert patch.image_groups == groups
    structured = BeautifulSoup(adapter.document_to_html(selected, urls, template='structured'), 'html.parser')
    assert all('object-fit:contain' in image['style']
               for image in structured.select('section[style*="overflow-x:auto"] img'))
    imported = adapter.html_to_ir(str(structured), external_document_id='structured-gallery')
    assert len(sliding_image_groups(imported)) == 1
    assert [block.content for block in imported.iter_blocks() if block.type == 'image'] == [
        'Path', 'Bench', 'Lawn', 'Single',
    ]


def test_wechat_keeps_unmarked_scrollable_content():
    html = ('<section style="padding:24px"><section style="overflow-x:auto;white-space:nowrap">'
            '<section><img src="https://example.com/a.jpg" /></section>'
            '<section><img src="https://example.com/b.jpg" /></section></section></section>')
    adapter = WeChatWriterAdapter()
    document = adapter.html_to_ir(html, external_document_id='scrollable-content')
    assert not sliding_image_groups(document)
    assert adapter.document_to_html(document) == html


def test_pasted_generated_image_urls_resolve_existing_assets():
    path = '/data/subagent/session/workflow/media/assets/flower.jpg'
    library = MediaAssetLibrary(library_id='media', assets={
        'flower': MediaAsset(media_asset_id='flower', asset_type='image',
                             source_type='image_generation', local_path=path),
    })
    document = parse_document_markdown(f'![Flowers](http://localhost:8090{path})', 'pasted', media_assets=library)
    assert document.blocks[0].references[0]['id'] == 'flower'
    assert document.blocks[0].references[0]['path'] == f'http://localhost:8090{path}'


def test_image_group_comments_preserve_source_and_escape_image_urls():
    body = ('---\r\ntitle: Example\r\n---\r\n\r\n```html\r\n'
            '<!-- writer-image-groups: invalid -->\r\n```\r\n\r\n<!-- ordinary comment -->\r\n')
    groups = [{'images': [{'source': 'a-->b.png', 'occurrence': 0}, {'source': 'c.png', 'occurrence': 0}],
               'layout': 'sliding'}]
    document = parse_document_markdown('# Article', 'comments')
    document.metadata['image_groups'] = groups
    saved = image_group_markdown(body, document)
    assert split_image_group_markdown(body) == (body, [])
    content, parsed = split_image_group_markdown(saved)
    assert content.rstrip() == body.rstrip()
    assert parsed == groups
    assert 'a--\\u003eb.png' in saved
    assert image_group_markdown(saved, document) == saved
    assert split_image_group_markdown(saved.replace('\r\n', '\n').replace('\n', '\r\n'))[1] == groups
    with pytest.raises(ValueError):
        split_image_group_markdown('<!-- writer-image-groups: invalid -->')
    with pytest.raises(ValueError):
        split_image_group_markdown('<!-- writer-image-groups: {} -->')
