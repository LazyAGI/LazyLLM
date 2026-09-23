from importlib.resources import files


_TEMPLATES = {
    'zh-CN': 'basic_zh.tex',
    'en-US': 'basic_en.tex',
}


def load_latex_template(language: str) -> str:
    '''Load a bundled basic LaTeX template by locale.'''
    try:
        name = _TEMPLATES[language]
    except KeyError as exc:
        raise ValueError(f'Unsupported LaTeX template language: {language!r}.') from exc
    return files(__package__).joinpath(name).read_text(encoding='utf-8')


def latex_template_resource(language: str):
    '''Return the bundled template resource for an internally selected locale.'''
    try:
        name = _TEMPLATES[language]
    except KeyError as exc:
        raise ValueError(f'Unsupported LaTeX template language: {language!r}.') from exc
    return files(__package__).joinpath(name)


def writer_latex_filter_resource():
    '''Return the bundled filter implementing the LazyMind Markdown contract.'''
    return files(__package__).joinpath('filters', 'writer.lua')


__all__ = ['load_latex_template']
