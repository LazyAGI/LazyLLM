import os

os.environ['LAZYLLM_DEBUG'] = '1'   # higher priority
os.environ['LAZYLLM_LOG_LEVEL'] = 'INFO'
os.environ['LAZYLLM_LOG_DIR'] = '~/.lazyllm'
os.environ['LAZYLLM_LOG_FILE_SIZE'] = "1 MB"

import lazyllm

lazyllm.LOG.critical('critical')
lazyllm.LOG.error('error')
lazyllm.LOG.warning('warning')
lazyllm.LOG.success('success')
lazyllm.LOG.info('info')
lazyllm.LOG.debug('debug')
lazyllm.LOG.trace('trace')

print(lazyllm.LOG.read())

try:
    1 / 0
except ZeroDivisionError:
    lazyllm.LOG.exception("An error occurred")
    
lazyllm.LOG.level("CUSTOM", no=15, color="<yellow>", icon="*")
lazyllm.LOG.log("CUSTOM", "This is a custom log level")

lazyllm.LOG.error('error')
lazyllm.LOG.error('error')