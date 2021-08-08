import logging
import os
import sys
import time
from collections import Counter
import logging.config

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.config.fileConfig('logging.conf')

senli_logging = None
if senli_logging is None:
    senli_logging = logging.getLogger('senli')
    senli_logging.info("senli used")
