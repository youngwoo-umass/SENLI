import logging
import os
import sys
import time
from collections import Counter
import logging.config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import logging as ab_logging

logging.config.fileConfig('logging.conf')
senli_logging = logging.getLogger('senli')
senli_logging.setLevel(logging.INFO)
senli_logging.info("senli used")

logging.root.info("root")
