import logging
import os
import sys
import time
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from absl import logging as ab_logging

#logging.root.addHandler(logging.StreamHandler())

if "TF_FILE_LOG" in os.environ:
    logging.root.addHandler(logging.FileHandler(os.environ["TF_FILE_LOG"]))
tf_logging = logging.getLogger('tensorflow')
tf_logging.info("1")

if tf_logging.handlers:
    tf_logging.handlers = []


def set_level_debug():
    tf_logging.setLevel(logging.DEBUG)


def reset_root_log_handler():
    logging.root.handlers = logging.root.handlers[:1]


def check(point_name):
    print()
    print(point_name)
    tf_logging.info(point_name)
    print("root logger handler : ", logging.root.handlers)
    print("logging.getLogger().handlers", logging.getLogger().handlers)
    print("tf_logging.handler", tf_logging.handlers)


class MyFormatter(logging.Formatter):
    def prefix(self, record):
        """Returns the absl log prefix for the log record.

        Args:
        record: logging.LogRecord, the record to get prefix for.
        """
        created_tuple = time.localtime(record.created)
        created_microsecond = int(record.created % 1.0 * 1e6)

        critical_prefix = ''
        level = record.levelno

        return '%s %s [%02d:%02d:%02d %s:%d] %s' % (
            logging._levelToName[level],
            record.name,
          created_tuple.tm_hour,
          created_tuple.tm_min,
          created_tuple.tm_sec,
          record.filename,
         record.lineno,
          critical_prefix)

    def format(self, record):
        result = super().format(record)
        result = self.prefix(record) + result
        return result


class TFFilter(logging.Filter):
    excludes = ["Outfeed finished for iteration",
                "TPUPollingThread found TPU",
                "Found small feature",
                "Could not load dynamic library",
                "Cannot dlopen some TensorRT libraries",
                "is deprecated"]

    def filter(self, record):
        for e in self.excludes:
            if e in record.msg:
                return False
        return True


h = ab_logging.get_absl_handler()
h.setFormatter(MyFormatter())
s_handler = logging.StreamHandler(sys.stderr)
logging.getLogger().addHandler(s_handler)




logging.getLogger('oauth2client.transport').setLevel(logging.WARNING)
tf_logging.addFilter(TFFilter())


class CounterFilter(logging.Filter):
    targets = ["Dequeue next", "Enqueue next"]
    counter = Counter()

    def filter(self, record):
        for e in self.targets:
            if e in record.msg:
                self.counter[e] += 1
                record.msg += " ({})".format(self.counter[e])
                return True
        return True


class MuteEnqueueFilter(logging.Filter):
    targets = ["Dequeue next", "Enqueue next"]
    seen = set()

    def filter(self, record):
        for pattern in self.targets:
            if pattern in record.msg:
                if pattern in self.seen:
                    return False
                self.seen.add(pattern)
                return True
        return True