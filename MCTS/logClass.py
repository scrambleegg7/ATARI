# -*- coding: utf-8 -*-



from logging import getLogger, StreamHandler, DEBUG
from logging import Formatter
import logging

class MyHandler(StreamHandler):

    def __init__(self):
        StreamHandler.__init__(self)
        fmt = '%(asctime)s %(filename)-10s %(levelname)-8s: %(message)s'
        fmt_date = '%Y-%m-%dT%T%Z'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)

class myLogger(object):

    def __init__(self):
        self.logger = getLogger(__name__)

        handler = StreamHandler()

        handler.setLevel(DEBUG)

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)

        self.logger.setLevel(DEBUG)

        self.logger.addHandler(handler)


    def __del__(self):
        root = self.logger
        map(root.removeHandler, root.handlers[:])
        map(root.removeFilter, root.filters[:])

    def __exit__(self):
        print("exit")

    def debug(self,instr):
        self.logger.debug(instr)

    def info(self,instr):
        self.logger.info(instr)
    def warning(self,instr):
        self.logger.warning(instr)
