import time

class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

class Timer(object):
    def __init__(self, name="Timer", logger = None):
        self.name = name
        self.logger = logger
        self.start_time = time.time()

    def end(self):
        duration = time.time() - self.start_time
        if self.logger is None:
            print("{} : {}".format(self.name, time.strftime("%H:%M:%S", time.gmtime(duration))))
        else:
            self.logger.info("{} : {}".format(self.name, time.strftime("%H:%M:%S", time.gmtime(duration))))