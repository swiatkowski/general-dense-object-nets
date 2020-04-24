class Logger(object):
    def __init__(self):
        self.storage = []

    def log(self, name, x, y=None, type='number'):
        self.storage.append((name, x, y, type))

    def send_logs(self):
        raise NotImplementedError('Abstract method')

    def clear(self):
        self.storage = []

    def exit(self):
        raise NotImplementedError('Abstract method')
