<<<<<<< HEAD
class Logger(object):
=======
class Logger:
>>>>>>> Add abstract and concrete logging integration
    def __init__(self):
        self.storage = []

    def log(self, name, value, type='number'):
        self.storage.append((name, value, type))

    def send_logs(self):
        raise NotImplementedError('Abstract method')

    def clear(self):
        self.storage = []

    def exit(self):
        raise NotImplementedError('Abstract method')
