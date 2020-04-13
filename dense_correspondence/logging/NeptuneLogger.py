import neptune

from dense_correspondence.logging.logger import Logger


class NeptuneLogger(Logger):
    def __init__(self, config):
        Logger.__init__(self)
        self.api = neptune

        logging_config = config['logging']
        namespace = logging_config['namespace']
        project = logging_config['project']
        experiment = logging_config['experiment']
        description = logging_config['description']
        tags = logging_config['tags']

        self.api.init('{}/{}'.format(namespace, project))
        self.api.create_experiment(name=experiment, description=description, tags=tags)

    def send_logs(self):
        for metric, value, type in self.storage:
            if type == 'number':
                self.api.log_metric(metric, value)
            elif type == 'text':
                self.api.log_text(metric, value)
            elif type == 'image':
                self.api.log_image(metric, value)
            elif type == 'artifact':
                self.api.log_artifact(value)
            else:
                raise Exception("Metric type '{}' not recognized. Supported types are: [number, text, iamge, artifact]".format(type))

        self.clear()

    def exit(self):
        self.api.stop()