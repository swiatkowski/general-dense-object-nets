from requests.models import HTTPError

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
        source_files = logging_config['upload_source_files']

        self.api.init('{}/{}'.format(namespace, project))
        self.api.create_experiment(name=experiment, description=description, tags=tags, upload_source_files=source_files)
        self.append_tags_from_config(config)
        self.error_count = 0

    def append_tags_from_config(self, config):
        loss_function_config = config['loss_function']
        loss_name = loss_function_config['name']
        self.api.append_tag([loss_name])

        if 'sampler' in loss_function_config:
            sampler_name = loss_function_config['sampler']['name']
            self.api.append_tag(['{}-sampler'.format(sampler_name)])

    def send_logs(self):
        for metric, x, y, type in self.storage:
            try:
                if type == 'number':
                    self.api.log_metric(metric, x=x, y=y)
                elif type == 'text':
                    self.api.log_text(metric, x=x, y=y)
                elif type == 'image':
                    self.api.log_image(metric, x=x, y=y)
                elif type == 'file':
                    self.api.log_artifact(x)
                else:
                    raise Exception("Metric type '{}' not recognized. Supported types are: [number, text, iamge, file]".format(type))
            except HTTPError as err:
                print(type, metric, x, err)
                self.error_count += 1
                if self.error_count > 100:
                    raise err

        self.clear()

    def exit(self):
        self.api.stop()