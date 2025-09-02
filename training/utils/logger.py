from comet_ml import Experiment as CometExperiment, ExistingExperiment
import tensorflow as tf

class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        
        if not self.config.exp_key:
            self.comet_exp = CometExperiment(api_key=config.api_key, project_name=config.proj_name)
        else:
            self.comet_exp = ExistingExperiment(api_key=config.api_key, previous_experiment=config.exp_key)
            
        self.key = self.comet_exp.get_key()

        self.comet_exp.log_parameters(config)
        self.comet_exp.log_code(folder="./")
        self.comet_exp.set_name(config.exp_description)

    def get_key(self):
        return self.key

    def log_metric(self, key, val, step):
        self.comet_exp.log_metric(key, val, step=step)

    def log_image(self, im, fname, step):
        self.comet_exp.log_image(im, name=fname, step=step)
        
    def log_dict(self, metric_dict, step, postfix=None):
        for key, value in metric_dict.items():
            if postfix is not None:
                key = key + postfix
            if isinstance(value, tf.Tensor) and tf.shape(value) == (1,):
                value = value.numpy()[0]
            if isinstance(value, (int, float)):
                self.comet_exp.log_metric(key, value, step=step)
                


    
