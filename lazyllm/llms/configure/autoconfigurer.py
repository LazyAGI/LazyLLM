from typing import List
from .configuration import (ConfigurationDatabase, HardwareConfiguration,
                            TrainingConfiguration, DeployConfiguration, OutputConfiguration)
from .protocol import TRAINING_RULE_SET, DEPLOY_RULE_SET


class AutoConfigurer(object):
    def __init__(self, database: ConfigurationDatabase):
        assert isinstance(database, ConfigurationDatabase)
        self._database = database

    def query(
        self, hc: HardwareConfiguration, clazz: type[OutputConfiguration]
    ) -> List[OutputConfiguration]:
        assert isinstance(hc, HardwareConfiguration)
        return self._database.query(hc, clazz)


class AutoFinetuneConfigurer(AutoConfigurer):
    def __init__(self, url):
        super().__init__(ConfigurationDatabase(url, TRAINING_RULE_SET))

    def query(self, hc: HardwareConfiguration) -> List[TrainingConfiguration]:
        return super().query(hc, TrainingConfiguration)


class AutoDeployConfigurer(AutoConfigurer):
    def __init__(self, url):
        super().__init__(ConfigurationDatabase(url, DEPLOY_RULE_SET))

    def query(self, hc: HardwareConfiguration) -> List[DeployConfiguration]:
        assert hc.trainable_params == 0, "trainable params must be 0 when inferencing"
        return super().query(hc, DeployConfiguration)
