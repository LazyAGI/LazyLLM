from configuration import (
    ConfigurationDatabase,
    HardwareConfiguration,
    TrainingConfiguration,
    DeployConfiguration,
    OutputConfiguration,
)
import protocol


class AutoConfigurer(object):
    def __init__(self, database: ConfigurationDatabase):
        assert isinstance(database, ConfigurationDatabase)
        self._database = database

    def query(
        self, hc: HardwareConfiguration, clazz: type[OutputConfiguration]
    ) -> list[OutputConfiguration]:
        assert isinstance(hc, HardwareConfiguration)
        return self._database.query(hc, clazz)


class AutoFinetuneConfigurer(AutoConfigurer):
    def __init__(self, url):
        super().__init__(ConfigurationDatabase(url, protocol.TRAINING_RULE_SET))

    def query(self, hc: HardwareConfiguration) -> list[TrainingConfiguration]:
        return super().query(hc, TrainingConfiguration)


class AutoDeployConfigurer(AutoConfigurer):
    def __init__(self, url):
        super().__init__(ConfigurationDatabase(url, protocol.DEPLOY_RULE_SET))

    def query(self, hc: HardwareConfiguration) -> list[DeployConfiguration]:
        assert hc.trainable_params == 0, "trainable params must be 0 when inferencing"
        return super().query(hc, DeployConfiguration)
