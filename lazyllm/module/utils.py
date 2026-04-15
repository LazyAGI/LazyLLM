import os


def light_reduce(cls):
    def rebuild(mid):
        inst = cls()._set_mid(mid)
        # Rebuilt instances are proxies to an already-deployed module on another
        # process; mark the deploy flag as set so that if this proxy needs to be
        # re-pickled downstream (e.g. RPC unpickles args then persists them as
        # info_pickle for worker subprocesses) the `_impl` assertion below is
        # satisfied. Without this, the second pickle raises
        # "TrainableModule shoule be deployed before used" for what is really a
        # remote-handle forwarded through multiple hops.
        flag = getattr(inst, '_lazyllm__get_deploy_tasks_once_flag', None)
        if flag is not None:
            flag.set(True, ignore_reset=True)
        return inst

    def _impl(self):
        if os.getenv('LAZYLLM_ON_CLOUDPICKLE', False) == 'ON':
            assert self._get_deploy_tasks.flag, f'{cls.__name__[1:-4]} shoule be deployed before used'
            return rebuild, (self._module_id,)
        return super(cls, self).__reduce__()
    cls.__reduce__ = _impl
    return cls
