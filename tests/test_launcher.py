from lazyllm import launchers

class TestFn_Launcher(object):
    
    def test_slurm(self):
        launcher = launchers.slurm(
            partition='pat_rd',
            nnode=1,
            nproc=1,
            ngpus=1,
            sync=False
            )
        assert launcher.partition == 'pat_rd'

    def test_empty(self):
        launcher = launchers.empty()
        assert launcher.subprocess == False

    def test_sco(self):
        launcher = launchers.sco(
            partition='pat_rd',
            nnode=1,
            nproc=1,
            ngpus=1,
            sync=False
        )
        assert launcher.partition == 'pat_rd'

    def test_remote(self):
        launcher = launchers.remote(
            partition='pat_rd',
            nnode=1,
            nproc=1,
            ngpus=1,
            sync=False
        )
        assert launcher.partition == 'pat_rd'
