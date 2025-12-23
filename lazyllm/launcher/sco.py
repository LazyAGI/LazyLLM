import os
import re
import time
import json
import subprocess
from queue import Queue
from collections import defaultdict

import lazyllm
from lazyllm import final, LOG
from .base import LazyLLMLaunchersBase, Job, Status

lazyllm.config.add('sco.workspace', str, 'your_workspace', 'SCO_WORKSPACE',
                   description='The default SCO workspace to use if no workspace is specified.')
lazyllm.config.add('sco_env_name', str, '', 'SCO_ENV_NAME',
                   description='The default SCO environment name to use if no environment name is specified.')
lazyllm.config.add('sco_keep_record', bool, False, 'SCO_KEEP_RECORD',
                   description='Whether to keep the record of the Sensecore job.')
lazyllm.config.add('sco_resource_type', str, 'N3lS.Ii.I60', 'SCO_RESOURCE_TYPE',
                   description='The default SCO resource type to use if no resource type is specified.')


@final
class ScoLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, launcher, sync=sync)
            # SCO job name must start with a letter
            self.name = 's_flag_' + self._generate_name()
            self.workspace_name = launcher.workspace_name
            self.torchrun = launcher.torchrun
            self.output_hooks = [self.output_hook]

        def output_hook(self, line):
            if not self.ip and 'LAZYLLMIP' in line:
                self.ip = line.split()[-1]

        def _wrap_cmd(self, cmd):
            launcher = self._launcher
            # Assemble the cmd
            sco_cmd = f'srun -p {launcher.partition} --workspace-id {self.workspace_name} ' \
                      f'--job-name={self.name} -f {launcher.framework} ' \
                      f'-r {lazyllm.config["sco_resource_type"]}.{launcher.ngpus} ' \
                      f'-N {launcher.nnode} --priority normal '

            torchrun_cmd = f'python -m torch.distributed.run --nproc_per_node {launcher.nproc} '

            if launcher.nnode == 1:
                # SCO for mpi：supports multiple cards in a single machine
                torchrun_cmd += f'--nnodes {launcher.nnode} --node_rank 0 '
            else:
                # SCO for All Reduce-DDP: support multiple machines and multiple cards
                torchrun_cmd += '--nnodes ${WORLD_SIZE} --node_rank ${RANK} ' \
                                '--master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} '
            pythonpath = os.getenv('PYTHONPATH', '')
            precmd = (f'''export PYTHONPATH={os.getcwd()}:{pythonpath}:$PYTHONPATH '''
                      f'''&& export PATH={os.path.join(os.path.expanduser('~'), '.local/bin')}:$PATH && ''')
            if lazyllm.config['sco_env_name']:
                precmd = f'source activate {lazyllm.config["sco_env_name"]} && ' + precmd
            env_vars = os.environ
            lazyllm_vars = {k: v for k, v in env_vars.items() if k.startswith('LAZYLLM')}
            if lazyllm_vars:
                precmd += ' && '.join(f'export {k}={v}' for k, v in lazyllm_vars.items()) + ' && '
            # For SCO: bash -c 'ifconfig | grep "inet " | awk "{printf \"LAZYLLMIP %s\\n\", \$2}"'
            precmd += '''ifconfig | grep "inet " | awk "{printf \\"LAZYLLMIP %s\\\\n\\", \$2}" &&'''  # noqa W605

            # Delete 'python' in cmd
            if self.torchrun and cmd.strip().startswith('python'):
                cmd = cmd.strip()[6:]
            return f'{sco_cmd} \'{precmd} {torchrun_cmd if self.torchrun else ""} {cmd}\''

        def _get_jobid(self):
            for i in range(5):
                time.sleep(2)  # Wait for cmd to be stably submitted to sco
                try:
                    id_str = subprocess.check_output([
                        'squeue', f'--workspace-id={self.workspace_name}',
                        '-o', 'jobname,jobid']).decode('utf-8')
                except Exception:
                    LOG.warning(f'Failed to capture job_id, retry the {i}-th time.')
                    continue
                pattern = re.compile(rf'{re.escape(self.name)}\s+(\S+)')
                match = pattern.search(id_str)
                if match:
                    self.jobid = match.group(1).strip()
                    break
                else:
                    LOG.warning(f'Failed to capture job_id, retry the {i}-th time.')

        def get_jobip(self):
            if self.ip:
                return self.ip
            else:
                raise RuntimeError('Cannot get IP.', f'JobID: {self.jobid}')

        def _scancel_job(self, cmd, max_retries=3):
            retries = 0
            while retries < max_retries:
                if self.status in (Status.Failed, Status.Cancelled, Status.Done):
                    break
                ps = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8', executable='/bin/bash')
                try:
                    stdout, stderr = ps.communicate(timeout=3)
                    if stdout:
                        LOG.info(stdout)
                        if 'success scancel' in stdout:
                            break
                    if stderr:
                        LOG.error(stderr)
                except subprocess.TimeoutExpired:
                    ps.kill()
                    LOG.warning(f'Command timed out, retrying... (Attempt {retries + 1}/{max_retries})')
                except Exception as e:
                    LOG.error('Try to scancel, but meet: ', e)
                retries += 1
                time.sleep(0.5)
            if retries == max_retries:
                LOG.error(f'Command failed after {max_retries} attempts.')

        def stop(self):
            if self.jobid:
                cmd = f'scancel --workspace-id={self.workspace_name} {self.jobid}'
                if lazyllm.config['sco_keep_record']:
                    LOG.warning(
                        f'`sco_keep_record` is on, not executing scancel. '
                        f'You can now check the logs on the web. '
                        f'To delete by terminal, you can execute: `{cmd}`'
                    )
                else:
                    self._scancel_job(cmd)
                    time.sleep(0.5)  # Avoid the execution of scancel and scontrol too close together.

            n = 0
            while self.status not in (Status.Done, Status.Cancelled, Status.Failed):
                time.sleep(1)
                n += 1
                if n > 25:
                    break

            if self.ps:
                self.ps.terminate()
                self.queue = Queue()
                self.output_thread_event.set()
                self.output_thread.join()

            self.jobid = None

        def wait(self):
            if self.ps:
                self.ps.wait()

        @property
        def status(self):
            if self.jobid:
                try:
                    id_str = subprocess.check_output(['scontrol', f'--workspace-id={self.workspace_name}',
                                                      'show', 'job', str(self.jobid)]).decode('utf-8')
                    id_json = json.loads(id_str)
                    job_state = id_json['state'].strip().lower()
                    if job_state == 'running':
                        return Status.Running
                    elif job_state in ['tbsubmitted', 'suspending']:
                        return Status.TBSubmitted
                    elif job_state in ['waiting', 'init', 'queueing', 'creating',
                                       'restarting', 'recovering', 'starting']:
                        return Status.InQueue
                    elif job_state in ['suspended']:
                        return Status.Cancelled
                    elif job_state == 'succeeded':
                        return Status.Done
                except Exception as e:
                    lazyllm.LOG.error(f'Failed to get job status, reason is {str(e)}')
            return Status.Failed

    def __init__(self, partition=None, workspace_name=lazyllm.config['sco.workspace'],
                 framework='pt', nnode=1, nproc=1, ngpus=1, torchrun=False, sync=True, **kwargs):
        assert nnode >= 1, 'Use at least one node.'
        assert nproc >= 1, 'Start at least one process.'
        assert type(workspace_name) is str, f'"workspace_name" is {workspace_name}. Please set workspace_name.'
        self.partition = partition if partition else lazyllm.config['partition']
        self.workspace_name = workspace_name
        self.framework = framework
        self.nnode = nnode
        self.nproc = nproc
        self.ngpus = ngpus or 1
        self.torchrun = torchrun
        self.sync = sync
        super(__class__, self).__init__()

    def makejob(self, cmd):
        return ScoLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, job) -> None:
        assert isinstance(job, ScoLauncher.Job), 'Sco launcher only support cmd'
        job.start()
        if self.sync:
            while job.status == Status.Running:
                time.sleep(10)
            job.stop()
        return job.return_value
