import os
import re
import time
import json
import random
import atexit
import threading
import subprocess
from enum import Enum
from queue import Queue
from datetime import datetime
from multiprocessing.util import register_after_fork
from collections import defaultdict
import uuid
import copy
import psutil

import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final, timeout, LOG

class Status(Enum):
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Pending = 3,
    Done = 100,
    Cancelled = 101,  # TODO(wangzhihong): support cancel job
    Failed = 102,


class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    Status = Status

    def __init__(self) -> None:
        self._id = str(uuid.uuid4().hex)

    def makejob(self, cmd):
        raise NotImplementedError

    def launch(self, *args, **kw):
        raise NotImplementedError

    def cleanup(self):
        for k, v in self.all_processes[self._id]:
            v.stop()
            LOG.info(f"killed job:{k}")
        self.all_processes.pop(self._id)
        self.wait()

    @property
    def status(self):
        if len(self.all_processes[self._id]) == 1:
            return self.all_processes[self._id][0][1].status
        elif len(self.all_processes[self._id]) == 0:
            return Status.Cancelled
        raise RuntimeError('More than one tasks are found in one launcher!')

    def wait(self):
        for _, v in self.all_processes[self._id]:
            v.wait()

    def clone(self):
        new = copy.deepcopy(self)
        new._id = str(uuid.uuid4().hex)
        return new


lazyllm.launchers['Status'] = Status

lazyllm.config.add('launcher', str, 'empty', 'DEFAULT_LAUNCHER')
lazyllm.config.add('partition', str, 'your_part', 'SLURM_PART')
lazyllm.config.add('sco.workspace', str, 'your_workspace', 'SCO_WORKSPACE')
lazyllm.config.add('sco_env_name', str, '', 'SCO_ENV_NAME')
lazyllm.config.add('sco_keep_record', bool, False, 'SCO_KEEP_RECORD')
lazyllm.config.add('sco_resource_type', str, 'N3lS.Ii.I60', 'SCO_RESOURCE_TYPE')
lazyllm.config.add('cuda_visible', bool, False, 'CUDA_VISIBLE')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, launcher, *, sync=True):
        assert isinstance(cmd, LazyLLMCMD)
        self._origin_cmd = cmd
        self.sync = sync
        self.launcher = launcher
        self.queue, self.jobid, self.ip, self.ps = Queue(), None, None, None
        self.output_hooks = []

    def _set_return_value(self):
        cmd = getattr(self, '_fixed_cmd', None)
        if cmd and callable(cmd.return_value):
            self.return_value = cmd.return_value(self)
        elif cmd and cmd.return_value:
            self.return_value = cmd.return_value
        else:
            self.return_value = self

    def get_executable_cmd(self, *, fixed=False):
        if fixed and hasattr(self, '_fixed_cmd'):
            LOG.info('Command is fixed!')
            return self._fixed_cmd
        cmd = self._origin_cmd
        if callable(cmd.cmd):
            cmd = cmd.with_cmd(cmd.cmd())
        self._fixed_cmd = cmd.with_cmd(self._wrap_cmd(cmd.cmd))
        return self._fixed_cmd

    # interfaces
    def stop(self): raise NotImplementedError
    @property
    def status(self): raise NotImplementedError
    def wait(self): pass
    def _wrap_cmd(self, cmd): return cmd

    def _start(self, *, fixed):
        cmd = self.get_executable_cmd(fixed=fixed)
        LOG.info(f'Command: {cmd}')
        if lazyllm.config['mode'] == lazyllm.Mode.Display: return
        self.ps = subprocess.Popen(cmd.cmd, shell=True, executable='/bin/bash',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._get_jobid()
        self._enqueue_subprocess_output(hooks=self.output_hooks)

        if self.sync:
            self.ps.wait()
        else:
            with timeout(3600, msg='Launch failed: No computing resources are available.'):
                while self.status in (Status.TBSubmitted, Status.InQueue, Status.Pending):
                    time.sleep(2)
            self.launcher.all_processes[self.launcher._id].append((self.jobid, self))

    def restart(self, *, fixed=False):
        self.stop()
        time.sleep(2)
        self._start(fixed=fixed)

    def start(self, *, restart=3, fixed=False):
        self._start(fixed=fixed)
        if not (lazyllm.config['mode'] == lazyllm.Mode.Display or self._fixed_cmd.checkf(self)):
            if restart > 0:
                for _ in range(restart):
                    self.restart(fixed=fixed)
                    if self._fixed_cmd.checkf(self): break
                else:
                    raise RuntimeError(f'Job failed after retrying {restart} times')
            else:
                raise RuntimeError('Job failed without retries')
        self._set_return_value()

    def _enqueue_subprocess_output(self, hooks=None):
        self.output_thread_event = threading.Event()

        def impl(out, queue):
            for line in iter(out.readline, b''):
                try:
                    line = line.decode('utf-8')
                except Exception:
                    pass
                queue.put(line)
                if hooks:
                    hooks(line) if callable(hooks) else [hook(line) for hook in hooks]
                LOG.info(f'{self.jobid}: {line.rstrip()}', )
                if self.output_thread_event.is_set():
                    break
            out.close()
        self.output_thread = threading.Thread(target=impl, args=(self.ps.stdout, self.queue))
        self.output_thread.daemon = True
        self.output_thread.start()

    def _generate_name(self):
        now = datetime.now()
        return str(hex(hash(now.strftime("%S%M") + str(random.randint(3, 2000)))))[2:10]

    def __deepcopy__(self, memo=None):
        raise RuntimeError('Cannot copy Job object')

@final
class EmptyLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, launcher, sync=sync)

        def _wrap_cmd(self, cmd):
            if self.launcher.ngpus == 0:
                return cmd
            gpus = self.launcher._get_idle_gpus()
            if gpus and lazyllm.config['cuda_visible']:
                if self.launcher.ngpus is None:
                    empty_cmd = f'CUDA_VISIBLE_DEVICES={gpus[0]} '
                elif self.launcher.ngpus <= len(gpus):
                    empty_cmd = 'CUDA_VISIBLE_DEVICES=' + \
                                ','.join([str(n) for n in gpus[:self.launcher.ngpus]]) + ' '
                else:
                    error_info = (f'Not enough GPUs available. Requested {self.launcher.ngpus} GPUs, '
                                  f'but only {len(gpus)} are available.')
                    LOG.error(error_info)
                    raise error_info
            else:
                empty_cmd = ''
            return empty_cmd + cmd

        def stop(self):
            if self.ps:
                try:
                    parent = psutil.Process(self.ps.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    LOG.warning(f"Process with PID {self.ps.pid} does not exist.")
                except psutil.AccessDenied:
                    LOG.warning(f"Permission denied when trying to kill process with PID {self.ps.pid}.")
                except Exception as e:
                    LOG.warning(f"An error occurred: {e}")

        @property
        def status(self):
            return_code = self.ps.poll()
            if return_code is None: job_status = Status.Running
            elif return_code == 0: job_status = Status.Done
            else: job_status = Status.Failed
            return job_status

        def _get_jobid(self):
            self.jobid = self.ps.pid if self.ps else None

        def get_jobip(self):
            return '0.0.0.0'

        def wait(self):
            if self.ps:
                self.ps.wait()

    def __init__(self, subprocess=False, ngpus=None, sync=True):
        super().__init__()
        self.subprocess = subprocess
        self.sync = sync
        self.ngpus = ngpus

    def makejob(self, cmd):
        return EmptyLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, f, *args, **kw):
        if isinstance(f, EmptyLauncher.Job):
            f.start()
            return f.return_value
        elif callable(f):
            if not self.subprocess:
                return f(*args, **kw)
            else:
                LOG.info("Async execution of callable object is not supported currently.")
                import multiprocessing
                p = multiprocessing.Process(target=f, args=args, kwargs=kw)
                p.start()
                p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')

    def _get_idle_gpus(self):
        try:
            order_list = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
        except Exception as e:
            LOG.warning(f"Get idle gpus failed: {e}, if you have no gpu-driver, ignor it.")
            return []
        lines = order_list.strip().split('\n')

        str_num = os.getenv('CUDA_VISIBLE_DEVICES', None)
        if str_num:
            sub_gpus = [int(x) for x in str_num.strip().split(',')]

        gpu_info = []
        for line in lines:
            index, memory_free = line.split(', ')
            if not str_num or int(index) in sub_gpus:
                gpu_info.append((int(index), int(memory_free)))
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        LOG.info('Memory left:\n' + '\n'.join([f'{item[0]} GPU, left: {item[1]} MiB' for item in gpu_info]))
        return [info[0] for info in gpu_info]

@final
class SlurmLauncher(LazyLLMLaunchersBase):
    # In order to obtain the jobid to monitor and terminate the job more
    # conveniently, only one srun command is allowed in one Job
    all_processes = defaultdict(list)
    count = 0

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True, **kw):
            super(__class__, self).__init__(cmd, launcher, sync=sync)
            self.name = self._generate_name()

        def _wrap_cmd(self, cmd):
            # Assemble the order
            slurm_cmd = f'srun -p {self.launcher.partition} -N {self.launcher.nnode} --job-name={self.name}'
            if self.launcher.nproc:
                slurm_cmd += f' -n{self.launcher.nproc}'
            if self.launcher.timeout:
                slurm_cmd += f' -t {self.launcher.timeout}'
            if self.launcher.ngpus:
                slurm_cmd += f' --gres=gpu:{self.launcher.ngpus}'
            return f'{slurm_cmd} bash -c \'{cmd}\''

        def _get_jobid(self):
            time.sleep(0.5)  # Wait for cmd to be stably submitted to slurm
            id_str = subprocess.check_output(['squeue', '--name=' + self.name, '--noheader'])
            if id_str:
                id_list = id_str.decode().strip().split()
                self.jobid = id_list[0]

        def get_jobip(self):
            id_str = subprocess.check_output(['squeue', '--name=' + self.name, '--noheader'])
            id_list = id_str.decode().strip().split()
            self.ip = id_list[10]
            return self.ip

        def stop(self):
            if self.jobid:
                cmd = f"scancel --quiet {self.jobid}"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 encoding='utf-8', executable='/bin/bash')
                self.jobid = None

            if self.ps:
                self.ps.terminate()
                self.queue = Queue()
                self.output_thread_event.set()
                self.output_thread.join()

        def wait(self):
            if self.ps:
                self.ps.wait()

        @property
        def status(self):
            # lookup job
            if self.jobid:
                jobinfo = subprocess.check_output(["scontrol", "show", "job", str(self.jobid)])
                job_state = None
                job_state = None
                for line in jobinfo.decode().split("\n"):
                    if "JobState" in line:
                        job_state = line.strip().split()[0].split("=")[1].strip().lower()
                        if job_state == 'running':
                            return Status.Running
                        elif job_state == 'tbsubmitted':
                            return Status.TBSubmitted
                        elif job_state == 'inqueue':
                            return Status.InQueue
                        elif job_state == 'pending':
                            return Status.Pending
                        elif job_state == 'done':
                            return Status.Done
                        elif job_state == 'cancelled':
                            return Status.Cancelled
                        else:
                            return Status.Failed
            else:
                return Status.Failed

    # TODO(wangzhihong): support configs; None -> lookup config
    def __init__(self, partition=None, nnode=1, nproc=1, ngpus=None, timeout=None, *, sync=True, **kwargs):
        super(__class__, self).__init__()
        # TODO: global config
        self.partition = partition if partition else lazyllm.config['partition']
        self.nnode, self.nproc, self.ngpus, self.timeout = nnode, nproc, ngpus, timeout
        self.sync = sync
        self.num_can_use_nodes = kwargs.get('num_can_use_nodes', 5)

    def makejob(self, cmd):
        return SlurmLauncher.Job(cmd, launcher=self, sync=self.sync)

    def _add_dict(self, node_ip, used_gpus, node_dict):
        if node_ip not in node_dict:
            node_dict[node_ip] = 8 - used_gpus
        else:
            node_dict[node_ip] -= used_gpus

    def _expand_nodelist(self, nodes_str):
        pattern = r'\[(.*?)\]'
        matches = re.search(pattern, nodes_str)
        result = []
        if matches:
            nums = matches.group(1).split(',')
            base = nodes_str.split('[')[0]
            result = [base + str(x) for x in nums]
        return result

    def get_idle_nodes(self, partion=None):
        '''
        Obtain the current number of available nodes based on the available number of GPUs.
        Return a dictionary with node IP as the key and the number of available GPUs as the value.
        '''
        if not partion:
            partion = self.partition
        num_can_use_nodes = self.num_can_use_nodes

        # Query the number of available GPUs for applied nodes
        nodesinfo = subprocess.check_output(["squeue", "-p", partion, '--noheader'])
        node_dict = dict()

        for line in nodesinfo.decode().split("\n"):
            if "gpu:" in line:
                node_info = line.strip().split()
                num_nodes = int(node_info[-3])
                num_gpus = int(node_info[-2].split(":")[-1])
                node_list = node_info[-1]
                if num_nodes == 1:
                    self._add_dict(node_list, num_gpus, node_dict)
                else:
                    avg_gpus = int(num_gpus / num_nodes)
                    result = self._expand_nodelist(node_list)
                    for x in result:
                        self._add_dict(x, avg_gpus, node_dict)

        # Obtain all available idle nodes in the specified partition
        idle_nodes = []
        nodesinfo = subprocess.check_output(["sinfo", "-p", partion, '--noheader'])
        for line in nodesinfo.decode().split("\n"):
            if "idle" in line:
                node_info = line.strip().split()
                num_nodes = int(node_info[-3])
                node_list = node_info[-1]
                if num_nodes == 1:
                    idle_nodes.append(node_list)
                else:
                    idle_nodes += self._expand_nodelist(node_list)

        # Add idle nodes under resource constraints
        num_allocated_nodes = len(node_dict)
        num_append_nodes = num_can_use_nodes - num_allocated_nodes

        for i, node_ip in enumerate(idle_nodes):
            if i + 1 <= num_append_nodes:
                node_dict[node_ip] = 8

        # Remove nodes with depleted GPUs
        node_dict = {k: v for k, v in node_dict.items() if v != 0}
        return node_dict

    def launch(self, job) -> None:
        assert isinstance(job, SlurmLauncher.Job), 'Slurm launcher only support cmd'
        job.start()
        if self.sync:
            while job.status == Status.Running:
                time.sleep(10)
            job.stop()
        return job.return_value


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
            launcher = self.launcher
            # Assemble the cmd
            sco_cmd = f'srun -p {launcher.partition} --workspace-id {self.workspace_name} ' \
                      f'--job-name={self.name} -f {launcher.framework} ' \
                      f'-r {lazyllm.config["sco_resource_type"]}.{launcher.ngpus} ' \
                      f'-N {launcher.nnode} --priority normal '

            torchrun_cmd = f'python -m torch.distributed.run --nproc_per_node {launcher.nproc} '

            if launcher.nnode == 1:
                # SCO for mpi：supports multiple cards in a single machine
                sco_cmd += '-m '
                torchrun_cmd += f'--nnodes {launcher.nnode} --node_rank 0 '
            else:
                # SCO for All Reduce-DDP: support multiple machines and multiple cards
                sco_cmd += '-d AllReduce '
                torchrun_cmd += '--nnodes ${WORLD_SIZE} --node_rank ${RANK} ' \
                                '--master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} '
            pythonpath = os.getenv('PYTHONPATH', '')
            precmd = (f'''export PYTHONPATH={os.getcwd()}:{pythonpath}:$PYTHONPATH '''
                      f'''&& export PATH={os.path.join(os.path.expanduser('~'), '.local/bin')}:$PATH && ''')
            if lazyllm.config['sco_env_name']:
                precmd = f"source activate {lazyllm.config['sco_env_name']} && " + precmd
            env_vars = os.environ
            lazyllm_vars = {k: v for k, v in env_vars.items() if k.startswith("LAZYLLM")}
            if lazyllm_vars:
                precmd += " && ".join(f"export {k}={v}" for k, v in lazyllm_vars.items()) + " && "
            # For SCO: bash -c 'ifconfig | grep "inet " | awk "{printf \"LAZYLLMIP %s\\n\", \$2}"'
            precmd += '''ifconfig | grep "inet " | awk "{printf \\"LAZYLLMIP %s\\\\n\\", \$2}" &&'''  # noqa W605

            # Delete 'python' in cmd
            if self.torchrun and cmd.strip().startswith('python'):
                cmd = cmd.strip()[6:]
            return f'{sco_cmd} bash -c \'{precmd} {torchrun_cmd if self.torchrun else ""} {cmd}\''

        def _get_jobid(self):
            for i in range(5):
                time.sleep(2)  # Wait for cmd to be stably submitted to sco
                try:
                    id_str = subprocess.check_output([
                        'squeue', f'--workspace-id={self.workspace_name}',
                        '-o', 'jobname,jobid']).decode("utf-8")
                except Exception:
                    LOG.warning(f'Failed to capture job_id, retry the {i}-th time.')
                    continue
                pattern = re.compile(rf"{re.escape(self.name)}\s+(\S+)")
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
                raise RuntimeError("Cannot get IP.", f"JobID: {self.jobid}")

        def _scancel_job(self, cmd, max_retries=3):
            retries = 0
            while retries < max_retries:
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
                    LOG.warning(f"Command timed out, retrying... (Attempt {retries + 1}/{max_retries})")
                except Exception as e:
                    LOG.error("Try to scancel, but meet: ", e)
                retries += 1
            if retries == max_retries:
                LOG.error(f"Command failed after {max_retries} attempts.")

        def stop(self):
            if self.jobid:
                cmd = f"scancel --workspace-id={self.workspace_name} {self.jobid}"
                if lazyllm.config["sco_keep_record"]:
                    LOG.warning(
                        f"`sco_keep_record` is on, not executing scancel. "
                        f"You can now check the logs on the web. "
                        f"To delete by terminal, you can execute: `{cmd}`"
                    )
                else:
                    self._scancel_job(cmd)
                    time.sleep(0.5)  # Avoid the execution of scancel and scontrol too close together.

            with lazyllm.timeout(25):
                while self.status not in (Status.Done, Status.Cancelled, Status.Failed):
                    time.sleep(1)

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
                                                      'show', 'job', str(self.jobid)]).decode("utf-8")
                    id_json = json.loads(id_str)
                    job_state = id_json['status_phase'].strip().lower()
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
                except Exception:
                    pass
            return Status.Failed

    def __init__(self, partition=None, workspace_name=lazyllm.config['sco.workspace'],
                 framework='pt', nnode=1, nproc=1, ngpus=1, torchrun=False, sync=True, **kwargs):
        assert nnode >= 1, "Use at least one node."
        assert nproc >= 1, "Start at least one process."
        assert ngpus >= 1, "Use at least one GPU."
        assert type(workspace_name) is str, f"'workspace_name' is {workspace_name}. Please set workspace_name."
        self.partition = partition if partition else lazyllm.config['partition']
        self.workspace_name = workspace_name
        self.framework = framework
        self.nnode = nnode
        self.nproc = nproc
        self.ngpus = ngpus
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


class RemoteLauncher(LazyLLMLaunchersBase):
    def __new__(cls, *args, sync=False, ngpus=1, **kwargs):
        return getattr(lazyllm.launchers, lazyllm.config['launcher'])(*args, sync=sync, ngpus=ngpus, **kwargs)


def cleanup():
    # empty
    for m in (EmptyLauncher, SlurmLauncher, ScoLauncher):
        while m.all_processes:
            _, vs = m.all_processes.popitem()
            for k, v in vs:
                v.stop()
                LOG.info(f"killed job:{k}")
    LOG.close()

atexit.register(cleanup)

def _exitf(*args, **kw):
    atexit._clear()
    atexit.register(cleanup)

register_after_fork(_exitf, _exitf)
