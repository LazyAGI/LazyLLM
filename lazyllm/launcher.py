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

import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final
from .flow import FlowBase

class Status(Enum):
    TBSubmitted = 0,
    InQueue = 1
    Running = 2,
    Pending = 3,
    Done = 100,
    Cancelled = 101, # TODO(wangzhihong): support cancel job
    Failed = 102,


class LazyLLMLaunchersBase(object, metaclass=LazyLLMRegisterMetaClass):
    def __init__(self) -> None:
        self.status = Status.TBSubmitted

    def makejob(self, cmd):
        raise NotImplementedError

    def launch(self, *args, **kw):
        raise NotImplementedError


lazyllm.launchers['Status'] = Status


def get_rand_str():
    now = datetime.now()
    return now.strftime("%S%M") + str(random.randint(3, 2000))


@final
class EmptyLauncher(LazyLLMLaunchersBase):
    def __init__(self, subprocess=False):
        super().__init__()
        self.subprocess = subprocess

    def makejob(self, cmd):
        job = Job(cmd)
        return job

    def launch(self, f, *args, **kw):
        if isinstance(f, Job):
            self.exec_cmd(f)
            return f.get_return_value()
        elif callable(f):
            if not self.subprocess:
                return f(*args, **kw)
            else:
                import multiprocessing
                p = multiprocessing.Process(target=f, args=args, kwargs=kw)
                p.start()
                p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')

    def exec_cmd(self, job):
        cmd = job.cmd
        print("Command:", cmd)
        if lazyllm.mode == lazyllm.Mode.Display:
            return
        p = subprocess.Popen(cmd.cmd, shell=True, encoding='utf-8', executable='/bin/bash')
        p.wait()
        return


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, *, sync=True):
        self.cmd = cmd
        self.return_value = cmd.return_value
        self.post_function = cmd.post_function
        self.sync = sync

    def get_return_value(self):
        return self.return_value if self.return_value else (
            self.post_function(self) if self.post_function else self)

    def start(self):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError

    @property
    def status(self):
        raise NotImplementedError

    def __deepcopy__(self, memo=None):
        raise RuntimeError('Cannot copy Job object')

    def restart(self):
        self.stop()
        time.sleep(2)
        self.start()

    def wait(self):
        pass

@final
class SlurmLauncher(LazyLLMLaunchersBase):
    # In order to obtain the jobid to monitor and terminate the job more
    # conveniently, only one srun command is allowed in one Job
    all_processes=dict()
    count = 0

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, sync=sync)
            self.name = str(hex(hash(get_rand_str() + cmd.cmd)))[2:10]
            self.queue = Queue()

            # Assemble the order
            self.slurm_cmd = f'srun -p {launcher.partition} -N {launcher.nnode} --job-name={self.name}'
            if launcher.nproc:
                self.slurm_cmd += f' -n{launcher.nproc}'
            if launcher.timeout:
                self.slurm_cmd += f' -t {launcher.timeout}'
            if launcher.ngpus:
                self.slurm_cmd += f' --gres=gpu:{launcher.ngpus}'
            self.cmd = cmd

            self.jobid = None
            self.ip = None
            self.ps = None
        
        def start(self):
            print("Command:", self.slurm_cmd + f' bash -c \'{self.cmd}\'')
            if lazyllm.mode == lazyllm.Mode.Display:
                return
            process = subprocess.Popen(self.slurm_cmd + f' bash -c \'{self.cmd.cmd}\'', shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.ps = process
            self.get_jobid()

            self.output_thread_event = threading.Event()
            def enqueue_output(out, queue):
                for line in iter(out.readline, b''):
                    try:
                        line = line.decode('utf-8')
                    except:
                        pass
                    queue.put(line)
                    print(f'{self.jobid}: ', line.rstrip())
                    if self.output_thread_event.is_set():
                        break
                out.close()
            self.output_thread = threading.Thread(target=enqueue_output, args=(self.ps.stdout, self.queue))
            self.output_thread.daemon = True
            self.output_thread.start()
            
            if self.sync:
                process.wait()
            else:
                SlurmLauncher.all_processes[self.jobid] = self

        def get_jobid(self):
            time.sleep(0.5) # Wait for cmd to be stably submitted to slurm
            id_str = subprocess.check_output(['squeue', '--name='+self.name, '--noheader'])
            if id_str:
                id_list = id_str.decode().strip().split()
                self.jobid = id_list[0]

        def get_jobip(self):
            id_str = subprocess.check_output(['squeue', '--name='+self.name, '--noheader'])
            id_list = id_str.decode().strip().split()
            self.ip = id_list[10]
            return self.ip

        def stop(self):
            if self.jobid:
                cmd = f"scancel --quiet {self.jobid}"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding='utf-8', executable='/bin/bash')
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
                        if job_state=='running':
                            return Status.Running
                        elif job_state=='tbsubmitted':
                            return Status.TBSubmitted
                        elif job_state=='inqueue':
                            return Status.InQueue
                        elif job_state=='pending':
                            return Status.Pending
                        elif job_state=='done':
                            return Status.Done
                        elif job_state=='cancelled':
                            return Status.Cancelled
                        else:
                            return Status.Failed
            else:
                return Status.Failed


    # TODO(wangzhihong): support configs; None -> lookup config
    def __init__(self, partition=None, nnode=1, nproc=1, ngpus=None, timeout=None, *, sync=True, **kwargs):
        self.partition = partition if partition else os.getenv('LAZYLLM_SLURM_PART', None)
        self.nnode, self.nproc, self.ngpus, self.timeout =nnode, nproc, ngpus, timeout
        self.sync = sync
        self.num_can_use_nodes = 5
        SlurmLauncher.count += 1
        super(__class__, self).__init__()

    def makejob(self, cmd):
        job = SlurmLauncher.Job(cmd, launcher=self)
        job.sync = self.sync
        return job

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
            result = [base+str(x) for x in nums]
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
                    avg_gpus = int(num_gpus/num_nodes)
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
                if num_nodes==1:
                    idle_nodes.append(node_list)
                else:
                    idle_nodes += self._expand_nodelist(node_list)

        # Add idle nodes under resource constraints
        num_allocated_nodes = len(node_dict)
        num_append_nodes = num_can_use_nodes - num_allocated_nodes

        for i, node_ip in enumerate(idle_nodes):
            if i+1 <= num_append_nodes:
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
        return job.get_return_value()


@final
class ScoLauncher(LazyLLMLaunchersBase):
    all_processes=dict()

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, sync=sync)
            # SCO job name must start with a letter
            self.name = 's' + str(hex(hash(get_rand_str() + cmd.cmd)))[2:10]
            self.queue = Queue()
            self.workspace_name = launcher.workspace_name
            self.torchrun = launcher.torchrun

            # Assemble the cmd
            self.sco_cmd = (
                f'srun -p {launcher.partition} '
                f'--workspace-name {self.workspace_name} '
                f'--job-name={self.name} '
                f'-f {launcher.framework} '
                f'-r N2lS.Ie.I60.{launcher.ngpus} '
                f'-N {launcher.nnode} '
                f'--priority highest '
                )
            self.torchrun_cmd = (
                'python -m torch.distributed.run '
                f'--nproc_per_node {launcher.nproc} '
                )
            if launcher.nnode==1:
                # SCO for mpi：supports multiple cards in a single machine
                self.sco_cmd += '-m '
                self.torchrun_cmd += (
                    f'--nnodes {launcher.nnode} '
                    '--node_rank 0 '
                )
            else:
                # SCO for All Reduce-DDP: support multiple machines and multiple cards
                self.sco_cmd += '-d AllReduce '
                self.torchrun_cmd += (
                    '--nnodes ${WORLD_SIZE} '
                    '--node_rank ${RANK} '
                    '--master_addr ${MASTER_ADDR} '
                    '--master_port ${MASTER_PORT} '
                )
            self.precmd = f'''export PYTHONPATH={os.getcwd()}:$PYTHONPATH && '''
            # For SCO: bash -c 'ifconfig | grep "inet " | awk "{printf \"LAZYLLMIP %s\\n\", \$2}"'
            self.precmd += '''ifconfig | grep "inet " | awk "{printf \\"LAZYLLMIP %s\\\\n\\", \$2}" && '''

            self.cmd = cmd
            if self.torchrun:
                # Delete 'python' in cmd
                if self.cmd.cmd.strip().startswith('python'):
                    self.cmd.cmd = self.cmd.cmd.strip()[6:]

            self.jobid = None
            self.ip = None
            self.ps = None
        
        def start(self):
            if self.torchrun:
                print("Command:", self.sco_cmd + f' bash -c \'{self.torchrun_cmd} {self.cmd}\'')
                cmd = f' bash -c \'{self.precmd}{self.torchrun_cmd} {self.cmd.cmd}\''
            else:
                print("Command:", self.sco_cmd + f' bash -c \'{self.cmd}\'')
                cmd = f' bash -c \'{self.precmd}{self.cmd.cmd}\''
            if lazyllm.mode == lazyllm.Mode.Display:
                return
            self.ps = subprocess.Popen(
                self.sco_cmd + cmd,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
                )
            self.get_jobid()

            self.output_thread_event = threading.Event()
            def enqueue_output(out, queue):
                for line in iter(out.readline, b''):
                    try:
                        line = line.decode('utf-8')
                    except:
                        pass
                    queue.put(line)
                    if not self.ip and 'LAZYLLMIP' in line:
                        self.ip = line.split()[-1]
                    print(f'{self.jobid}: ', line.rstrip())
                    if self.output_thread_event.is_set():
                        break
                out.close()
            self.output_thread = threading.Thread(target=enqueue_output, args=(self.ps.stdout, self.queue))
            self.output_thread.daemon = True
            self.output_thread.start()
            
            if self.sync:
                self.ps.wait()
            else:
                SlurmLauncher.all_processes[self.jobid] = self

        def get_jobid(self):
            time.sleep(0.5) # Wait for cmd to be stably submitted to sco
            id_str = subprocess.check_output([
                'squeue',
                f'--workspace-name={self.workspace_name}',
                '-o',
                'jobname,jobid']).decode("utf-8")
            pattern = re.compile(rf"{re.escape(self.name)}\s+(\S+)")
            match = pattern.search(id_str)
            if match:
                self.jobid = match.group(1).strip()

        def get_jobip(self):
            if self.ip:
                return self.ip
            else:
                raise RuntimeError("Cannot get IP.", f"JobID: {self.jobid}")

        def stop(self):
            if self.jobid:
                cmd = f"scancel --workspace-name={self.workspace_name} {self.jobid}"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding='utf-8', executable='/bin/bash')
            if self.ps:
                self.ps.terminate()
                self.queue = Queue()
                self.output_thread_event.set()
                self.output_thread.join()

        @property
        def status(self):
            if self.jobid:
                try:
                    id_str = subprocess.check_output([
                        'scontrol',
                        f'--workspace-name={self.workspace_name}',
                        'show',
                        'job',
                        str(self.jobid)
                        ]).decode("utf-8")
                    id_json = json.loads(id_str)
                    job_state = id_json['status_phase'].strip().lower()
                    if job_state=='running':
                        return Status.Running
                    elif job_state=='tbsubmitted':
                        return Status.TBSubmitted
                    elif job_state==['suspending', 'suspended']:
                        return Status.TBSubmitted
                    elif job_state in['waiting', 'init', 'queueing', 'creating', 'restarting', 'recovering', 'starting']:
                        return Status.InQueue
                    elif job_state=='succeeded':
                            return Status.Done
                    else:
                        # status: failed, deleting
                        return Status.Failed
                except:
                    return Status.Failed
            else:
                return Status.Failed

    def __init__(self,
            partition=None,
            workspace_name='expert-services',
            framework='pt',
            nnode=1,
            nproc=1,
            ngpus=1,
            torchrun=False,
            sync=True,
            **kwargs):
        assert nnode >= 1, "Use at least one node."
        assert nproc >= 1, "Start at least one process."
        assert ngpus >= 1, "Use at least one GPU."
        assert type(partition) is str, f"'partition' is {partition}. Please set partition."
        assert type(workspace_name) is str, f"'workspace_name' is {workspace_name}. Please set workspace_name."
        self.partition = partition
        self.workspace_name = workspace_name
        self.framework = framework
        self.nnode = nnode
        self.nproc = nproc
        self.ngpus = ngpus
        self.torchrun = torchrun
        self.sync = sync
        super(__class__, self).__init__()

    def makejob(self, cmd):
        job = ScoLauncher.Job(cmd, launcher=self)
        job.sync = self.sync
        return job

    def launch(self, job) -> None:
        assert isinstance(job, ScoLauncher.Job), 'Sco launcher only support cmd'
        job.start()
        if self.sync:
            while job.status == Status.Running:
                time.sleep(10)
            job.stop()
        return job.get_return_value()


def cleanup():
    # empty

    # slurm
    for k, v in SlurmLauncher.all_processes.items():
        v.stop()
        print(f"killed job:{k}")

    # sco
    for k, v in ScoLauncher.all_processes.items():
        v.stop()
        print(f"killed job:{k}")

atexit.register(cleanup)
