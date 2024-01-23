from typing import Any
import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final
from .flow import FlowBase
from enum import Enum
import os
import re
import time
import subprocess
import atexit

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


def exec_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding='utf-8', executable='/bin/bash')
    out, err = p.communicate()
    return_code = p.returncode
    assert return_code == 0, f'Exec cmd \'{cmd}\' failed, errmsg: {err}'
    return out.strip()


@final
class EmptyLauncher(LazyLLMLaunchersBase):
    def makejob(self, cmd):
        return cmd.cmd

    def launch(self, f, *args, **kw):
        if isinstance(f, (str, tuple, list)):
            return exec_cmd(f)
        elif callable(f):
            return f(*args, **kw)
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')


@final
class SubprocessLauncher(LazyLLMLaunchersBase):
    def makejob(self, cmd):
        return cmd.cmd

    # TODO(wangzhihong): support return value
    def launch(self, f, *args, **kw) -> None:
        if isinstance(f, (str, tuple, list)):
            return exec_cmd(f)
        elif callable(f):
            import multiprocessing
            p = multiprocessing.Process(target=f, args=args, kwargs=kw)
            p.start()
            p.join()
        else:
            raise RuntimeError('Invalid cmd given, please check the return value of cmd.')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, *, sync=True):
        self.cmd = cmd.cmd
        self.return_value = cmd.return_value
        self.post_function = cmd.post_function
        self.sync = sync

    def get_return_value(self):
        return self.return_value if self.return_value else (
            self.post_function(self) if self.post_function else self)

    def start(self):
        raise NotImplementedError

    @property
    def status(self):
        raise NotImplementedError

    def __deepcopy__(self, memo=None):
        raise RuntimeError('Cannot copy Job object')


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
            self.name = str(hex(hash(cmd)))[2:]

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
            process = subprocess.Popen(self.slurm_cmd + f' bash -c \'{self.cmd.cmd}\'', shell=True, encoding='utf-8', executable='/bin/bash')
            self.ps = process
            self.get_jobid()
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
                # os.system(f"scancel --quiet {self.jobid}")
                cmd = f"scancel --quiet {self.jobid}"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding='utf-8', executable='/bin/bash')
            if self.ps:
                self.ps.terminate()

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
    def __init__(self, partition=None, nnode=1, nproc=1, ngpus=None, timeout=None, *, sync=True):
        self.partition = partition
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
    def __init__(self, nproc, ngpus, timeout):
        self.nproc, self.ngpus, self.timeout = nproc, ngpus, timeout
        super(__class__, self).__init__()

    def launch(self, cmd) -> None:
        assert isinstance(cmd, str), 'Sco launcher only support cmd'
        os.system(f'{cmd}')


def cleanup():
    # empty

    # slurm
    for k, v in SlurmLauncher.all_processes.items():
        v.stop()
        print(f"killed job:{k}")

    # sco

atexit.register(cleanup)
