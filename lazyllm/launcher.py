import os
import re
import time
import json
import random
import atexit
import threading
import subprocess
import multiprocessing
from enum import Enum
from queue import Queue
from datetime import datetime
from multiprocessing.util import register_after_fork
from collections import defaultdict
import uuid
import copy
import psutil

import lazyllm
from lazyllm import LazyLLMRegisterMetaClass, LazyLLMCMD, final, LOG

from lazyllm.thirdparty import kubernetes as k8s
import requests
import yaml
from typing import Literal

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
lazyllm.config.add('k8s_env_name', str, '', 'K8S_ENV_NAME')
lazyllm.config.add('k8s_config_path', str, '', 'K8S_CONFIG_PATH')


# store cmd, return message and command output.
# LazyLLMCMD's post_function can get message form this class.
class Job(object):
    def __init__(self, cmd, launcher, *, sync=True):
        assert isinstance(cmd, LazyLLMCMD)
        self._origin_cmd = cmd
        self.sync = sync
        self._launcher = launcher
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
        self.ps = subprocess.Popen(cmd.cmd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        self._get_jobid()
        self._enqueue_subprocess_output(hooks=self.output_hooks)

        if self.sync:
            self.ps.wait()
        else:
            self._launcher.all_processes[self._launcher._id].append((self.jobid, self))
            n = 0
            while self.status in (Status.TBSubmitted, Status.InQueue, Status.Pending):
                time.sleep(2)
                n += 1
                if n > 1800:  # 3600s
                    self._launcher.all_processes[self._launcher._id].pop()
                    LOG.error('Launch failed: No computing resources are available.')
                    break

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
                    try:
                        line = line.decode('gb2312')
                    except Exception:
                        pass
                if isinstance(line, str):
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
class K8sLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)
    namespace = "default"

    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super().__init__(cmd, launcher, sync=sync)
            self.deployment_name = f"deployment-{uuid.uuid4().hex[:8]}"
            self.namespace = launcher.namespace
            self.volume_configs = launcher.volume_configs
            self.gateway_name = launcher.gateway_name
            self.gateway_class_name = launcher.gateway_class_name
            self.deployment_port = 8080
            self.host = launcher.http_host
            self.path = launcher.http_path
            self.svc_type = launcher.svc_type
            self.gateway_retry = launcher.gateway_retry
            self.image = launcher.image
            self.resource_config = launcher.resource_config

        def _wrap_cmd(self, cmd):
            pythonpath = os.getenv("PYTHONPATH", '')
            precmd = (f'''export PYTHONPATH={os.getcwd()}:{pythonpath}:$PYTHONPATH '''
                      f'''&& export PATH={os.path.join(os.path.expanduser('~'), '.local/bin')}:$PATH &&''')
            if lazyllm.config['k8s_env_name']:
                precmd = f"source activate {lazyllm.config['k8s_env_name']} && " + precmd
            env_vars = os.environ
            lazyllm_vars = {k: v for k, v in env_vars.items() if k.startswith("LAZYLLM")}
            if lazyllm_vars:
                precmd += " && ".join(f"export {k}={v}" for k, v in lazyllm_vars.items()) + " && "
            precmd += '''ifconfig | grep "inet " | awk "{printf \\"LAZYLLMIP %s\\\\n\\", \$2}" &&'''  # noqa W605
            port_match = re.search(r"--open_port=(\d+)", cmd)
            if port_match:
                port = port_match.group(1)
                LOG.info(f"Port: {port}")
                self.deployment_port = int(port)
            else:
                LOG.info("Port not found")
                raise ValueError("Failed to obtain application port.")
            return precmd + " " + cmd

        def _create_deployment_spec(self, cmd, volume_configs=None):
            container = k8s.client.V1Container(
                name=self.deployment_name,
                image=self.image,
                image_pull_policy="IfNotPresent",
                command=["bash", "-c", cmd],
                resources=k8s.client.V1ResourceRequirements(
                    requests=self.resource_config.get("requests", {"cpu": "2", "memory": "16Gi"}),
                    limits=self.resource_config.get("requests", {"cpu": "2", "memory": "16Gi"})
                ),
                volume_mounts=[] if not volume_configs else [
                    k8s.client.V1VolumeMount(
                        mount_path=vol_config["mount_path"] if "__CURRENT_DIR__" not in vol_config['mount_path']
                        else vol_config['mount_path'].replace("__CURRENT_DIR__", os.getcwd()),
                        name=vol_config["name"]
                    ) for vol_config in volume_configs
                ]
            )

            volumes = []
            if volume_configs:
                for vol_config in volume_configs:
                    if "nfs_server" in vol_config and "nfs_path" in vol_config:
                        volumes.append(
                            k8s.client.V1Volume(
                                name=vol_config["name"],
                                nfs=k8s.client.V1NFSVolumeSource(
                                    server=vol_config["nfs_server"],
                                    path=vol_config["nfs_path"] if "__CURRENT_DIR__" not in vol_config['nfs_path']
                                    else vol_config['nfs_path'].replace("__CURRENT_DIR__", os.getcwd()),
                                    read_only=vol_config.get("read_only", False)
                                )
                            )
                        )
                    elif "host_path" in vol_config:
                        volumes.append(
                            k8s.client.V1Volume(
                                name=vol_config["name"],
                                host_path=k8s.client.V1HostPathVolumeSource(
                                    path=vol_config["host_path"] if "__CURRENT_DIR__" not in vol_config['host_path']
                                    else vol_config['host_path'].replace("__CURRENT_DIR__", os.getcwd()),
                                    type="Directory"
                                )
                            )
                        )
                    else:
                        LOG.error(f"{vol_config} configuration error.")
                        raise

            template = k8s.client.V1PodTemplateSpec(
                metadata=k8s.client.V1ObjectMeta(labels={"app": self.deployment_name}),
                spec=k8s.client.V1PodSpec(restart_policy="Always", containers=[container], volumes=volumes)
            )
            deployment_spec = k8s.client.V1DeploymentSpec(
                replicas=1,
                template=template,
                selector=k8s.client.V1LabelSelector(match_labels={"app": self.deployment_name})
            )
            return k8s.client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=k8s.client.V1ObjectMeta(name=self.deployment_name),
                spec=deployment_spec
            )

        def _create_deployment(self, *, fixed=False):
            api_instance = k8s.client.AppsV1Api()
            cmd = self.get_executable_cmd(fixed=fixed)
            deployment = self._create_deployment_spec(cmd.cmd, self.volume_configs)
            try:
                api_instance.create_namespaced_deployment(
                    body=deployment,
                    namespace=self.namespace
                )
                LOG.info(f"Kubernetes Deployment '{self.deployment_name}' created successfully.")
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when creating Kubernetes Deployment: {e}")
                raise

        def _delete_deployment(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            api_instance = k8s.client.AppsV1Api()
            try:
                api_instance.delete_namespaced_deployment(
                    name=self.deployment_name,
                    namespace=self.namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy="Foreground")
                )
                LOG.info(f"Kubernetes Deployment {self.deployment_name} deleted.")

                if wait_for_completion:
                    self._wait_for_deployment_deletion(timeout=timeout, check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f"Kubernetes Deployment '{self.deployment_name}' already deleted.")
                else:
                    LOG.error(f"Exception when deleting Kubernetes Deployment: {e}")
                    raise

        def _wait_for_deployment_deletion(self, timeout, check_interval):
            api_instance = k8s.client.AppsV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    api_instance.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
                    LOG.info(f"Waiting for Kubernetes Deployment '{self.deployment_name}' to be deleted...")
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f"Kubernetes Deployment '{self.deployment_name}' successfully deleted.")
                        return
                    else:
                        LOG.error(f"Error while checking Deployment deletion status: {e}")
                        raise
                time.sleep(check_interval)
            LOG.warning(f"Timeout while waiting for Kubernetes Deployment '{self.deployment_name}' to be deleted.")

        def _expose_deployment(self):
            api_instance = k8s.client.CoreV1Api()
            service = k8s.client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=k8s.client.V1ObjectMeta(name=f"service-{self.deployment_name}"),
                spec=k8s.client.V1ServiceSpec(
                    selector={"app": self.deployment_name},
                    ports=[k8s.client.V1ServicePort(port=self.deployment_port, target_port=self.deployment_port)],
                    type="ClusterIP"
                )
            )
            try:
                api_instance.create_namespaced_service(
                    namespace=self.namespace,
                    body=service
                )
                LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' created and exposed successfully.")
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when creating Service: {e}")
                raise

        def _delete_service(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            svc_instance = k8s.client.CoreV1Api()
            service_name = f"service-{self.deployment_name}"
            try:
                svc_instance.delete_namespaced_service(
                    name=service_name,
                    namespace=self.namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy="Foreground")
                )
                LOG.info(f"Kubernetes Service '{service_name}' deleted.")

                if wait_for_completion:
                    self._wait_for_service_deletion(service_name=service_name,
                                                    timeout=timeout,
                                                    check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f"Kubernetes Service '{service_name}' already deleted.")
                else:
                    LOG.error(f"Exception when deleting Kubernetes Service: {e}")
                    raise

        def _wait_for_service_deletion(self, service_name, timeout, check_interval):
            svc_instance = k8s.client.CoreV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    svc_instance.read_namespaced_service(name=service_name, namespace=self.namespace)
                    LOG.info(f"Waiting for kubernetes Service '{service_name}' to be deleted...")
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f"Kubernetes Service '{service_name}' successfully deleted.")
                        return
                    else:
                        LOG.error(f"Error while checking Service deletion status: {e}")
                        raise
                time.sleep(check_interval)
            LOG.warning(f"Timeout while waiting for kubernetes Service '{service_name}' to be deleted.")

        def _create_or_update_gateway(self):
            networking_api = k8s.client.CustomObjectsApi()
            gateway_spec = {
                "apiVersion": "gateway.networking.k8s.io/v1beta1",
                "kind": "Gateway",
                "metadata": {
                    "name": self.gateway_name,
                    "namespace": self.namespace,
                    "annotations": {
                        "networking.istio.io/service-type": self.svc_type
                    }
                },
                "spec": {
                    "gatewayClassName": self.gateway_class_name,
                    "listeners": [
                        {
                            "name": f"httproute-{self.deployment_name}",
                            "port": self.deployment_port,
                            "protocol": "HTTP",
                        }
                    ]
                }
            }

            try:
                existing_gateway = networking_api.get_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="gateways",
                    name=self.gateway_name
                )

                existing_gateway['spec']["listeners"].extend(gateway_spec['spec']["listeners"])
                networking_api.replace_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="gateways",
                    name=self.gateway_name,
                    body=existing_gateway
                )
                LOG.info(f"Kubernetes Gateway '{self.gateway_name}' updated successfully.")
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    try:
                        networking_api.create_namespaced_custom_object(
                            group="gateway.networking.k8s.io",
                            version="v1beta1",
                            namespace=self.namespace,
                            plural="gateways",
                            body=gateway_spec
                        )
                        LOG.info(f"Kubernetes Gateway '{self.gateway_name}' created successfully.")
                    except k8s.client.rest.ApiException as e_create:
                        LOG.error(f"Exception when creating Gateway: {e_create}")
                        raise
                else:
                    LOG.error(f"Exception when updating Gateway: {e}")
                    raise

        def _delete_or_update_gateway(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            gateway_instance = k8s.client.CustomObjectsApi()
            try:
                gateway = gateway_instance.get_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="gateways",
                    name=self.gateway_name
                )

                listeners = gateway['spec']['listeners']
                gateway['spec']['listeners'] = [
                    listener for listener in listeners if listener['name'] != f"httproute-{self.deployment_name}"
                ]

                if gateway['spec']['listeners']:
                    gateway_instance.replace_namespaced_custom_object(
                        group="gateway.networking.k8s.io",
                        version="v1beta1",
                        namespace=self.namespace,
                        plural="gateways",
                        name=self.gateway_name,
                        body=gateway
                    )
                    LOG.info(f"Kubernetes Gateway '{self.gateway_name}' deleted updated.")

                    if wait_for_completion:
                        self._wait_for_gateway_update(timeout=timeout, check_interval=check_interval)
                else:
                    gateway_instance.delete_namespaced_custom_object(
                        group="gateway.networking.k8s.io",
                        version="v1beta1",
                        namespace=self.namespace,
                        plural="gateways",
                        name=self.gateway_name
                    )
                    LOG.info(f"Kubernetes Gateway '{self.gateway_name}' deleted.")

                    if wait_for_completion:
                        self._wait_for_gateway_deletion(timeout=timeout, check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f"Gateway '{self.gateway_name}' already deleted.")
                else:
                    LOG.error(f"Exception when deleting or updating Gateway: {e}")
                    raise

        def _wait_for_gateway_deletion(self, timeout, check_interval):
            gateway_instance = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    gateway_instance.get_namespaced_custom_object(
                        group="gateway.networking.k8s.io",
                        version="v1beta1",
                        namespace=self.namespace,
                        plural="gateways",
                        name=self.gateway_name
                    )
                    LOG.info(f"Waiting for Gateway '{self.gateway_name}' to be deleted...")
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f"Gateway '{self.gateway_name}' successfully deleted.")
                        return
                    else:
                        LOG.error(f"Error while checking Gateway deletion status: {e}")
                        raise
                time.sleep(check_interval)
            LOG.warning(f"Timeout while waiting for Gateway '{self.gateway_name}' to be deleted.")

        def _wait_for_gateway_update(self, timeout, check_interval):
            gateway_instance = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    gateway_instance.get_namespaced_custom_object(
                        group="gateway.networking.k8s.io",
                        version="v1beta1",
                        namespace=self.namespace,
                        plural="gateways",
                        name=self.gateway_name
                    )
                    LOG.info(f"Gateway '{self.gateway_name}' status check passed.")
                    return
                except k8s.client.rest.ApiException as e:
                    LOG.error(f"Error while checking Gateway update status: {e}")
                    raise
                time.sleep(check_interval)
            LOG.warning(f"Timeout while waiting for Gateway '{self.gateway_name}' update.")

        def _create_httproute(self):
            custom_api = k8s.client.CustomObjectsApi()

            httproute_name = f"httproute-{self.deployment_name}"
            httproute_spec = {
                "apiVersion": "gateway.networking.k8s.io/v1beta1",
                "kind": "HTTPRoute",
                "metadata": {
                    "name": httproute_name,
                    "namespace": self.namespace
                },
                "spec": {
                    "parentRefs": [{
                        "name": self.gateway_name,
                        "port": self.deployment_port,
                        "sectionName": httproute_name
                    }],
                    "rules": [{
                        "matches": [{
                            "path": {
                                "type": "PathPrefix",
                                "value": self.path
                            }
                        }],
                        "backendRefs": [{
                            "name": f"service-{self.deployment_name}",
                            "port": self.deployment_port
                        }]
                    }]
                }
            }

            if self.host:
                httproute_spec["spec"]["hostnames"] = [self.host]

            try:
                custom_api.create_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="httproutes",
                    body=httproute_spec
                )
                LOG.info(f"Kubernetes HTTPRoute '{httproute_name}' created successfully.")
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when creating HTTPRoute: {e}")
                raise

        def _delete_httproute(self, wait_for_deletion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            httproute_instance = k8s.client.CustomObjectsApi()
            httproute_name = f"httproute-{self.deployment_name}"
            try:
                httproute_instance.delete_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="httproutes",
                    name=httproute_name
                )
                LOG.info(f"Kubernetes HTTPRoute '{httproute_name}' delete initiated.")
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f"HTTPRoute '{httproute_name}' already deleted.")
                    return
                else:
                    LOG.error(f"Exception when deleting HTTPRoute: {e}")
                    raise

            if wait_for_deletion:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        httproute_instance.get_namespaced_custom_object(
                            group="gateway.networking.k8s.io",
                            version="v1beta1",
                            namespace=self.namespace,
                            plural="httproutes",
                            name=httproute_name
                        )
                        LOG.info(f"Waiting for HTTPRoute '{httproute_name}' to be deleted...")
                    except k8s.client.rest.ApiException as e:
                        if e.status == 404:
                            LOG.info(f"HTTPRoute '{httproute_name}' successfully deleted.")
                            return
                        else:
                            LOG.error(f"Error while checking HTTPRoute status: {e}")
                            raise
                    time.sleep(check_interval)
                LOG.warning(f"Timeout while waiting for HTTPRoute '{httproute_name}' to be deleted.")

        def _start(self, *, fixed=False):
            self._create_deployment(fixed=fixed)
            self._expose_deployment()
            self._create_or_update_gateway()
            self._create_httproute()
            self.jobid = self._get_jobid()
            self._launcher.all_processes[self._launcher._id].append((self.jobid, self))
            ret = self.wait()
            LOG.info(ret)

        def stop(self):
            self._delete_or_update_gateway()
            self._delete_httproute()
            self._delete_service()
            self._delete_deployment()

        def _get_jobid(self):
            return f"service-{self.deployment_name}"

        def _get_gateway_service_name(self):
            core_api = k8s.client.CoreV1Api()
            try:
                services = core_api.list_namespaced_service(namespace=self.namespace)

                for service in services.items:
                    labels = service.metadata.labels
                    if labels and ("gateway" in labels.get("app", "") or self.gateway_name in service.metadata.name):
                        LOG.info(f"Kubernetes Gateway service name: {service.metadata.name}")
                        return service.metadata.name

                LOG.warning("No Service was found corresponding to the specified Gateway.")
                return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when retrieving Gateway Service: {e}")
                return None

        def _get_gateway_deployment_name(self):  # noqa: C901
            core_api = k8s.client.CoreV1Api()
            apps_v1 = k8s.client.AppsV1Api()

            gateway_service_name = self._get_gateway_service_name()
            try:
                service = core_api.read_namespaced_service(gateway_service_name, self.namespace)
                selector = service.spec.selector
                if selector:
                    label_selector = ",".join(f"{k}={v}" for k, v in selector.items())
                    pods = core_api.list_namespaced_pod(self.namespace, label_selector=label_selector).items
                    if not pods:
                        LOG.warning(f"No Pods found for Service '{gateway_service_name}' in namespace "
                                    f"'{self.namespace}'.")
                        return None

                    deployments = set()
                    for pod in pods:
                        for owner in pod.metadata.owner_references:
                            if owner.kind == "ReplicaSet":
                                rs = apps_v1.read_namespaced_replica_set(owner.name, self.namespace)
                                for rs_owner in rs.metadata.owner_references:
                                    if rs_owner.kind == "Deployment":
                                        deployments.add(rs_owner.name)

                    if deployments:
                        for deployment_name in deployments:
                            isRestart = False
                            deployment = apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
                            for container in deployment.spec.template.spec.containers:
                                if container.name == "istio-proxy" and container.image_pull_policy == "Always":
                                    container.image_pull_policy = "IfNotPresent"
                                    isRestart = True
                            if isRestart:
                                apps_v1.replace_namespaced_deployment(name=deployment_name, namespace=self.namespace,
                                                                      body=deployment)
                                LOG.info(f"Updated {deployment_name} with imagePullPolicy 'IfNotPresent'")
                        return list(deployments)
                    else:
                        LOG.warning(f"No Deployment found for Gateway '{self.gateway_name}' in namespace "
                                    f"'{self.namespace}'.")
                        return None
                else:
                    LOG.warning(f"Kubernetes Service '{gateway_service_name}' does not have a selector.")
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Error fetching Service '{gateway_service_name}': {e}")
                return None

        def _get_gateway_ip(self):
            core_api = k8s.client.CoreV1Api()
            gateway_service_name = self._get_gateway_service_name()
            if gateway_service_name is None:
                raise ValueError("Kubernetes Gateway service name not found.")
            try:
                service = core_api.read_namespaced_service(
                    name=gateway_service_name,
                    namespace=self.namespace
                )

                if service.spec.type == "LoadBalancer":
                    if service.status.load_balancer.ingress:
                        ip = service.status.load_balancer.ingress[0].ip
                        return ip
                    else:
                        LOG.warning("The LoadBalancer IP has not been assigned yet.")
                        return None
                elif service.spec.type == "NodePort":
                    nodes = core_api.list_node()
                    node_ip = nodes.items[0].status.addresses[0].address
                    return node_ip
                elif service.spec.type == "ClusterIP":
                    return service.spec.cluster_ip
                else:
                    LOG.warning("Unsupported Service type.")
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when retrieving gateway IP: {e}")
                return None

        def _get_httproute_host(self):
            custom_api = k8s.client.CustomObjectsApi()
            try:
                httproute = custom_api.get_namespaced_custom_object(
                    group="gateway.networking.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="httproutes",
                    name=f"httproute-{self.deployment_name}"
                )

                hostnames = httproute.get("spec", {}).get("hostnames", [])
                if hostnames:
                    return hostnames[0]
                else:
                    LOG.warning("Kubernetes HTTPRoute has no configured hostnames.")
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when retrieving HTTPRoute host: {e}")
                return None

        def get_jobip(self):
            host = self._get_httproute_host()
            ip = self._get_gateway_ip()
            LOG.info(f"gateway ip: {ip}, hostname: {host}")
            return host if host else ip

        def wait_for_deployment_ready(self, timeout=300):
            api_instance = k8s.client.AppsV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    deployment_status = api_instance.read_namespaced_deployment_status(
                        name=self.deployment_name,
                        namespace=self.namespace
                    ).status
                    if deployment_status.available_replicas and deployment_status.available_replicas > 0:
                        LOG.info(f"Kubernetes Deployment '{self.deployment_name}' is running.")
                        return True
                    time.sleep(2)
                except k8s.client.rest.ApiException as e:
                    LOG.error(f"Exception when reading Deployment status: {e}")
                    raise
            LOG.warning(f"Timed out waiting for Deployment '{self.deployment_name}' to be ready.")
            return False

        def wait_for_service_ready(self, timeout=300):
            svc_instance = k8s.client.CoreV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    service = svc_instance.read_namespaced_service(
                        name=f"service-{self.deployment_name}",
                        namespace=self.namespace
                    )
                    if service.spec.type == "LoadBalancer" and service.status.load_balancer.ingress:
                        ip = service.status.load_balancer.ingress[0].ip
                        LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' is ready with IP: {ip}")
                        return ip
                    elif service.spec.cluster_ip:
                        LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' is ready with ClusterIP: "
                                 f"{service.spec.cluster_ip}")
                        return service.spec.cluster_ip
                    elif service.spec.type == "NodePort":
                        node_ports = [p.node_port for p in service.spec.ports]
                        if node_ports:
                            nodes = svc_instance.list_node()
                            for node in nodes.items:
                                for address in node.status.addresses:
                                    if address.type == "InternalIP":
                                        node_ip = address.address
                                        LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' is ready on "
                                                 f"NodePort(s): {node_ports} at Node IP: {node_ip}")
                                        return {"ip": node_ip, "ports": node_ports}
                                    elif address.type == "ExternalIP":
                                        node_ip = address.address
                                        LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' is ready on "
                                                 f"NodePort(s): {node_ports} at External Node IP: {node_ip}")
                                        return {"ip": node_ip, "ports": node_ports}
                    LOG.info(f"Kubernetes Service 'service-{self.deployment_name}' is not ready yet. Retrying...")
                    time.sleep(2)
                except k8s.client.rest.ApiException as e:
                    LOG.error(f"Exception when reading Service status: {e}")
                    raise
            LOG.warning(f"Timed out waiting for Service 'service-{self.deployment_name}' to be ready.")
            return None

        def _is_gateway_ready(self, timeout):
            url = f"http://{self.get_jobip()}:{self.deployment_port}{self.path}"
            for i in range(self.gateway_retry):
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code != 503:
                        LOG.info(f"Kubernetes Service is ready at '{url}'")
                        self.queue.put(f"Uvicorn running on {url}")
                        return True
                    else:
                        LOG.info(f"Kubernetes Service at '{url}' returned status code {response.status_code}")
                except requests.RequestException as e:
                    LOG.error(f"Failed to access service at '{url}': {e}")
                    raise
                time.sleep(timeout)

            self.queue.put(f"ERROR: Kubernetes Service failed to start on '{url}'.")
            return False

        def wait_for_gateway(self, timeout=300, interval=5):  # noqa: C901
            core_v1 = k8s.client.CoreV1Api()
            apps_v1 = k8s.client.AppsV1Api()
            gateway_service_name = self._get_gateway_service_name()
            gateway_deployment_names = self._get_gateway_deployment_name()
            service_ready = False
            deployment_ready = False

            start_time = time.time()
            while time.time() - start_time < timeout:
                if not service_ready:
                    try:
                        service = core_v1.read_namespaced_service(gateway_service_name, self.namespace)
                        if service.spec.type in ["NodePort", "LoadBalancer"]:
                            if service.spec.type == "LoadBalancer":
                                if service.status.load_balancer.ingress:
                                    LOG.info(f"Kubernetes Service '{gateway_service_name}' is ready with "
                                             "LoadBalancer IP.")
                                    service_ready = True
                                else:
                                    LOG.info(f"Kubernetes Service '{gateway_service_name}' LoadBalancer IP "
                                             "not available yet.")
                                    service_ready = False
                            else:
                                if any(port.node_port for port in service.spec.ports):
                                    LOG.info(f"Kubernetes Service '{gateway_service_name}' is ready with "
                                             "NodePort configuration.")
                                    service_ready = True
                                else:
                                    LOG.info(f"Kubernetes Service '{gateway_service_name}' NodePort not assigned yet.")
                                    service_ready = False
                        else:
                            LOG.error(f"Unexpected Kubernetes Service type: {service.spec.type}.")
                            service_ready = False
                    except k8s.client.rest.ApiException as e:
                        LOG.error(f"Kubernetes Service '{gateway_service_name}' not found yet: {e}")
                        service_ready = False
                if not deployment_ready:
                    for deployment_name in gateway_deployment_names:
                        try:
                            deployment = apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
                            if deployment.status.available_replicas and deployment.status.available_replicas > 0:
                                LOG.info(f"Kubernetes Deployment '{deployment_name}' is ready with "
                                         f"{deployment.status.available_replicas} replicas.")
                                deployment_ready = True
                                break
                            else:
                                LOG.info(f"Kubernetes Deployment '{deployment_name}' is not fully ready yet.")
                                deployment_ready = False
                        except k8s.client.rest.ApiException as e:
                            LOG.warning(f"Kubernetes Deployment '{deployment_name}' not found yet: {e}")
                            deployment_ready = False

                if service_ready and deployment_ready and self._is_gateway_ready(timeout=interval):
                    LOG.info(f"Kubernetes Gateway '{self.gateway_name}' is fully ready.")
                    return True

                time.sleep(interval)

            LOG.error(f"Kubernetes Gateway '{self.gateway_name}' failed to become ready with {timeout} seconds.")
            return False

        def wait_for_httproute(self, timeout=300):
            custom_api = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    httproutes = custom_api.list_namespaced_custom_object(
                        group="gateway.networking.k8s.io",
                        version="v1beta1",
                        namespace=self.namespace,
                        plural="httproutes"
                    ).get('items', [])

                    for httproute in httproutes:
                        if httproute['metadata']['name'] == f"httproute-{self.deployment_name}":
                            LOG.info(f"Kubernetes HTTPRoute 'httproute-{self.deployment_name}' is ready.")
                            return True
                    LOG.info(f"Waiting for HTTPRoute 'httproute-{self.deployment_name}' to be ready...")
                except k8s.client.rest.ApiException as e:
                    LOG.error(f"Exception when checking HTTPRoute status: {e}")
                    raise

                time.sleep(2)
            LOG.warning(f"Timeout waiting for HTTPRoute 'httproute-{self.deployment_name}' to be ready.")
            return False

        def wait(self):
            deployment_ready = self.wait_for_deployment_ready()
            if not deployment_ready:
                raise TimeoutError("Kubernetes Deployment did not become ready in time.")

            service_ip = self.wait_for_service_ready()
            if not service_ip:
                raise TimeoutError("Kubernetes Service did not become ready in time.")

            httproute_ready = self.wait_for_httproute()
            if not httproute_ready:
                raise TimeoutError("Kubernetes Httproute did not become ready in time.")

            gateway_ready = self.wait_for_gateway()
            if not gateway_ready:
                raise TimeoutError("Kubernetes Gateway did not become ready in time.")

            return {"deployment": Status.Running, "service_ip": service_ip,
                    "gateway": Status.Running, "httproute": Status.Running}

        @property
        def status(self):
            api_instance = k8s.client.AppsV1Api()
            try:
                deployment_status = api_instance.read_namespaced_deployment_status(
                    name=self.deployment_name,
                    namespace=self.namespace
                ).status
                if deployment_status.available_replicas and deployment_status.available_replicas > 0:
                    return Status.Running
                else:
                    return Status.Pending
            except k8s.client.rest.ApiException as e:
                LOG.error(f"Exception when reading Deployment status: {e}")
                return Status.Failed

    def __init__(self, kube_config_path=None, volume_configs=None, image=None, resource_config=None,
                 namespace=None, gateway_name=None, gateway_class_name=None, host=None, path=None,
                 svc_type: Literal["LoadBalancer", "NodePort", "ClusterIP"] = None, retry=3,
                 sync=True, ngpus=None, **kwargs):
        super().__init__()
        self.gateway_retry = retry
        self.sync = sync
        self.ngpus = ngpus
        config_data = self._read_config_file(lazyllm.config['k8s_config_path']) if lazyllm.config['k8s_config_path'] \
            else {}
        self.volume_configs = volume_configs if volume_configs else config_data.get('volume', [])
        self.image = image if image else config_data.get('container_image', "lazyllm/lazyllm:k8s_launcher")
        self.resource_config = resource_config if resource_config else config_data.get('resource', {})
        self.kube_config_path = kube_config_path if kube_config_path \
            else config_data.get('kube_config_path', "~/.kube/config")
        self.svc_type = svc_type if svc_type else config_data.get("svc_type", "LoadBalancer")
        self.namespace = namespace if namespace else config_data.get("namespace", "default")
        self.gateway_name = gateway_name if gateway_name else config_data.get("gateway_name", "lazyllm-gateway")
        self.gateway_class_name = gateway_class_name if gateway_class_name \
            else config_data.get("gateway_class_name", "istio")
        self.http_host = host if host else config_data.get("host", None)
        self.http_path = path if path else config_data.get("path", '/generate')

    def _read_config_file(self, file_path):
        assert os.path.isabs(file_path), "Resource config file path must be an absolute path."
        with open(file_path, 'r') as fp:
            try:
                data = yaml.safe_load(fp)
                return data
            except yaml.YAMLError as e:
                LOG.error(f"Exception when reading resource configuration file: {e}")
                raise ValueError("Kubernetes resource configuration file format error.")

    def makejob(self, cmd):
        k8s.config.load_kube_config(self.kube_config_path)
        return K8sLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, f, *args, **kw):
        if isinstance(f, K8sLauncher.Job):
            f.start()
            LOG.info("Launcher started successfully.")
            self.job = f
            return f.return_value
        elif callable(f):
            LOG.info("Async execution in Kubernetes is not supported currently.")
            raise RuntimeError("Kubernetes launcher requires a Deployment object.")

@final
class EmptyLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)

    @final
    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super(__class__, self).__init__(cmd, launcher, sync=sync)

        def _wrap_cmd(self, cmd):
            if self._launcher.ngpus == 0:
                return cmd
            gpus = self._launcher._get_idle_gpus()
            if gpus and lazyllm.config['cuda_visible']:
                if self._launcher.ngpus is None:
                    empty_cmd = f'CUDA_VISIBLE_DEVICES={gpus[0]} '
                elif self._launcher.ngpus <= len(gpus):
                    empty_cmd = 'CUDA_VISIBLE_DEVICES=' + \
                                ','.join([str(n) for n in gpus[:self._launcher.ngpus]]) + ' '
                else:
                    error_info = (f'Not enough GPUs available. Requested {self._launcher.ngpus} GPUs, '
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
            return '127.0.0.1'

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
            slurm_cmd = f'srun -p {self._launcher.partition} -N {self._launcher.nnode} --job-name={self.name}'
            if self._launcher.nproc:
                slurm_cmd += f' -n{self._launcher.nproc}'
            if self._launcher.timeout:
                slurm_cmd += f' -t {self._launcher.timeout}'
            if self._launcher.ngpus:
                slurm_cmd += f' --gres=gpu:{self._launcher.ngpus}'
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
            launcher = self._launcher
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
                except Exception as e:
                    lazyllm.LOG.error(f'Failed to get job status, reason is {str(e)}')
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
    for m in (EmptyLauncher, SlurmLauncher, ScoLauncher, K8sLauncher):
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
