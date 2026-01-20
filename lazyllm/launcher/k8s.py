import os
import re
import time
import uuid
import yaml
import requests
from typing import Literal
from collections import defaultdict

import lazyllm
from lazyllm import final, LOG
from lazyllm.thirdparty import kubernetes as k8s
from .base import LazyLLMLaunchersBase, Job, Status


lazyllm.config.add('k8s_env_name', str, '', 'K8S_ENV_NAME',
                   description='The default k8s environment name to use if no environment name is specified.')
lazyllm.config.add('k8s_config_path', str, '', 'K8S_CONFIG_PATH',
                   description='The default k8s configuration path to use if no configuration path is specified.')
lazyllm.config.add('k8s_device_type', str, 'nvidia.com/gpu', 'K8S_DEVICE_TYPE',
                   description='The default k8s device type to use if no device type is specified.')


@final
class K8sLauncher(LazyLLMLaunchersBase):
    all_processes = defaultdict(list)
    namespace = 'default'

    class Job(Job):
        def __init__(self, cmd, launcher, *, sync=True):
            super().__init__(cmd, launcher, sync=sync)
            self.launch_type = launcher.launch_type
            prefix = 'deployment' if self.launch_type == 'inference' else 'job'
            self.deployment_name = f'{prefix}-{uuid.uuid4().hex[:8]}'
            self.ngpus = launcher.ngpus
            self.namespace = launcher.namespace
            self.volume_configs = launcher.volume_configs
            self.gateway_name = launcher.gateway_name
            self.gateway_class_name = launcher.gateway_class_name
            self.deployment_port = 8080
            self.host = launcher.http_host
            self.path = launcher.http_path
            self.svc_type = launcher.svc_type
            self.gateway_retry = launcher.gateway_retry
            self.on_gateway = launcher.on_gateway
            self.image = launcher.image
            self.resource_config = launcher.resource_config if launcher.resource_config else {}

        def _wrap_cmd(self, cmd):
            pythonpath = os.getenv('PYTHONPATH', '')
            precmd = (f'''export PYTHONPATH={os.getcwd()}:{pythonpath}:$PYTHONPATH '''
                      f'''&& export PATH={os.path.join(os.path.expanduser('~'), '.local/bin')}:$PATH &&''')
            if lazyllm.config['k8s_env_name']:
                precmd = f'source activate {lazyllm.config["k8s_env_name"]} && ' + precmd
            env_vars = os.environ
            lazyllm_vars = {k: v for k, v in env_vars.items() if k.startswith('LAZYLLM')}
            if lazyllm_vars:
                precmd += ' && '.join(f'export {k}={v}' for k, v in lazyllm_vars.items()) + ' && '
            precmd += '''ifconfig | grep "inet " | awk "{printf \\"LAZYLLMIP %s\\\\n\\", \$2}" &&'''  # noqa W605
            if self.launch_type == 'inference':
                port_match = re.search(r'--(?:open_)?port=(\d+)', cmd)
                if port_match:
                    port = port_match.group(1)
                    LOG.info(f'Port: {port}')
                    self.deployment_port = int(port)
                else:
                    LOG.info('Port not found')
                    raise ValueError('Failed to obtain application port.')
            return precmd + ' ' + cmd

        def _create_container_and_volumes(self, cmd, volume_configs=None):
            device_type = lazyllm.config['k8s_device_type']
            resource_config = self.resource_config.get('requests', {'cpu': '2', 'memory': '16Gi'})
            if device_type:
                resource_config[device_type] = self.ngpus

            container = k8s.client.V1Container(
                name=self.deployment_name,
                image=self.image,
                image_pull_policy='IfNotPresent',
                command=['bash', '-c', cmd],
                resources=k8s.client.V1ResourceRequirements(
                    requests=resource_config,
                    limits=resource_config
                ),
                volume_mounts=[] if not volume_configs else [
                    k8s.client.V1VolumeMount(
                        mount_path=vol_config['mount_path'] if '__CURRENT_DIR__' not in vol_config['mount_path']
                        else vol_config['mount_path'].replace('__CURRENT_DIR__', os.getcwd()),
                        name=vol_config['name']
                    ) for vol_config in volume_configs
                ]
            )

            volumes = []
            if volume_configs:
                for vol_config in volume_configs:
                    if 'nfs_server' in vol_config and 'nfs_path' in vol_config:
                        volumes.append(
                            k8s.client.V1Volume(
                                name=vol_config['name'],
                                nfs=k8s.client.V1NFSVolumeSource(
                                    server=vol_config['nfs_server'],
                                    path=vol_config['nfs_path'] if '__CURRENT_DIR__' not in vol_config['nfs_path']
                                    else vol_config['nfs_path'].replace('__CURRENT_DIR__', os.getcwd()),
                                    read_only=vol_config.get('read_only', False)
                                )
                            )
                        )
                    elif 'host_path' in vol_config:
                        volumes.append(
                            k8s.client.V1Volume(
                                name=vol_config['name'],
                                host_path=k8s.client.V1HostPathVolumeSource(
                                    path=vol_config['host_path'] if '__CURRENT_DIR__' not in vol_config['host_path']
                                    else vol_config['host_path'].replace('__CURRENT_DIR__', os.getcwd()),
                                    type='Directory'
                                )
                            )
                        )
                    else:
                        LOG.error(f'{vol_config} configuration error.')
                        raise

            return container, volumes

        def _create_deployment_spec(self, cmd, volume_configs=None):
            container, volumes = self._create_container_and_volumes(cmd, volume_configs)

            template = k8s.client.V1PodTemplateSpec(
                metadata=k8s.client.V1ObjectMeta(labels={'app': self.deployment_name}),
                spec=k8s.client.V1PodSpec(restart_policy='Always', containers=[container], volumes=volumes)
            )
            deployment_spec = k8s.client.V1DeploymentSpec(
                replicas=1,
                template=template,
                selector=k8s.client.V1LabelSelector(match_labels={'app': self.deployment_name})
            )
            return k8s.client.V1Deployment(
                api_version='apps/v1',
                kind='Deployment',
                metadata=k8s.client.V1ObjectMeta(name=self.deployment_name),
                spec=deployment_spec
            )

        def _create_job_spec(self, cmd, volume_configs=None):
            container, volumes = self._create_container_and_volumes(cmd, volume_configs)

            # use OnFailure for job to avoid infinite restart
            template = k8s.client.V1PodTemplateSpec(
                metadata=k8s.client.V1ObjectMeta(labels={'app': self.deployment_name}),
                spec=k8s.client.V1PodSpec(restart_policy='OnFailure', containers=[container], volumes=volumes)
            )
            job_spec = k8s.client.V1JobSpec(
                template=template,
                backoff_limit=3
            )
            return k8s.client.V1Job(
                api_version='batch/v1',
                kind='Job',
                metadata=k8s.client.V1ObjectMeta(name=self.deployment_name),
                spec=job_spec
            )

        def _create_deployment(self, *, cmd):
            api_instance = k8s.client.AppsV1Api()
            deployment = self._create_deployment_spec(cmd.cmd, self.volume_configs)
            try:
                api_instance.create_namespaced_deployment(
                    body=deployment,
                    namespace=self.namespace
                )
                LOG.info(f'Kubernetes Deployment "{self.deployment_name}" created successfully.')
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when creating Kubernetes Deployment: {e}')
                raise

        def _create_job(self, *, cmd):
            api_instance = k8s.client.BatchV1Api()
            job = self._create_job_spec(cmd.cmd, self.volume_configs)
            try:
                api_instance.create_namespaced_job(
                    body=job,
                    namespace=self.namespace
                )
                LOG.info(f'Kubernetes Job "{self.deployment_name}" created successfully.')
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when creating Kubernetes Job: {e}')
                raise

        def _delete_deployment(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            api_instance = k8s.client.AppsV1Api()
            try:
                api_instance.delete_namespaced_deployment(
                    name=self.deployment_name,
                    namespace=self.namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy='Foreground')
                )
                LOG.info(f'Kubernetes Deployment {self.deployment_name} deleted.')

                if wait_for_completion:
                    self._wait_for_deployment_deletion(timeout=timeout, check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f'Kubernetes Deployment "{self.deployment_name}" already deleted.')
                else:
                    LOG.error(f'Exception when deleting Kubernetes Deployment: {e}')
                    raise

        def _delete_job(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            api_instance = k8s.client.BatchV1Api()
            try:
                api_instance.delete_namespaced_job(
                    name=self.deployment_name,
                    namespace=self.namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy='Foreground')
                )
                LOG.info(f'Kubernetes Job {self.deployment_name} deleted.')

                if wait_for_completion:
                    self._wait_for_job_deletion(timeout=timeout, check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f'Kubernetes Job "{self.deployment_name}" already deleted.')
                else:
                    LOG.error(f'Exception when deleting Kubernetes Job: {e}')
                    raise

        def _wait_for_deployment_deletion(self, timeout, check_interval):
            api_instance = k8s.client.AppsV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    api_instance.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
                    LOG.info(f'Waiting for Kubernetes Deployment "{self.deployment_name}" to be deleted...')
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f'Kubernetes Deployment "{self.deployment_name}" successfully deleted.')
                        return
                    else:
                        LOG.error(f'Error while checking Deployment deletion status: {e}')
                        raise
                time.sleep(check_interval)
            LOG.warning(f'Timeout while waiting for Kubernetes Deployment "{self.deployment_name}" to be deleted.')

        def _wait_for_job_deletion(self, timeout, check_interval):
            api_instance = k8s.client.BatchV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    api_instance.read_namespaced_job(name=self.deployment_name, namespace=self.namespace)
                    LOG.info(f'Waiting for Kubernetes Job "{self.deployment_name}" to be deleted...')
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f'Kubernetes Job "{self.deployment_name}" successfully deleted.')
                        return
                    else:
                        LOG.error(f'Error while checking Job deletion status: {e}')
                        raise
                time.sleep(check_interval)
            LOG.warning(f'Timeout while waiting for Kubernetes Job "{self.deployment_name}" to be deleted.')

        def _expose_deployment(self):
            api_instance = k8s.client.CoreV1Api()
            service = k8s.client.V1Service(
                api_version='v1',
                kind='Service',
                metadata=k8s.client.V1ObjectMeta(name=f'service-{self.deployment_name}'),
                spec=k8s.client.V1ServiceSpec(
                    selector={'app': self.deployment_name},
                    ports=[k8s.client.V1ServicePort(port=self.deployment_port, target_port=self.deployment_port)],
                    type='ClusterIP'
                )
            )
            try:
                api_instance.create_namespaced_service(
                    namespace=self.namespace,
                    body=service
                )
                LOG.info(f'Kubernetes Service "service-{self.deployment_name}" created and exposed successfully.')
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when creating Service: {e}')
                raise

        def _delete_service(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            svc_instance = k8s.client.CoreV1Api()
            service_name = f'service-{self.deployment_name}'
            try:
                svc_instance.delete_namespaced_service(
                    name=service_name,
                    namespace=self.namespace,
                    body=k8s.client.V1DeleteOptions(propagation_policy='Foreground')
                )
                LOG.info(f'Kubernetes Service "{service_name}" deleted.')

                if wait_for_completion:
                    self._wait_for_service_deletion(service_name=service_name,
                                                    timeout=timeout,
                                                    check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f'Kubernetes Service "{service_name}" already deleted.')
                else:
                    LOG.error(f'Exception when deleting Kubernetes Service: {e}')
                    raise

        def _wait_for_service_deletion(self, service_name, timeout, check_interval):
            svc_instance = k8s.client.CoreV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    svc_instance.read_namespaced_service(name=service_name, namespace=self.namespace)
                    LOG.info(f'Waiting for kubernetes Service "{service_name}" to be deleted...')
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f'Kubernetes Service "{service_name}" successfully deleted.')
                        return
                    else:
                        LOG.error(f'Error while checking Service deletion status: {e}')
                        raise
                time.sleep(check_interval)
            LOG.warning(f'Timeout while waiting for kubernetes Service "{service_name}" to be deleted.')

        def _create_or_update_gateway(self):
            networking_api = k8s.client.CustomObjectsApi()
            gateway_spec = {
                'apiVersion': 'gateway.networking.k8s.io/v1beta1',
                'kind': 'Gateway',
                'metadata': {
                    'name': self.gateway_name,
                    'namespace': self.namespace,
                    'annotations': {
                        'networking.istio.io/service-type': self.svc_type
                    }
                },
                'spec': {
                    'gatewayClassName': self.gateway_class_name,
                    'listeners': [
                        {
                            'name': f'httproute-{self.deployment_name}',
                            'port': self.deployment_port,
                            'protocol': 'HTTP',
                        }
                    ]
                }
            }

            try:
                existing_gateway = networking_api.get_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='gateways',
                    name=self.gateway_name
                )

                existing_gateway['spec']['listeners'].extend(gateway_spec['spec']['listeners'])
                networking_api.replace_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='gateways',
                    name=self.gateway_name,
                    body=existing_gateway
                )
                LOG.info(f'Kubernetes Gateway "{self.gateway_name}" updated successfully.')
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    try:
                        networking_api.create_namespaced_custom_object(
                            group='gateway.networking.k8s.io',
                            version='v1beta1',
                            namespace=self.namespace,
                            plural='gateways',
                            body=gateway_spec
                        )
                        LOG.info(f'Kubernetes Gateway "{self.gateway_name}" created successfully.')
                    except k8s.client.rest.ApiException as e_create:
                        LOG.error(f'Exception when creating Gateway: {e_create}')
                        raise
                else:
                    LOG.error(f'Exception when updating Gateway: {e}')
                    raise

        def _delete_or_update_gateway(self, wait_for_completion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            gateway_instance = k8s.client.CustomObjectsApi()
            try:
                gateway = gateway_instance.get_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='gateways',
                    name=self.gateway_name
                )

                listeners = gateway['spec']['listeners']
                gateway['spec']['listeners'] = [
                    listener for listener in listeners if listener['name'] != f'httproute-{self.deployment_name}'
                ]

                if gateway['spec']['listeners']:
                    gateway_instance.replace_namespaced_custom_object(
                        group='gateway.networking.k8s.io',
                        version='v1beta1',
                        namespace=self.namespace,
                        plural='gateways',
                        name=self.gateway_name,
                        body=gateway
                    )
                    LOG.info(f'Kubernetes Gateway "{self.gateway_name}" deleted updated.')

                    if wait_for_completion:
                        self._wait_for_gateway_update(timeout=timeout, check_interval=check_interval)
                else:
                    gateway_instance.delete_namespaced_custom_object(
                        group='gateway.networking.k8s.io',
                        version='v1beta1',
                        namespace=self.namespace,
                        plural='gateways',
                        name=self.gateway_name
                    )
                    LOG.info(f'Kubernetes Gateway "{self.gateway_name}" deleted.')

                    if wait_for_completion:
                        self._wait_for_gateway_deletion(timeout=timeout, check_interval=check_interval)
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f'Gateway "{self.gateway_name}" already deleted.')
                else:
                    LOG.error(f'Exception when deleting or updating Gateway: {e}')
                    raise

        def _wait_for_gateway_deletion(self, timeout, check_interval):
            gateway_instance = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    gateway_instance.get_namespaced_custom_object(
                        group='gateway.networking.k8s.io',
                        version='v1beta1',
                        namespace=self.namespace,
                        plural='gateways',
                        name=self.gateway_name
                    )
                    LOG.info(f'Waiting for Gateway "{self.gateway_name}" to be deleted...')
                except k8s.client.rest.ApiException as e:
                    if e.status == 404:
                        LOG.info(f'Gateway "{self.gateway_name}" successfully deleted.')
                        return
                    else:
                        LOG.error(f'Error while checking Gateway deletion status: {e}')
                        raise
                time.sleep(check_interval)
            LOG.warning(f'Timeout while waiting for Gateway "{self.gateway_name}" to be deleted.')

        def _wait_for_gateway_update(self, timeout, check_interval):
            gateway_instance = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    gateway_instance.get_namespaced_custom_object(
                        group='gateway.networking.k8s.io',
                        version='v1beta1',
                        namespace=self.namespace,
                        plural='gateways',
                        name=self.gateway_name
                    )
                    LOG.info(f'Gateway "{self.gateway_name}" status check passed.')
                    return
                except k8s.client.rest.ApiException as e:
                    LOG.error(f'Error while checking Gateway update status: {e}')
                    raise
                time.sleep(check_interval)
            LOG.warning(f'Timeout while waiting for Gateway "{self.gateway_name}" update.')

        def _create_httproute(self):
            custom_api = k8s.client.CustomObjectsApi()

            httproute_name = f'httproute-{self.deployment_name}'
            httproute_spec = {
                'apiVersion': 'gateway.networking.k8s.io/v1beta1',
                'kind': 'HTTPRoute',
                'metadata': {
                    'name': httproute_name,
                    'namespace': self.namespace
                },
                'spec': {
                    'parentRefs': [{
                        'name': self.gateway_name,
                        'port': self.deployment_port,
                        'sectionName': httproute_name
                    }],
                    'rules': [{
                        'matches': [{
                            'path': {
                                'type': 'PathPrefix',
                                'value': self.path
                            }
                        }],
                        'backendRefs': [{
                            'name': f'service-{self.deployment_name}',
                            'port': self.deployment_port
                        }]
                    }]
                }
            }

            if self.host:
                httproute_spec['spec']['hostnames'] = [self.host]

            try:
                custom_api.create_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='httproutes',
                    body=httproute_spec
                )
                LOG.info(f'Kubernetes HTTPRoute "{httproute_name}" created successfully.')
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when creating HTTPRoute: {e}')
                raise

        def _delete_httproute(self, wait_for_deletion=True, timeout=60, check_interval=5):
            k8s.config.load_kube_config(self._launcher.kube_config_path)
            httproute_instance = k8s.client.CustomObjectsApi()
            httproute_name = f'httproute-{self.deployment_name}'
            try:
                httproute_instance.delete_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='httproutes',
                    name=httproute_name
                )
                LOG.info(f'Kubernetes HTTPRoute "{httproute_name}" delete initiated.')
            except k8s.client.rest.ApiException as e:
                if e.status == 404:
                    LOG.info(f'HTTPRoute "{httproute_name}" already deleted.')
                    return
                else:
                    LOG.error(f'Exception when deleting HTTPRoute: {e}')
                    raise

            if wait_for_deletion:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        httproute_instance.get_namespaced_custom_object(
                            group='gateway.networking.k8s.io',
                            version='v1beta1',
                            namespace=self.namespace,
                            plural='httproutes',
                            name=httproute_name
                        )
                        LOG.info(f'Waiting for HTTPRoute "{httproute_name}" to be deleted...')
                    except k8s.client.rest.ApiException as e:
                        if e.status == 404:
                            LOG.info(f'HTTPRoute "{httproute_name}" successfully deleted.')
                            return
                        else:
                            LOG.error(f'Error while checking HTTPRoute status: {e}')
                            raise
                    time.sleep(check_interval)
                LOG.warning(f'Timeout while waiting for HTTPRoute "{httproute_name}" to be deleted.')

        def _start(self, *, fixed=False):
            cmd = self.get_executable_cmd(fixed=fixed)
            if self.launch_type == 'inference':
                self._create_deployment(cmd=cmd)
                self._expose_deployment()
                if self.on_gateway:
                    self._create_or_update_gateway()
                    self._create_httproute()
            else:
                self._create_job(cmd=cmd)

            self.jobid = self._get_jobid()
            self._launcher.all_processes[self._launcher._id].append((self.jobid, self))
            ret = self.wait()
            LOG.info(ret)

        def stop(self):
            if self.launch_type == 'inference':
                if self.on_gateway:
                    self._delete_or_update_gateway()
                    self._delete_httproute()
                self._delete_service()
                self._delete_deployment()
            else:
                self._delete_job()

        def _get_jobid(self):
            return f'service-{self.deployment_name}' if self.launch_type == 'inference' \
                else f'job-{self.deployment_name}'

        def _get_gateway_service_name(self):
            core_api = k8s.client.CoreV1Api()
            try:
                services = core_api.list_namespaced_service(namespace=self.namespace)

                for service in services.items:
                    labels = service.metadata.labels
                    if labels and ('gateway' in labels.get('app', '') or self.gateway_name in service.metadata.name):
                        LOG.info(f'Kubernetes Gateway service name: {service.metadata.name}')
                        return service.metadata.name

                LOG.warning('No Service was found corresponding to the specified Gateway.')
                return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when retrieving Gateway Service: {e}')
                return None

        def _get_gateway_deployment_name(self):  # noqa: C901
            core_api = k8s.client.CoreV1Api()
            apps_v1 = k8s.client.AppsV1Api()

            gateway_service_name = self._get_gateway_service_name()
            try:
                service = core_api.read_namespaced_service(gateway_service_name, self.namespace)
                selector = service.spec.selector
                if selector:
                    label_selector = ','.join(f'{k}={v}' for k, v in selector.items())
                    pods = core_api.list_namespaced_pod(self.namespace, label_selector=label_selector).items
                    if not pods:
                        LOG.warning(f'No Pods found for Service "{gateway_service_name}" in namespace '
                                    f'"{self.namespace}".')
                        return None

                    deployments = set()
                    for pod in pods:
                        for owner in pod.metadata.owner_references:
                            if owner.kind == 'ReplicaSet':
                                rs = apps_v1.read_namespaced_replica_set(owner.name, self.namespace)
                                for rs_owner in rs.metadata.owner_references:
                                    if rs_owner.kind == 'Deployment':
                                        deployments.add(rs_owner.name)

                    if deployments:
                        for deployment_name in deployments:
                            isRestart = False
                            deployment = apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
                            for container in deployment.spec.template.spec.containers:
                                if container.name == 'istio-proxy' and container.image_pull_policy == 'Always':
                                    container.image_pull_policy = 'IfNotPresent'
                                    isRestart = True
                            if isRestart:
                                apps_v1.replace_namespaced_deployment(name=deployment_name, namespace=self.namespace,
                                                                      body=deployment)
                                LOG.info(f'Updated {deployment_name} with imagePullPolicy "IfNotPresent"')
                        return list(deployments)
                    else:
                        LOG.warning(f'No Deployment found for Gateway "{self.gateway_name}" in namespace '
                                    f'"{self.namespace}".')
                        return None
                else:
                    LOG.warning(f'Kubernetes Service "{gateway_service_name}" does not have a selector.')
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Error fetching Service "{gateway_service_name}": {e}')
                return None

        def _get_gateway_ip(self):
            core_api = k8s.client.CoreV1Api()
            gateway_service_name = self._get_gateway_service_name()
            if gateway_service_name is None:
                raise ValueError('Kubernetes Gateway service name not found.')
            try:
                service = core_api.read_namespaced_service(
                    name=gateway_service_name,
                    namespace=self.namespace
                )

                if service.spec.type == 'LoadBalancer':
                    if service.status.load_balancer.ingress:
                        ip = service.status.load_balancer.ingress[0].ip
                        return ip
                    else:
                        LOG.warning('The LoadBalancer IP has not been assigned yet.')
                        return None
                elif service.spec.type == 'NodePort':
                    nodes = core_api.list_node()
                    node_ip = nodes.items[0].status.addresses[0].address
                    return node_ip
                elif service.spec.type == 'ClusterIP':
                    return service.spec.cluster_ip
                else:
                    LOG.warning('Unsupported Service type.')
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when retrieving gateway IP: {e}')
                return None

        def _get_httproute_host(self):
            custom_api = k8s.client.CustomObjectsApi()
            try:
                httproute = custom_api.get_namespaced_custom_object(
                    group='gateway.networking.k8s.io',
                    version='v1beta1',
                    namespace=self.namespace,
                    plural='httproutes',
                    name=f'httproute-{self.deployment_name}'
                )

                hostnames = httproute.get('spec', {}).get('hostnames', [])
                if hostnames:
                    return hostnames[0]
                else:
                    LOG.warning('Kubernetes HTTPRoute has no configured hostnames.')
                    return None
            except k8s.client.rest.ApiException as e:
                LOG.error(f'Exception when retrieving HTTPRoute host: {e}')
                return None

        def get_jobip(self):
            if not self.on_gateway: return f'service-{self.deployment_name}'
            host = self._get_httproute_host()
            ip = self._get_gateway_ip()
            LOG.info(f'gateway ip: {ip}, hostname: {host}')
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
                        LOG.info(f'Kubernetes Deployment "{self.deployment_name}" is running.')
                        return True
                    time.sleep(2)
                except k8s.client.rest.ApiException as e:
                    LOG.error(f'Exception when reading Deployment status: {e}')
                    raise
            LOG.warning(f'Timed out waiting for Deployment "{self.deployment_name}" to be ready.')
            return False

        def _is_service_ready(self, timeout):
            if self.on_gateway: return True
            url = f'http://service-{self.deployment_name}:{self.deployment_port}{self.path}'
            for i in range(self.gateway_retry):
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code != 503:
                        LOG.info(f'Kubernetes Service is ready at "{url}"')
                        self.queue.put(f'Uvicorn running on {url}')
                        return True
                    else:
                        LOG.info(f'Kubernetes Service at "{url}" returned status code {response.status_code}')
                except requests.RequestException as e:
                    LOG.error(f'Failed to access service at "{url}": {e}, retry: {i}/{self.gateway_retry}')
                    # raise
                time.sleep(timeout)

            self.queue.put(f'ERROR: Kubernetes Service failed to start on "{url}".')
            return False

        def _wait_for_service_ready(self, timeout=300):
            svc_instance = k8s.client.CoreV1Api()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    service = svc_instance.read_namespaced_service(
                        name=f'service-{self.deployment_name}',
                        namespace=self.namespace
                    )
                    if service.spec.type == 'LoadBalancer' and service.status.load_balancer.ingress:
                        ip = service.status.load_balancer.ingress[0].ip
                        LOG.info(f'Kubernetes Service "service-{self.deployment_name}" is ready with IP: {ip}')
                        return ip
                    elif service.spec.cluster_ip:
                        LOG.info(f'Kubernetes Service "service-{self.deployment_name}" is ready with ClusterIP: '
                                 f'{service.spec.cluster_ip}')
                        return service.spec.cluster_ip
                    elif service.spec.type == 'NodePort':
                        node_ports = [p.node_port for p in service.spec.ports]
                        if node_ports:
                            nodes = svc_instance.list_node()
                            for node in nodes.items:
                                for address in node.status.addresses:
                                    if address.type == 'InternalIP':
                                        node_ip = address.address
                                        LOG.info(f'Kubernetes Service "service-{self.deployment_name}" is ready on '
                                                 f'NodePort(s): {node_ports} at Node IP: {node_ip}')
                                        return {'ip': node_ip, 'ports': node_ports}
                                    elif address.type == 'ExternalIP':
                                        node_ip = address.address
                                        LOG.info(f'Kubernetes Service "service-{self.deployment_name}" is ready on '
                                                 f'NodePort(s): {node_ports} at External Node IP: {node_ip}')
                                        return {'ip': node_ip, 'ports': node_ports}
                    LOG.info(f'Kubernetes Service "service-{self.deployment_name}" is not ready yet. Retrying...')
                    time.sleep(2)
                except k8s.client.rest.ApiException as e:
                    LOG.error(f'Exception when reading Service status: {e}')
                    raise
            LOG.warning(f'Timed out waiting for Service "service-{self.deployment_name}" to be ready.')
            return None

        def wait_for_service_ready(self, timeout=300, interval=5):
            _service_ready = self._wait_for_service_ready(timeout=timeout)
            return _service_ready if _service_ready and self._is_service_ready(timeout=interval) else None

        def _is_gateway_ready(self, timeout):
            url = f'http://{self.get_jobip()}:{self.deployment_port}{self.path}'
            for _ in range(self.gateway_retry):
                try:
                    response = requests.get(url, timeout=timeout)
                    if response.status_code != 503:
                        LOG.info(f'Kubernetes Service is ready at "{url}"')
                        self.queue.put(f'Uvicorn running on {url}')
                        return True
                    else:
                        LOG.info(f'Kubernetes Service at "{url}" returned status code {response.status_code}')
                except requests.RequestException as e:
                    LOG.error(f'Failed to access service at "{url}": {e}')
                    raise
                time.sleep(timeout)

            self.queue.put(f'ERROR: Kubernetes Service failed to start on "{url}".')
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
                        if service.spec.type in ['NodePort', 'LoadBalancer']:
                            if service.spec.type == 'LoadBalancer':
                                if service.status.load_balancer.ingress:
                                    LOG.info(f'Kubernetes Service "{gateway_service_name}" is ready with '
                                             'LoadBalancer IP.')
                                    service_ready = True
                                else:
                                    LOG.info(f'Kubernetes Service "{gateway_service_name}" LoadBalancer IP '
                                             'not available yet.')
                                    service_ready = False
                            else:
                                if any(port.node_port for port in service.spec.ports):
                                    LOG.info(f'Kubernetes Service "{gateway_service_name}" is ready with '
                                             'NodePort configuration.')
                                    service_ready = True
                                else:
                                    LOG.info(f'Kubernetes Service "{gateway_service_name}" NodePort not assigned yet.')
                                    service_ready = False
                        else:
                            LOG.error(f'Unexpected Kubernetes Service type: {service.spec.type}.')
                            service_ready = False
                    except k8s.client.rest.ApiException as e:
                        LOG.error(f'Kubernetes Service "{gateway_service_name}" not found yet: {e}')
                        service_ready = False
                if not deployment_ready:
                    for deployment_name in gateway_deployment_names:
                        try:
                            deployment = apps_v1.read_namespaced_deployment(deployment_name, self.namespace)
                            if deployment.status.available_replicas and deployment.status.available_replicas > 0:
                                LOG.info(f'Kubernetes Deployment "{deployment_name}" is ready with '
                                         f'{deployment.status.available_replicas} replicas.')
                                deployment_ready = True
                                break
                            else:
                                LOG.info(f'Kubernetes Deployment "{deployment_name}" is not fully ready yet.')
                                deployment_ready = False
                        except k8s.client.rest.ApiException as e:
                            LOG.warning(f'Kubernetes Deployment "{deployment_name}" not found yet: {e}')
                            deployment_ready = False

                if service_ready and deployment_ready and self._is_gateway_ready(timeout=interval):
                    LOG.info(f'Kubernetes Gateway "{self.gateway_name}" is fully ready.')
                    return True

                time.sleep(interval)

            LOG.error(f'Kubernetes Gateway "{self.gateway_name}" failed to become ready with {timeout} seconds.')
            return False

        def wait_for_httproute(self, timeout=300):
            custom_api = k8s.client.CustomObjectsApi()
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    httproutes = custom_api.list_namespaced_custom_object(
                        group='gateway.networking.k8s.io',
                        version='v1beta1',
                        namespace=self.namespace,
                        plural='httproutes'
                    ).get('items', [])

                    for httproute in httproutes:
                        if httproute['metadata']['name'] == f'httproute-{self.deployment_name}':
                            LOG.info(f'Kubernetes HTTPRoute "httproute-{self.deployment_name}" is ready.')
                            return True
                    LOG.info(f'Waiting for HTTPRoute "httproute-{self.deployment_name}" to be ready...')
                except k8s.client.rest.ApiException as e:
                    LOG.error(f'Exception when checking HTTPRoute status: {e}')
                    raise

                time.sleep(2)
            LOG.warning(f'Timeout waiting for HTTPRoute "httproute-{self.deployment_name}" to be ready.')
            return False

        def wait(self):
            if self.launch_type == 'inference':
                deployment_ready = self.wait_for_deployment_ready()
                if not deployment_ready:
                    raise TimeoutError('Kubernetes Deployment did not become ready in time.')

                service_ip = self.wait_for_service_ready(interval=10)
                if not service_ip:
                    raise TimeoutError('Kubernetes Service did not become ready in time.')

                httproute_ready = True if not self.on_gateway else self.wait_for_httproute()
                if not httproute_ready:
                    raise TimeoutError('Kubernetes Httproute did not become ready in time.')

                gateway_ready = True if not self.on_gateway else self.wait_for_gateway()
                if not gateway_ready:
                    raise TimeoutError('Kubernetes Gateway did not become ready in time.')

                return {'deployment': Status.Running, 'service_ip': service_ip,
                        'gateway': Status.Running, 'httproute': Status.Running}
            else:
                return {'job': Status.Running}

        @property
        def status(self):
            if self.launch_type == 'inference':
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
                    LOG.error(f'Exception when reading Deployment status: {e}')
                    return Status.Failed
            else:
                api_instance = k8s.client.BatchV1Api()
                try:
                    job_status = api_instance.read_namespaced_job_status(
                        name=self.deployment_name,
                        namespace=self.namespace
                    ).status
                    if getattr(job_status, 'succeeded', 0) and job_status.succeeded >= 1:
                        return Status.Done
                    elif getattr(job_status, 'active', 0) and job_status.active >= 1:
                        return Status.Running
                    elif getattr(job_status, 'failed', 0) and job_status.failed >= 1:
                        return Status.Failed
                    else:
                        return Status.Pending
                except k8s.client.rest.ApiException as e:
                    LOG.error(f'Exception when reading Job status: {e}')
                    return Status.Failed

    def __init__(self, kube_config_path=None, volume_configs=None, image=None, resource_config=None,
                 namespace=None, on_gateway=None, gateway_name=None, gateway_class_name=None, host=None, path=None,
                 svc_type: Literal['LoadBalancer', 'NodePort', 'ClusterIP'] = None, retry=3,
                 sync=True, ngpus=None, **kwargs):
        super().__init__()
        self.gateway_retry = retry
        self.sync = sync
        self.ngpus = ngpus
        self.launch_type = kwargs.get('launch_type', 'inference')
        config_data = self._read_config_file(lazyllm.config['k8s_config_path']) if lazyllm.config['k8s_config_path'] \
            else {}
        self.volume_configs = volume_configs if volume_configs else config_data.get('volume', [])
        self.image = image if image else config_data.get('container_image', 'lazyllm/lazyllm:k8s_launcher')
        self.resource_config = resource_config if resource_config else config_data.get('resource', {})
        self.kube_config_path = kube_config_path if kube_config_path \
            else config_data.get('kube_config_path', '~/.kube/config')
        self.svc_type = svc_type if svc_type else config_data.get('svc_type', 'LoadBalancer')
        self.namespace = namespace if namespace else config_data.get('namespace', 'default')
        self.on_gateway = on_gateway if on_gateway else config_data.get('on_gateway', False)
        self.gateway_name = gateway_name if gateway_name else config_data.get('gateway_name', 'lazyllm-gateway')
        self.gateway_class_name = gateway_class_name if gateway_class_name \
            else config_data.get('gateway_class_name', 'istio')
        self.http_host = host if host else config_data.get('host', None)
        self.http_path = path if path else config_data.get('path', '/generate')

    def _read_config_file(self, file_path):
        assert os.path.isabs(file_path), 'Resource config file path must be an absolute path.'
        with open(file_path, 'r') as fp:
            try:
                data = yaml.safe_load(fp)
                return data
            except yaml.YAMLError as e:
                LOG.error(f'Exception when reading resource configuration file: {e}')
                raise ValueError('Kubernetes resource configuration file format error.')

    def makejob(self, cmd):
        # TODO(wangzhihong): support thread-local kube config by `client = config.new_client_from_config`
        k8s.config.load_kube_config(self.kube_config_path)
        return K8sLauncher.Job(cmd, launcher=self, sync=self.sync)

    def launch(self, f, *args, **kw):
        if isinstance(f, K8sLauncher.Job):
            f.start()
            LOG.info('Launcher started successfully.')
            self.job = f
            return f.return_value
        elif callable(f):
            LOG.info('Async execution in Kubernetes is not supported currently.')
            raise RuntimeError('Kubernetes launcher requires a Deployment object.')
