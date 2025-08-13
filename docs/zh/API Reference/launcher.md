::: lazyllm.LazyLLMLaunchersBase
    options:
      members:
      - makejob
      - launch
      - cleanup
      - wait
      - clone

::: lazyllm.launcher.EmptyLauncher
    options:
      heading_level: 3

::: lazyllm.launcher.RemoteLauncher
    options:
      heading_level: 3

::: lazyllm.launcher.SlurmLauncher
    options:
      heading_level: 3
      filters:
      - '!get_idle'

::: lazyllm.launcher.ScoLauncher
    options:
      heading_level: 3

::: lazyllm.launcher.Job
    options:
      heading_level: 3

::: lazyllm.launcher.K8sLauncher
    options:
      heading_level: 3
      members: [makejob, launch]