## 环境依赖

### 依赖及场景说明

- 微调（基于alpaca-lora）: datasets, deepspeed, faiss-cpu, fire, gradio, numpy, peft, torch, transformers
- 微调（基于collie）: collie-lm, numpy, peft, torch, transformers, datasets, deepspeed, fire
- 推理（基于lightllm）: lightllm
- 推理（基于vllm）: vllm

### 基础依赖

- fastapi: FastAPI 是一个现代、快速（高性能）的Web框架，用于构建API，与Python 3.6+类型提示一起使用。
- loguru: Loguru 是一个Python日志库，旨在通过简洁、易用的API提供灵活的日志记录功能。
- pydantic: Pydantic 是一个数据验证和设置管理工具，它使用Python的类型注解来验证数据。
- Requests: Requests 是一个Python HTTP库，用于发送HTTP请求。它简单易用，是Python中发起Web请求的常用库。
- uvicorn: Uvicorn 是一个轻量级、快速的ASGI服务器，用于运行Python 3.6+的Web应用程序。
- cloudpickle: 一个Python序列化库，能够序列化Python对象到字节流，以便跨Python程序和解释器传输。
- flake8: 一个代码风格检查工具，用于检测Python代码中的错误，并遵守PEP 8编码标准。
- gradio: 一个用于快速创建简单Web界面的库，允许用户与Python模型进行交互。
- gradio_client: Gradio的客户端库，允许用户从远程服务器加载和使用Gradio界面。
- protobuf: Google的Protocol Buffers的Python实现，用于序列化结构化数据。
- setuptools: 一个Python包安装和分发工具，用于打包和分发Python应用程序和库。


## 在不同操作系统上安装

### windows

#### step 1: 安装git 
下载并安装：
https://github.com/git-for-windows/git/releases/download/v2.50.1.windows.1/Git-2.50.1-64-bit.exe

#### step 2: 安装python
官网：https://python.p2hp.com/downloads/
推荐： python3.10.9
1. 选择对应版本下载，安装时选择 Customize installation 自定义安装路径，勾选下面的加入PATH
!!! Note
    如果已经安装过可选择 uninstall 卸载后重新安装

![install_python](../assets/env/install_python.png)

2. 自定义安装路径为，可以设置为 D:\Python\Python310

![set_python_install_path](../assets/env/set_python_install_path.png)


#### step 3: 安装和使用VS Code
1. 下载vscode并安装
2. 安装python组件

![vscode_extensions](../assets/env/vscode_extensions.png)

3. 在vscode中随便打开一个python文件后，可在最下面选择python解释器

![vscode_interpret](../assets/env/vscode_interpret.png)

4. 默认会识别到所有的解释器，选择一个；或者手动输入两遍 D:\Python\Python310\python.exe

![vscode_interpret_manual](../assets/env/vscode_interpret_manual.png)

5. 终端中选用git bash 就可以使用类似 Linux 的命令行环境

![git_bash](../assets/env/git_bash.png)

#### step 4: 安装LazyLLM
1. 在终端中通过命令行安装lazyllm
```code
pip install lazyllm
```

2. 设置环境变量 key

在powershell中，通过如下代码设置
```code
$env:LAZYLLM_SENSENOVA_API_KEY = "7ACAxxxxxxxxxxxxxxx"
$env:LAZYLLM_SENSENOVA_SECRET_KEY = "2B0F7xxxxxxxxxxxxxxxx"
```

在bash中，通过如下代码设置
```code
export LAZYLLM_SENSENOVA_API_KEY="7ACACxxxxxxxxxxxxxxx"
export LAZYLLM_SENSENOVA_SECRET_KEY="2B0F72xxxxxxxxxxxxxx"
```

### windows with wsl

#### 前置条件
1. 查看内部版本,Win + r 输入winver 要求大于19041；否则需更新windows系统

![winversion](../assets/env/winversion.png)
![winversion2](../assets/env/winversion_2.png)

2. 打开任务管理器，确认cpu虚拟化开启。

![virtualize](../assets/env/virtualize.png)
![winversion2](../assets/env/virtualize_2.png)

如果没有的话，需打开，并重启电脑

![winversion3](../assets/env/virtualize_3.png)
![winversion4](../assets/env/virtualize_4.png)

#### 下载wsl2内核更新包
WSL 2 Linux内核更新包地址：https://aka.ms/wsl2kernel
下载好后，直接运行文件

#### 安装linux系统
1. 调出powershell 以管理员身份运行，然后查看在线商店下载的可用 Linux 分发版的列表
```code
PS C:\Users\name> wsl --list --online
以下是可安装的有效分发的列表。
请使用“wsl --install -d <分发>”安装。

NAME                            FRIENDLY NAME
Ubuntu                          Ubuntu
Debian                          Debian GNU/Linux
kali-linux                      Kali Linux Rolling
Ubuntu-18.04                    Ubuntu 18.04 LTS
Ubuntu-20.04                    Ubuntu 20.04 LTS
Ubuntu-22.04                    Ubuntu 22.04 LTS
Ubuntu-24.04                    Ubuntu 24.04 LTS
OracleLinux_7_9                 Oracle Linux 7.9
OracleLinux_8_10                Oracle Linux 8.10
OracleLinux_9_5                 Oracle Linux 9.5
openSUSE-Leap-15.6              openSUSE Leap 15.6
SUSE-Linux-Enterprise-15-SP6    SUSE Linux Enterprise 15 SP6
openSUSE-Tumbleweed             openSUSE Tumbleweed
```

2. 查看已安装的系统，（默认没有安装过）
```code
PS C:\Users\name> wsl --list --verbose
适用于 Linux 的 Windows 子系统没有已安装的分发版。
可以通过访问 Microsoft Store 来安装分发版:
https://aka.ms/wslstore
```

3. 安装指定系统
```code
PS C:\Users\name>  wsl --install -d Ubuntu-22.04
正在安装: Ubuntu 22.04 LTS
[=                          3.0%  
```

4. 安装完后要输入一个账密

![passward](../assets/env/wsl_passward.png)

5. 查看映射的本地路径
Win + r 输入 \\wsl$
点击Ubantu文件夹，右键，点击映射网络驱动器就可以添加到我的电脑里了，注意只有启动Ubantu之后才可以打开该磁盘。

![map](../assets/env/map.png)

#### 在vscode中使用wsl
1. 安装插件wsl
2. 终端打开wsl
3. 安装python和lazyllm

#### 本地命令行使用
直接搜索wsl，打开，即可进入子系统

### macOS

