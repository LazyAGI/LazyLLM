# LazyLLM
[中文](README.md) |  [EN](README.ENG.md)

## 一、定位

LazyLLM作为一款协助开发者构建多模态的大语言模型的一站式落地的工具，它基于商汤大装置AI专家服务团队在与客户沟通场景和交付客户的经验，贯穿了从数据处理、训练、微调、部署、推理、评测、交付等开发过程中的各个环节。本工具集成了在构建大模型项目的[各个环节](#三能力)中我们认为有价值的工具，并定义了在多个[典型业务场景](#四典型业务场景)下的标准作业程序(Standard Operating Procedure, SOP)。本工具建议被作为顶层工具来使用，不建议作为元素被集成到其他工具内被使用。

## 二、开始使用

### 2.1 工作流

LazyLLM定义了工作流(flow)，用来串联各个环节，以搭建自己的应用程序。目前框架支持的工作流有Pipeline和Parallel。

#### Pipeline
Pipeline中会包含若干个元素，每个元素会被认为一个环节，每个环节顺序执行，上一环节的输出会作为下一环节的输入。Pipeline中可以加入PostAction，Pipeline的输出会给到PostAction执行一些额外的代码，但PostAction的输出不会给到下一级。其工作流可以视作:
```
input -> module1 -> module2 -> ... -> moduleN -> output
                                              \> post-action
```

#### Parallel
Parallel中会包含若干个元素，每个元素分别执行，Parallel的输入会给到每一个元素，各个元素的输出会合并后作为Parallel的输出。其工作流可以视作:
```
      /> module11 -> ... -> module1N -> out1 \
input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
      \> module31 -> ... -> module3N -> out3 /
```

#### 返回值
流中的每个元素原则上只能有一个返回值，返回多个会被认为是tuple。当确实需要返回多个值时，可以使用`lazyllm.package`对返回值进行打包，当`lazyllm.package`流转到下一个可执行的功能时，会自动解包并传入到不同的形参中去。

### 2.2 基本的例子
LazyLLM通过pipeline把基本的元素组合起来，构成完整的工作流程。下面展示一个基本的使用例子:
```python
import lazyllm
ppl = lazyllm.pipeline(
    lazyllm.parallel(
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1'),
            post_action=deploy.lightllm,
        ),
        lazyllm.pipeline(
            finetune.alpacalora(base_model='./base-model2', target_path='./finetune-target2'),
            deploy.lightllm,
        ),
    ),
)
ppl.run('trainset', 'evalset')
```

### 2.3 自定义函数
LazyLLM支持自定义函数并且注册到对应的功能模块中，供pipeline去使用，其对应的接口为`lazyllm.llmregister`。支持被注册的功能模块有dataproc、finetune、deploy、validate等，但考虑到某些模块的复杂性，这里不建议用户自行注册finetune、deploy。如果想注册一个函数，则可以给函数加上`@lazyllm.llmregister`; 否则如果想注册一个bash执行的命令，则可以写一个返回bash命令的函数，给函数加上`@lazyllm.llmregister.cmd`。下面给出一个具体的例子：

```python
import lazyllm

@lazyllm.llmregister('dataproc')
def gen_data():
    return package('trainset', 'evalset')

@lazyllm.llmregister.cmd('validate')
def val1(in1, in2):
    return 'echo 0'

ppl = lazyllm.pipeline(
    dataproc.gen_data,
    finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1'),
    deploy.lightllm('url'),
    validate.val1
)
ppl.start()
```

* 注：未注册的函数也可以被pipeline使用，但不支持设置launcher等参数。该参数会在[跨平台](#26-跨平台)一节详细描述。

### 2.4 灵活传参

流式的搭建模型虽然方便，但它存在着“下一环节的输入只能是上一环节的输出”的问题，这使得开发者在搭建工程的时候不那么灵活。我们引入了“索引”和“参数绑定”的机制来解决这个问题。下面给出一个具体的例子:
```python
import lazyllm
from lazyllm import bind, root, _0

@lazyllm.llmregister('dataproc')
def gen_data():
    return package('trainset', 'evalset')

@lazyllm.llmregister('validate')
def val1(in1, in2):
    print(in1, in2)
    return in1


@lazyllm.llmregister('validate')
def val2(in1, in2):
    print(in1, in2)

ppl = lazyllm.pipeline(
    pp1=lazyllm.pipeline(
        proc=dataproc.gen_data,
        val1=validate.val1,
    ), 
    val2=bind(validate.val2, root.pp1.proc, _0),
)
ppl.start()
```

### 2.5 查看结构

在[灵活传参](#24-灵活传参)一节中，我们介绍了可以通过`root.xx.yy.zz`对暂未实例化的pipeline的节点进行索引，那么了解整个pipeline的结构成为了关键的一步。这里我们重写了各个对象的`__repr__`函数，使得我们能够展示pipeline的层级结构。

```python
import lazyllm
from lazyllm import bind, root, _0

@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    return package(idx + 1, idx + 1)

@lazyllm.llmregister('validate')
def eval(evalset, url, job=None):
    print(f'eval all. evalset: {evalset}, url: {url}, eval_all done. job: {job}')

named_ppl = lazyllm.pipeline(
    data=dataproc.gen_data(),
    finetune=lazyllm.parallel(
        stage1=lazyllm.pipeline(
            sft=finetune.alpacalora(base_model='./base-model1', target_path='./finetune-target1', launcher=launchers.slurm()),
            deploy=deploy.lightllm('http://www.myserver1.com'),
        ),
    ),
    val=bind(validate.eval, 'evalset', _0, root.finetune.stage1.deploy.deploy_stage2.job),
)
```

可以使用`named_ppl.__repr__()`函数查看层级结构。 在实际调试的时候，也可以通过bash打开一个python交互程序来输出，假设上述文件命名为test.pu，示例如下：
```bash
$ python
>>> from test import named_ppl
>>> named_ppl
<Pipeline> [
    data <lazyllm.llm.core.dataproc.gen_data>,
    finetune <Parallel> [
        stage1 <Pipeline> [
            sft <lazyllm.llm.core.finetune.AlpacaloraFinetune>,
            deploy <Lightllm> [
                deploy_stage4 <function show_io at 0x7fdedb3ed1e0>,
                deploy_stage2 <lazyllm.llm.core.deploy.lllmserver>(bind args:placeholder._0, 1, 64000, 2),
                deploy_stage3 <lazyllm.llm.core.deploy.RelayServer>,
                deploy_stage4 <function show_io at 0x7fdedb3ed1e0>
            ]
        ]
    ],
    val <lazyllm.llm.core.validate.eval>(bind args:'evalset', placeholder._0, <lazyllm.common.AttrTree object at 0x7fdeda955da0>)
]
```
如上面的示例，每一个元素会给出名称和<类型>，如果是流结构，还会额外用`[]`给出其内部包含的元素。如果是蕴含了参数绑定的元素，则会展示其绑定的真实元素，并且通过`(bind args: xx)`展示其绑定的参数。

### 2.6 跨平台

该工具支持运行在多种集群环境上，包括裸金属、slurm和sensecore。在实际搭建的时候，主要通过给模块提供launcher这个参数来决定该模块运行在哪个平台上，可选的launcher有empty、slurm和sensecore。一般情况下，一个程序内不会同时出现slurm和sensecore，这是因为目前没有一个管理节点可以同时把任务提交到slurm和sco上。

```python
ppl = lazyllm.pipeline(
    finetune.alpacalora(base_model='./base-model', target_path='./finetune-target', launcher=launchers.slurm),
    deploy.lightllm('http://www.myserver1.com', launcher=launchers.slurm(ngpus=8)),
    post_action=validate.eval_stage1(launcher=launchers.empty),
)
```

如果launcher是slurm和sensecore的话，则调用的模块必须是一个bash命令；但我们支持将一些可执行的函数透传到新的bash进程中，例如在部署时候，可能要对大模型的输入/输出做前后处理，此时我们支持将前后处理函数定义在主进程，然后通过bash启动的推理的脚本能读到该函数并执行。

* P1-TODO: 未来可能会同时支持linux和windows，假如有客户需要

### 2.7 运行模式

LazyLLM提供了三种运行模式，分别是Display、Normal和Debug。

#### Display
在Display模式下，所有的cmd命令都不会真正的被执行，而是通过命令行打印出来。例如在[2.5 查看结构](#25-查看结构)中所述的案例，在Display模式下执行`named_ppl.start(0)`的结果为：
```bash
idx 0: gen data done
Command: srun -p None -N 1 --job-name=x59b29882ced265a9 -n1 bash -c 'python /mnt/cache/wangzhihong/lazyllm/lazyllm/llms/finetune/alpaca-lora/finetune.py --base_model=./base-model1 --output_dir=./finetune-target1 --data_path=1 --batch_size=64 --micro_batch_size=4 --num_epochs=2 --learning_rate=0.0005 --cutoff_len=1030 --filter_nums=1024 --val_set_size=200 --lora_r=8 --lora_alpha=32 --lora_dropout=0.05 --lora_target_modules="[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]" --modules_to_save="[word_embeddings, output_layer]" --deepspeed="ds.json" --prompt_with_background=True --train_on_inputs=True 2>&1 | tee ./finetune-target1/LLM_$(date +"%Y-%m-%d_%H-%M-%S").log'
input or output is: ./finetune-target1
Command: srun -p None -N 1 --job-name=x3b5eca3122db7596 -n1 bash -c 'python -m lightllm.server.api_server --model_dir ./finetune-target1 --tp 1 --nccl_port 20864 --max_total_token_num 64000 --tokenizer_mode "auto" --port 35444 --host "0.0.0.0" --eos_id 2 --trust_remote_code '
Command: srun -p None -N 1 --job-name=4771159e863eadf0 -n1 bash -c 'python test2.py --target_url=http://x3b5eca3122db7596:35444/generate --before_function="b'\x80\x04N.' --after_function="b'\x80\x04N.'"'
input or output is: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631af28>
eval all. evalset: evalset, url: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631af28>, eval_all done. job: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631aef0(Readonly)>
```

#### Debug
在Debug模式下，会给出更加细致的错误信息。

### 2.8 全局配置

## 三、能力

本工具覆盖开发过程中的各个环节，从数据处理、训练、微调、部署、推理、评测，直到最终交付，在每个环境都包含了大量的称手的工具，并提供统一且简单的使用方式，让开发者可以轻松的使用这些工具。

### 3.1 数据处理

- [ ] 预置各种数据生成的模板。
- [ ] 支持自定义数据预处理。

### 3.2 继续预训练

- [ ] 支持多种模型，包括llama、llama2、internLM、chatglm等；支持其常见的尺寸，包括7B、13B、20B、70B等
- [ ] 支持读取huggingface格式的checkpoint

### 3.3 微调

#### 支持模型
支持多种模型，包括llama、llama2、internLM、chatglm等；支持其常见的尺寸，包括7B、13B、20B、70B等

#### 支持算法
支持多种微调算法，考虑LoRA、QLoRA等

#### 支持框架
支持多种微调框架，包括peft、easyllm、colle等。

#### Auto
Auto是LazyLLM主推的使用方式，它秉承着“让懒惰进行到底”的思想，支持用户选定基模型(hf格式)和训练集之后，根据提供的机器情况，结合过往的经验自动选择框架认为最优的算法、超参数和框架。使用方式如下：
```python
finetune.auto('chatglm3-6b', launcher=launchers.slurm(ngpus=32))
```
#### 国产芯片支持

### 3.4 部署

- [ ] 支持多种模型，包括llama、llama2、internLM、chatglm等；支持其常见的尺寸，包括7B、13B、20B、70B等
- [ ] 提供API和简单页面两种方式用于推理
- [ ] 支持部署前后处理服务
- [ ] 支持负载均衡
- [ ] 支持在国产芯片上部署推理服务
- [ ] 只支持一个basemodel在同一个进程中可以和多个不同的lora-model结合进行推理

### 3.5 推理

- [x] 支持单独推理和batch推理
- [x] 支持多轮对话式的交互式推理

### 3.6 评测

- [ ] 针对典型场景，预置该场景常用的评测方式和报告输出
- [ ] 集成典型功能下的常见评测算法

### 3.7 交付

- [ ] 支持源码交付、镜像交付等多种交付方式。
- [ ] 自动打包源码和模型。

## 四、典型业务场景

### 4.1 文档问答（Document QA）
### 4.2 AI Agent客服对话
### 4.3 报告生成
### 4.4 搜索增强
### 4.5 ...

## 五、场景工具依赖

|场景           |数据处理|训练|微调|部署|推理|评测|交付|
|--------------|-------|---|---|---|----|---|----|
|文档问答        |      |   |   |   |    |    |   |
|AI Agent客服对话|      |   |   |   |    |    |   |
|报告生成        |      |   |   |   |    |    |   |
|搜索增强        |      |   |   |   |    |    |   |

## 六、界面

## 七、路线 RoadMap

- [ ] 把基础的微调和推理的能力加上，把继续预训练也集成进去
- [ ] 然后打磨再一些细节，比如优化报错体验、用户做全局配置、自动查找空余节点等等；
- [ ] 把更多的微调/推理框架也加进去
- [ ] 支持用户选择模型和算法，之后我们根据用户的环境情况，自动选择最优的框架和参数；
- [ ] 针对典型的场景，加入其需要的工具，例如Document QA需要加入文本解析、知识库构建、数据库等
- [ ] 加带界面的web服务，如对话系统、知识库系统等等
