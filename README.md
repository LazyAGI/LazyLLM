# LazyLLM
[中文](README.md) |  [EN](README.ENG.md)

## 一、定位

LazyLLM作为一款协助开发者构建多模态的大语言模型的一站式落地的工具，它基于商汤大装置AI专家服务团队在与客户沟通场景和交付客户的经验，贯穿了从数据处理、训练、微调、部署、推理、评测、交付等开发过程中的各个环节。本工具集成了在构建大模型项目的[各个环节](#三能力)中我们认为有价值的工具，并定义了在多个[典型业务场景](#四典型业务场景)下的标准作业程序(Standard Operating Procedure, SOP)。本工具建议被作为顶层工具来使用，不建议作为元素被集成到其他工具内被使用。

## 二、开始使用

### 2.1 模块

模块(Module)是LazyLLM中的顶层核心功能组件。LazyLLM提供了TrainableModule、UrlModule、ServerModule、WebModule等几类通用的基础模块。用户以搭积木的方式可轻松将不同模块组合起来搭建出自己的应用，值得注意的是每个LazyLLM模块往往对应了用户实际的业务功能模块，而且用户不用操心训练、部署等各种底层工程实现，应用搭建好后一键即可实现应用的微调、部署、推理和发布。这使得用户只用关心业务逻辑，而不用关心实现细节。

#### 2.1.1 基本模块

LazyLLM中的模块是一个可调用的基本功能单元，**可调用性**提供了数据流转的通路，**基本功能单元**提供了该模块所具有对数据处理的特定能力。

|基本模块|功能简介|
|:---:|:---|
|SequenceModule|可将其它模块按顺序依次调用执行|
|UrlModule|可将某个可访问的**URL**包装为一个模块|
|ActionModule|可包装lazyLLM更底层的各类**flow**到模块中，实现更精细的数据流转控制|
|ServerModule|可将任意可调用的对象(典型如：函数、LazyLLM的任意Module)包装为一个可通过URL访问的服务，并提供前后处理的接口|
|WebModule|可将任意可调用的对象包装为一个用户界面|
|TrainableModule|可包装一个LLM模型，并提供训练（包括继续预训练与微调）、部署、推理及评测的能力，其训练和推理框架都可以指定|

#### 2.1.2 基本例子

##### 使用说明
在使用lazyLLM前，请将lazyLLM的仓库路径添加到`PYTHONPATH`中，并根据实际运行环境情况设置如下环境变量：
```bash
# 设置默认提交任务的引擎：slurm、sco等
export LAZYLLM_DEFAULT_LAUNCHER=slurm
# 设置对应引擎可用的分区：
export LAZYLLM_SLURM_PART=pat_rd
```

##### TrainableModule 微调部署推理一键启
TrainableModule是一个集合了训练（继续预训练和微调训练）、部署、推理于一体的Module。
```python
m = lazyllm.TrainableModule('path/to/base/model', 'path/to/target/file').finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode(
    'finetune')
m.evalset([1, 2, 3, 4, 5, 6])
m.update()
print(m.eval_result)
```
注意：
- `TrainableModule`需传入两个参数路径：基模型的路径和存储的路径；
- 指定假的微调引擎: `finetune.dummy`
- 指定假的推理引擎：`deploy.dummy`
- 设置训练的模式为微调: `finetune`

输出：
```bash
['reply for 1', 'reply for 2', 'reply for 3', 'reply for 4', 'reply for 5', 'reply for 6']
```

##### ServerModule 包装任意可调用对象为服务
ServerModule 可将任意可调用对象包装为服务，这个可调用对象可以是函数、LazyLLM的各类Module等。被包装后的可调用对象可以通过ServerModule对象直接调用来传入数据（见下面例子），或者向其暴露的URL发起请求来传入数据（见UrlModule的例子第一个服务）。

这里将一个函数包含为一个服务：

```python
def func(x):
    return str(x)+' after'
m = lazyllm.ServerModule(func)
m.evalset([1, 2, 3, 4, 5, 6])
m.update()
print(m.eval_result)
```
输出：
```bash
['1 after', '2 after', '3 after', '4 after', '5 after', '6 after']
```

##### UrlModule 包装任意URL为Module
往往用户可能只有一个可发送请求的URL，这时就可以将这个URL包装为一个Module来集成到应用中。
```python
# 启动一个服务，以提供一个可访问的URL
def func(x):
    return str(x)+' after'
m1 = lazyllm.ServerModule(func)
m1.update()

# 使用UrlModule 将刚生成的URL包装为一个Module。
m2 = lazyllm.Module.url(m1._url)
m2.update()
res  = m2('hi')
print('Got Response: ',res)
```
输出：
```bash
Got Response:  hi after
```

##### WebModule 包装任意可调用对象为客户端
WebModule 可将任意可调用的对象包装为一个客户端。
例如包装一个函数：
```python
def func(x):
    return 'reply ' + x
m = lazyllm.WebModule(func)
m.update()
```

例如包装一个LazyLLM的Module:
```python
mm = lazyllm.TrainableModule(stream=True).finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune')
m = lazyllm.WebModule(mm)
m.update()
```

输出：
```bash
Running on local URL:  http://0.0.0.0:20566
```
点击URL可在本地访问给出的客户端。

##### ActionModule 包装一个flow为Module
ActionModule 可将LazyLLM中的底层flow包装为一个Module实现更精细的数据控制流程。LazyLLM中常见的flow见第二部分工作流。

这里包装LazyLLM中的一个flow类：pipeline
```python
from lazyllm import pipeline
def func1(x):
    return str(x) + ' func1 '
mm = lazyllm.TrainableModule('path1/to/base/model', 'path1/to/target/file').finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune')
m = lazyllm.Module.action(pipeline(func1, mm)) # lazyllm.ActionModule == lazyllm.Module.action
m.evalset([1, 2, 3, 4, 5, 6])
m.update()
print(m.eval_result)
```
输出：
```bash
['reply for 1 func1 ', 'reply for 2 func1 ', 'reply for 3 func1 ', 'reply for 4 func1 ', 'reply for 5 func1 ', 'reply for 6 func1 ']
```

##### SequenceModule 将多个Module串联起来
SequenceModule 可以将LazyLLM中的多个Module串联起来构成一个Module:
```python
m = lazyllm.SequenceModule(
    lazyllm.TrainableModule('path1/to/base/model', 'path1/to/target/file').finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune'),
    lazyllm.TrainableModule('path2/to/base/model', 'path2/to/target/file').finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune'),
)
m.evalset([1, 2, 3, 4, 5, 6])
m.update()
print(m.eval_result)
```
输出：
```python
['reply for reply for 1', 'reply for reply for 2', 'reply for reply for 3', 'reply for reply for 4', 'reply for reply for 5', 'reply for reply for 6']
```

##### 三分钟搞定业务应用

现在让我们搭建一个应用：这个应用基于一个大模型，对外实现一个对话窗口。用户在窗口中输入信息，经过前处理后送给大模型，大模型生成相关内容，然后再进行后处理，最后将后处理的结果返回给客户端进行展示。
其中大模型的微调和部署引擎这里都用dummy的以方便演示。

```python
# 包装一个大模型：
LLM = lazyllm.TrainableModule('path1/to/base/model', 'path1/to/target/file').finetune_method(finetune.dummy).deploy_method(deploy.dummy).mode('finetune')

# 将LLM套入到一个带前后处理的服务中：
def pre_func(x):
    return str(x)+' Add Backgrounds'
def post_func(x):
    return 'After proccess ' + x
PrePost = lazyllm.ServerModule(LLM, pre=pre_func, post=post_func)

# 把该服务包装成一个带客户端的应用：
Chat = lazyllm.WebModule(PrePost)

# 一键启动模型的微调、推理和部署，等待最后给出访问的网址：
Chat.update()
```


### 2.2 工作流

LazyLLM定义了工作流(flow)，用来串联各个环节，以搭建自己的应用程序。目前框架支持的工作流有Pipeline和Parallel。

#### 2.2.1 基本工作流

##### Pipeline
Pipeline中会包含若干个元素，每个元素会被认为一个环节，每个环节顺序执行，上一环节的输出会作为下一环节的输入。Pipeline中可以加入PostAction，Pipeline的输出会给到PostAction执行一些额外的代码，但PostAction的输出不会给到下一级。其工作流可以视作:
```
input -> module1 -> module2 -> ... -> moduleN -> output
                                              \> post-action
```

##### Parallel
Parallel中会包含若干个元素，每个元素分别执行，Parallel的输入会给到每一个元素，各个元素的输出会合并后作为Parallel的输出。其工作流可以视作:
```
      /> module11 -> ... -> module1N -> out1 \
input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
      \> module31 -> ... -> module3N -> out3 /
```

##### Parallel.sequential
Parallel.sequential(parallel in dataflow, serial in executing)数据流转和Parallel一致，但执行顺序是依次执行，不同于Parallel是并行执行的。其工作流如下：
```
      /> module11 -> ... -> module1N -> out1 \
input -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
      \> module31 -> ... -> module3N -> out3 /
```

##### Diverter
Diverter要求输入是LazyLLM中的package类型，且package的大小要与分支数量一致，不同分支是不同的元素序列。这样package中每个元素被传递给不同的分支，最后将所有分支的输出合并为一个package。其工作流如下：
```
                  /> in1 -> module11 -> ... -> module1N -> out1 \
  (in1, in2, in3) -> in2 -> module21 -> ... -> module2N -> out2 -> (out1, out2, out3)
                  \> in3 -> module31 -> ... -> module3N -> out3 /
```


##### Warp
Warp要求输入是LazyLLM中的package类型，它们会被同时送入同一个分支的元素序列，最后将每个输入对应的结果合并为一个package。其工作流如下：
```
                  /> in1 \                            /> out1 \
  (in1, in2, in3) -> in2 -> module1 -> ... -> moduleN -> out2 -> (out1, out2, out3)
                  \> in3 /                            \> out3 /
```

##### switch
switch要求输入是个package, 如果只含有一输入，那么该输入会作为条件判断和元素的输入；如果包含两个输入，那么第一个输入会作为条件判断的输入，第二个输入会作为元素的输入。其工作流如下：
```
     case cond1: input -> module11 -> ... -> module1N -> out; break
     case cond2: input -> module21 -> ... -> module2N -> out; break
     case cond3: input -> module31 -> ... -> module3N -> out; break
```

##### IFS
IFS会先将输入送入到可调用逻辑判断函数，根据返回True和False来选择将输入送入到对应的分支中进行处理。其工作流如下：
```
result = cond(input) ? tpath(input) : fpath(input)
```

##### Loop
Loop会把前一个元素的输出送往下一个元素的输入，最后一个元素的输出又作为第一个元素的输入，以此循环往复。跳出循环的条件是用户可以配置Loop循环的个数，或者配置一个可调用的条件对象来对每次元素的输出数据进行判断。
```
in(out) -> module1 -> ... -> moduleN -> exp, out -> out
```

##### 返回值
流中的每个元素原则上只能有一个返回值，返回多个会被认为是tuple。当确实需要返回多个值时，可以使用`lazyllm.package`对返回值进行打包，当`lazyllm.package`流转到下一个可执行的功能时，会自动解包并传入到不同的形参中去。

#### 2.2.2 基本的例子
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

#### 2.2.3 自定义函数
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

* 注：未注册的函数也可以被pipeline使用，但不支持设置launcher等参数。该参数会在[跨平台](#23-跨平台)一节详细描述。

#### 2.2.4 灵活传参

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

#### 2.2.5 查看结构

在[灵活传参](#224-灵活传参)一节中，我们介绍了可以通过`root.xx.yy.zz`对暂未实例化的pipeline的节点进行索引，那么了解整个pipeline的结构成为了关键的一步。这里我们重写了各个对象的`__repr__`函数，使得我们能够展示pipeline的层级结构。

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
                deploy_stage1 <function show_io at 0x7fdedb3ed1e0>,
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

### 2.3 跨平台

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

### 2.4 运行模式

LazyLLM提供了三种运行模式，分别是Display、Normal和Debug。

#### Display
在Display模式下，所有的cmd命令都不会真正的被执行，而是通过命令行打印出来。例如在[2.2.5 查看结构](#225-查看结构)中所述的案例，在Display模式下执行`named_ppl.start(0)`的结果为：
```bash
idx 0: gen data done
Command: srun -p None -N 1 --job-name=x59b29882ced265a9 -n1 bash -c 'python /mnt/cache/wangzhihong/lazyllm/lazyllm/llms/finetune/alpaca-lora/finetune.py --base_model=./base-model1 --output_dir=./finetune-target1 --data_path=1 --batch_size=64 --micro_batch_size=4 --num_epochs=2 --learning_rate=0.0005 --cutoff_len=1030 --filter_nums=1024 --val_set_size=200 --lora_r=8 --lora_alpha=32 --lora_dropout=0.05 --lora_target_modules="[query_key_value,dense,dense_4h_to_h,dense_h_to_4h]" --modules_to_save="[word_embeddings, output_layer]" --deepspeed="ds.json" --prompt_template_name=alpaca --train_on_inputs=True 2>&1 | tee ./finetune-target1/LLM_$(date +"%Y-%m-%d_%H-%M-%S").log'
input or output is: ./finetune-target1
Command: srun -p None -N 1 --job-name=x3b5eca3122db7596 -n1 bash -c 'python -m lightllm.server.api_server --model_dir ./finetune-target1 --tp 1 --nccl_port 20864 --max_total_token_num 64000 --tokenizer_mode "auto" --port 35444 --host "0.0.0.0" --eos_id 2 --trust_remote_code '
Command: srun -p None -N 1 --job-name=4771159e863eadf0 -n1 bash -c 'python test2.py --target_url=http://x3b5eca3122db7596:35444/generate --before_function="b'\x80\x04N.' --after_function="b'\x80\x04N.'"'
input or output is: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631af28>
eval all. evalset: evalset, url: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631af28>, eval_all done. job: <lazyllm.launcher.SlurmLauncher.Job object at 0x7f0ee631aef0(Readonly)>
```

#### Debug
在Debug模式下，会给出更加细致的错误信息。

### 2.5 全局配置

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
