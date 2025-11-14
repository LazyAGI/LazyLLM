# 上下文感知 AI 销售助手

本示例展示了如何使用 LazyLLM 快速构建一个具备销售对话理解与阶段感知能力的智能销售助手（Sales Assistant）。

该助手能够根据对话上下文自动判断当前销售阶段，并生成符合语境的专业销售回复。

!!! abstract "通过本节，您将学习如何构建一个上下文感知型销售助手，包括以下要点："

    - 如何使用 [OnlineChatModule][lazyllm.module.OnlineChatModule] 构建销售对话的核心语言理解与生成模块。
    - 如何实现 [SalesStageAnalyzer] 自动识别销售对话阶段，如“介绍”“需求分析”“异议处理”等。
    - 如何实现 [SalesConversationAgent] 根据阶段与历史上下文生成自然的销售话术。
    - 如何使用 [SalesGPT] 作为主控制器，实现“阶段分析 → 回复生成 → 对话更新”的完整逻辑循环。
    - 如何通过主函数 `main()` 启动一个交互式销售对话演示，实现端到端的销售模拟体验。

## 设计思路

要构建一个能够根据上下文动态调整话术的智能销售助手，我们需要让系统既“懂销售逻辑”，又“能理解对话语境”。

因此，整个 SalesGPT 的设计围绕“对话阶段识别 + 智能销售应答”两大核心目标展开。

首先，我们使用 LazyLLM 的 `OnlineChatModule` 作为核心语言模型，用于理解对话内容并生成自然、专业的销售回应。

然后，将系统拆分为两个关键模块：

- `SalesStageAnalyzer`：分析当前对话处于销售流程的哪个阶段（如介绍、需求挖掘、异议处理等）；
- `SalesConversationAgent`：根据阶段信息和对话历史生成相应回复，并在句末添加 `<END_OF_TURN>` 用于回合控制。

最后，通过主控制器 `SalesGPT` 将两者结合：它负责维护对话上下文、判断阶段、生成回复，形成一个完整的“分析 → 生成 → 更新”的循环流程。

整体流程如下：

![sales_assistant](../assets/sales_assistant.png)

## 环境准备

### 安装依赖

在使用前，请先执行以下命令安装所需库：

```bash
pip install lazyllm
```

### 导入依赖包

```python
from lazyllm import OnlineChatModule
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
```

### 环境变量

在流程中会使用到在线大模型，您需要设置 API 密钥（以 Qwen 为例）：

```bash
export LAZYLLM_QWEN_API_KEY = "sk-******"
```

> ❗ 注意：平台的 API_KEY 申请方式参考[官方文档](docs.lazyllm.ai/)。

## 代码实现

### 销售阶段分析器

在销售对话中，不同的阶段代表着销售人员与客户关系的不同发展程度。

例如，开场介绍阶段注重建立信任，而后期的“异议处理”或“成交”阶段则更关注说服与行动引导。

因此，我们需要一个模块来自动识别当前对话所处的销售阶段，从而让系统在合适的时机输出最符合上下文的回应。

下面的 `SalesStageAnalyzer` 类正是为此设计的。

```python
class SalesStageAnalyzer(ModuleBase):
    '''销售对话阶段分析器，用于识别当前对话应该处于哪个销售阶段'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # 销售对话阶段定义
        self.conversation_stages = {
            '1': '介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。',
            '2': '资格确认：确认潜在客户是否是合适的人选，确保他们有购买决策权。',
            '3': '价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。',
            '4': '需求分析：通过开放式问题了解潜在客户的需求和痛点。',
            '5': '解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。',
            '6': '异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。',
            '7': '成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。'
        }

        # 阶段分析提示模板
        stage_analyzer_prompt = '''你是一个销售助手，帮助销售代理确定销售对话应该进入哪个阶段。

        以下是对话历史：
        ===
        {conversation_history}
        ===

        现在根据对话历史确定销售代理在销售对话中的下一个即时对话阶段，从以下选项中选择：
        1. 介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。
        2. 资格确认：确认潜在客户是否是合适的人选，确保他们有购买决策权。
        3. 价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。
        4. 需求分析：通过开放式问题了解潜在客户的需求和痛点。
        5. 解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。
        6. 异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。
        7. 成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。

        只回答1到7之间的数字，表示对话应该继续的阶段。答案必须只是一个数字，不要添加任何其他内容。
        如果没有对话历史，输出1。
        不要回答其他任何内容。'''

        self.prompter = ChatPrompter(instruction=stage_analyzer_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, conversation_history: str) -> str:
        '''分析对话历史并返回当前应该处于的销售阶段'''
        response = self.llm({'conversation_history': conversation_history})
        stage_id = ''.join(filter(str.isdigit, response.strip()))
        if not stage_id or int(stage_id) not in range(1, 8):
            stage_id = '1'  # 默认返回介绍阶段

        if self.verbose:
            print(f'对话阶段分析结果: {stage_id} - {self.conversation_stages[stage_id]}')

        return stage_id
```

**上下文缓存 / 会话隔离**

LazyLLM 内部支持“模块级上下文缓存”机制。

通过使用 `.used_by()` 方法，系统会将同一个 LLM 实例区分为不同的“调用者（caller）”：

- 不同模块可以独立缓存各自的 prompt–response 结果；
- 系统可根据模块 ID 维护独立的会话状态，实现上下文隔离。

> 💡 拓展：`ModuleBase`  是 LazyLLM 框架的核心基类，所有模块（如对话生成器、阶段分析器、知识检索器等）都基于它构建。
> 它定义了统一的接口和核心能力，抽象了训练、部署、推理与评估的通用逻辑，并提供模块级的管理机制。
> 更多详情见[官方 API 文档](https://docs.lazyllm.ai/en/stable/API%20Reference/module/#lazyllm.module.ModuleBase)。

### 销售对话代理

在确定了销售阶段之后，我们需要一个“智能销售代理”来生成与该阶段匹配的自然语言回复。
这个模块的目标是让 AI 模拟真实的销售人员，根据上下文和阶段自动生成下一句合理的销售话术。

```python
class SalesConversationAgent(ModuleBase):
    '''销售对话代理，根据当前阶段生成相应的回复'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # 销售对话提示模板
        sales_conversation_prompt = '''你叫{salesperson_name}，是一名{salesperson_role}。
        你在{salesperson_name}公司工作。{company_name}的业务是：{company_business}
        公司价值观是：{company_values}
        你联系潜在客户的目的是：{conversation_purpose}
        你联系潜在客户的方式是：{conversation_type}

        如果被问及从哪里获得用户的联系信息，请说从公开记录中获得。
        保持回复简短以保持用户的注意力。不要产生列表，只回答问题。
        你必须根据之前的对话历史和当前对话阶段来回应。
        每次只生成一个回复！生成完成后，以'<END_OF_TURN>'结尾，给用户回应的机会。

        当前对话阶段：{conversation_stage}
        对话历史：
        {conversation_history}
        {salesperson_name}：'''

        self.prompter = ChatPrompter(instruction=sales_conversation_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, salesperson_name: str, salesperson_role: str, company_name: str,
                company_business: str, company_values: str, conversation_purpose: str,
                conversation_type: str, conversation_stage: str, conversation_history: str) -> str:
        
        # 生成销售对话回复
        response = self.llm({
            'salesperson_name': salesperson_name,
            'salesperson_role': salesperson_role,
            'company_name': company_name,
            'company_business': company_business,
            'company_values': company_values,
            'conversation_purpose': conversation_purpose,
            'conversation_type': conversation_type,
            'conversation_stage': conversation_stage,
            'conversation_history': conversation_history
        })

        display_response = response.replace('<END_OF_TURN>', '').strip()
        if self.verbose:
            print(f'{salesperson_name}: {display_response}')

        return response
```

销售对话代理 `SalesConversationAgent` 的作用是：

- 接收当前销售阶段（如“需求分析”或“成交”）；
- 根据销售人员角色、公司背景、对话目的等信息；
- 自动生成一条自然、简短且符合销售逻辑的回复。

与阶段分析器相比，这个模块关注 “怎么说”，而不是 “到哪一步”。
两者配合使用，可以形成一个完整的销售对话闭环。

> 💡 提示：结合上一节的“销售阶段分析器”，可以让系统实现自动识别阶段 + 智能生成回复的完整销售模拟流程。

### 主控制器

在前两节中，我们分别实现了：

- 阶段分析器（`SalesStageAnalyzer`）：判断当前销售对话处于哪个阶段；
- 销售对话代理（`SalesConversationAgent`）：根据阶段生成符合语境的销售回复。

接下来，我们将它们整合到一个统一的主控制器中，构建一个上下文感知、可循环互动的销售 AI 助手。

```python
class SalesGPT(ModuleBase):
    '''上下文感知的AI销售助手主控制器'''

    def __init__(
        self,
        llm: ModuleBase,
        salesperson_name: str = '张销售',
        salesperson_role: str = '业务发展代表',
        company_name: str = '优质睡眠',
        company_business: str = '优质睡眠是一家高端床垫公司...',
        company_values: str = '优质睡眠的使命是通过提供最佳的睡眠解决方案...',
        conversation_purpose: str = '了解他们是否希望通过购买高端床垫来改善睡眠质量。',
        conversation_type: str = '电话',
        verbose: bool = True
    ):
        super().__init__()

        # 设置销售人员信息
        self.salesperson_name = salesperson_name
        self.salesperson_role = salesperson_role
        self.company_name = company_name
        self.company_business = company_business
        self.company_values = company_values
        self.conversation_purpose = conversation_purpose
        self.conversation_type = conversation_type
        self.verbose = verbose

        # 初始化状态
        self.conversation_history = []
        self.current_conversation_stage = '1'

        # 初始化组件
        self.stage_analyzer = SalesStageAnalyzer(llm, verbose=verbose)
        self.sales_conversation_agent = SalesConversationAgent(llm, verbose=verbose)

        # 对话阶段定义
        self.conversation_stages = {
            '1': '介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。',
            '2': '资格确认：确认潜在客户是否是合适的人选，确保他们有购买决策权。',
            '3': '价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。',
            '4': '需求分析：通过开放式问题了解潜在客户的需求和痛点。',
            '5': '解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。',
            '6': '异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。',
            '7': '成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。'
        }

    def seed_agent(self):
        '''初始化销售代理'''
        if self.verbose:
            print(f'{self.salesperson_name}: (等待用户输入...)')

    def determine_conversation_stage(self):
        '''确定当前对话应该处于的阶段'''
        conversation_text = '\n'.join(self.conversation_history)
        self.current_conversation_stage = self.stage_analyzer(conversation_text)
        if self.verbose:
            print(f'当前对话阶段: {self.conversation_stages[self.current_conversation_stage]}')

    def human_step(self, human_input: str):
        '''处理人类输入'''
        human_input = human_input + '<END_OF_TURN>'
        self.conversation_history.append(human_input)
        if self.verbose:
            print(f'用户: {human_input.replace('<END_OF_TURN>', '')}')

    def step(self):
        '''执行销售代理的一步对话'''
        conversation_text = '\n'.join(self.conversation_history)

        ai_message = self.sales_conversation_agent(
            salesperson_name=self.salesperson_name,
            salesperson_role=self.salesperson_role,
            company_name=self.company_name,
            company_business=self.company_business,
            company_values=self.company_values,
            conversation_purpose=self.conversation_purpose,
            conversation_type=self.conversation_type,
            conversation_stage=self.conversation_stages[self.current_conversation_stage],
            conversation_history=conversation_text
        )

        self.conversation_history.append(ai_message)
        return ai_message.replace('<END_OF_TURN>', '')
```

`SalesGPT` 负责管理销售对话的状态与流程控制，核心职责包括：

- 管理对话历史；
- 动态判断销售阶段；
- 让 AI 在每个阶段生成合适的销售回复；
- 实现人类输入与模型响应的交替互动。

换句话说，它就像一个“销售导演”，负责：
“让每个销售阶段自然衔接，让 AI 始终说对的话。”

### 主函数实现

现在展示如何通过主函数运行一个完整的 `SalesGPT` 智能销售助手。

主函数主要完成以下任务：

- 创建 LLM 实例：加载对话模型，用于生成销售话术与智能响应；
- 配置销售代理参数：定义销售员身份、公司信息和对话目标；
- 初始化销售代理：让模型具备基础背景知识和对话阶段判断能力；
- 启动交互循环：模拟真实的客户对话过程。

```python
def main():
    '''主函数：演示SalesGPT的使用'''
    print('=== 上下文感知AI销售助手演示 ===\n')

    # 设置LLM
    llm = OnlineChatModule()

    # 配置销售代理
    config = {
        'salesperson_name': '李销售',
        'salesperson_role': '业务发展代表',
        'company_name': '优质睡眠',
        'company_business': '优质睡眠是一家高端床垫公司，为客户提供最舒适和支撑性的睡眠体验。',
        'company_values': '优质睡眠的使命是通过提供最佳的睡眠解决方案来帮助人们获得更好的睡眠。',
        'conversation_purpose': '了解他们是否希望通过购买高端床垫来改善睡眠质量。',
        'conversation_type': '电话'
    }

    # 创建销售代理
    sales_agent = SalesGPT(llm, **config)

    # 初始化代理
    print('初始化销售代理...')
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()

    # 开始对话循环
    print('\n开始销售对话...')
    while True:
        sales_agent.step()
        user_input = input('\n请输入您的回复 (输入 \'quit\' 退出): ')
        if user_input.lower() == 'quit':
            print('对话结束，感谢使用！')
            break
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()

if __name__ == '__main__':
    main()
```

## 效果展示

下面我们将展示一下 `SalesGPT` 的实际运行效果。

```bash
=== 上下文感知AI销售助手演示 ===

初始化销售代理...
李销售: (等待用户输入...)
对话阶段分析结果: 1 - 介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。
当前对话阶段: 介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。你的问候应该是欢迎的。始终在问候中明确说明联系潜在客户的原因。

开始销售对话...
李销售: 您好！我是李销售，来自优质睡眠公司。我们专注于提供高端床垫和睡眠解决方案，帮助人们改善睡眠质量。请问您最近是否在寻找提升睡眠体验的方法呢？

请输入您的回复 (输入 'quit' 退出): 是
用户: 是
对话阶段分析结果: 4 - 需求分析：通过开放式问题了解潜在客户的需求和痛点。
当前对话阶段: 需求分析：通过开放式问题了解潜在客户的需求和痛点。仔细倾听他们的回应并做笔记。
李销售: 太好了，很高兴听到您对改善睡眠体验感兴趣。请问您目前主要面临哪些睡眠问题，或者您希望在睡眠体验上有哪些具体的提升呢？

请输入您的回复 (输入 'quit' 退出): 晚上睡不好，容易失眠
用户: 晚上睡不好，容易失眠
对话阶段分析结果: 4 - 需求分析：通过开放式问题了解潜在客户的需求和痛点。
当前对话阶段: 需求分析：通过开放式问题了解潜在客户的需求和痛点。仔细倾听他们的回应并做笔记。
李销售: 我明白了，晚上睡不好和容易失眠确实很影响生活质量。您是否尝试过使用某些特殊材质的床垫或枕头来改善这种情况呢？

请输入您的回复 (输入 'quit' 退出): 还没有
用户: 还没有
对话阶段分析结果: 3 - 价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。
当前对话阶段: 价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。
李销售: 那么，让我向您介绍我们的产品如何能够帮助到您。我们的床垫采用先进材料和设计，能够根据您的身体提供定制化的支撑，减少翻身次数，帮助您更快入睡并深度睡眠。此外，我们的枕头采用记忆泡沫，能够适应您的头部和颈部形状，进一步提高睡眠质量。这些产品已经帮助许多客户显著改善了他们的睡眠问题。您有兴趣了解更多吗？

请输入您的回复 (输入 'quit' 退出): 介绍一下你们的产品和价格
用户: 介绍一下你们的产品和价格
对话阶段分析结果: 5 - 解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。
当前对话阶段: 解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。
李销售: 当然可以。我们的产品线包括三种主要系列：经典系列、豪华系列和定制系列。

1. **经典系列**：适合追求舒适与支撑的客户。价格大约在5000元到10000元之间。

2. **豪华系列**：采用更高端的材料和先进技术，适合追求极致睡眠体验的客户。价格大约在10000元到20000元之间。

3. **定制系列**：根据您的具体需求和身体状况量身定制，提供最个性化的睡眠解决方案。价格从20000元起。

我们的床垫和枕头都提供一定期限的质保，确保您能够安心使用。如果您有兴趣，我们可以安排一次免费的睡眠咨询，以便更好地了解您的需求并提供更准确的推荐。您觉得怎么样？

请输入您的回复 (输入 'quit' 退出): 太贵了
用户: 太贵了
对话阶段分析结果: 6 - 异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。
当前对话阶段: 异议处理：处理潜在客户对产品/服务的任何异议，准备提供证据或推荐信支持你的说法。
李销售: 我理解您的担忧，高端床垫的价格确实比普通床垫要高一些。但考虑到我们的产品能够显著改善睡眠质量，减少失眠问题，从长远来看，这其实是一项非常值得的投资。优质的睡眠对健康和工作效率都有极大的正面影响，能够帮助您在日常生活和工作中表现得更好。

此外，我们提供免费试用期和质保服务，确保您对产品有充分的体验和信心。如果您愿意，我们可以安排一次免费的睡眠咨询，以便为您提供更个性化的建议。您觉得如何？

请输入您的回复 (输入 'quit' 退出): 可以
用户: 可以
对话阶段分析结果: 7 - 成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。
当前对话阶段: 成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。确保总结讨论内容并重申好处。
李销售: 太好了，我会立即为您安排一次免费的睡眠咨询。请问您本周哪一天方便？我们可以安排专业人员与您联系，进一步了解您的具体需求并提供个性化的建议。期待帮助您改善睡眠质量！

请输入您的回复 (输入 'quit' 退出): 明天吧
用户: 明天吧
对话阶段分析结果: 7 - 成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。
当前对话阶段: 成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。确保总结讨论内容并重申好处。
李销售: 非常感谢，我会为您安排明天进行睡眠咨询。我们的专业人员将会与您联系，确保您能获得最合适的睡眠解决方案。如果有任何其他问题或需要进一步的帮助，请随时告诉我。祝您今晚有一个好梦！

请输入您的回复 (输入 'quit' 退出): quit
对话结束，感谢使用！
```

## 总结

本教程展示了如何使用LazyLLM框架构建一个上下文感知的AI销售助手。通过模块化设计和阶段化管理，系统能够模拟真实的销售对话过程，为潜在客户提供个性化的服务体验。

这种实现方式不仅适用于销售场景，也可以扩展到其他需要上下文感知的对话系统中，如客服、咨询、教育等领域。
