# 上下文感知AI销售助手

本教程将介绍如何使用LazyLLM框架实现一个上下文感知的AI销售助手，该助手能够根据对话阶段自动调整其行为和回复策略。

## 概述

SalesGPT是一个上下文感知的AI销售助手，它能够：

1. **识别对话阶段**：自动分析当前对话处于哪个销售阶段
2. **动态调整策略**：根据对话阶段调整回复内容和策略
3. **自然对话流程**：模拟真实的销售对话过程

## 架构设计

### 核心组件

1. **SalesStageAnalyzer（销售阶段分析器）**
   - 分析对话历史
   - 确定当前应该处于的销售阶段

2. **SalesConversationAgent（销售对话代理）**
   - 根据当前阶段生成相应的回复
   - 维护销售人员的身份和公司信息

3. **SalesGPT（主控制器）**
   - 协调各个组件
   - 管理对话历史和状态

### 销售对话阶段

系统定义了7个销售对话阶段：

1. **介绍阶段**：开始对话，介绍自己和公司
2. **资格确认**：确认潜在客户是否有购买决策权
3. **价值主张**：解释产品/服务的独特价值
4. **需求分析**：了解客户的需求和痛点
5. **解决方案展示**：展示产品作为解决方案
6. **异议处理**：处理客户的疑虑和异议
7. **成交**：提议下一步行动，促成交易

## 代码实现

### 1. 导入必要的库

```python
import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
```

### 2. 实现销售阶段分析器

```python
class SalesStageAnalyzer(ModuleBase):
    """销售对话阶段分析器，用于识别当前对话应该处于哪个销售阶段"""
    
    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        
        # 销售对话阶段定义
        self.conversation_stages = {
            "1": "介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。",
            "2": "资格确认：确认潜在客户是否是合适的人选，确保他们有购买决策权。",
            "3": "价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。",
            "4": "需求分析：通过开放式问题了解潜在客户的需求和痛点。",
            "5": "解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。",
            "6": "异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。",
            "7": "成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。"
        }
        
        # 阶段分析提示模板
        stage_analyzer_prompt = """你是一个销售助手，帮助销售代理确定销售对话应该进入哪个阶段。

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
不要回答其他任何内容。"""
        
        self.prompter = ChatPrompter(instruction=stage_analyzer_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)
    
    def forward(self, conversation_history: str) -> str:
        """分析对话历史并返回当前应该处于的销售阶段"""
        response = self.llm({"conversation_history": conversation_history})
        stage_id = ''.join(filter(str.isdigit, response.strip()))
        if not stage_id or int(stage_id) not in range(1, 8):
            stage_id = "1"  # 默认返回介绍阶段
        
        if self.verbose:
            print(f"对话阶段分析结果: {stage_id} - {self.conversation_stages[stage_id]}")
        
        return stage_id
```

### 3. 实现销售对话代理

```python
class SalesConversationAgent(ModuleBase):
    """销售对话代理，根据当前阶段生成相应的回复"""
    
    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        
        # 销售对话提示模板
        sales_conversation_prompt = """你叫{salesperson_name}，是一名{salesperson_role}。
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
{salesperson_name}："""
        
        self.prompter = ChatPrompter(instruction=sales_conversation_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)
    
    def forward(self, salesperson_name: str, salesperson_role: str, company_name: str, 
                company_business: str, company_values: str, conversation_purpose: str,
                conversation_type: str, conversation_stage: str, conversation_history: str) -> str:
        """生成销售对话回复"""
        response = self.llm({
            "salesperson_name": salesperson_name,
            "salesperson_role": salesperson_role,
            "company_name": company_name,
            "company_business": company_business,
            "company_values": company_values,
            "conversation_purpose": conversation_purpose,
            "conversation_type": conversation_type,
            "conversation_stage": conversation_stage,
            "conversation_history": conversation_history
        })
        
        display_response = response.replace('<END_OF_TURN>', '').strip()
        if self.verbose:
            print(f"{salesperson_name}: {display_response}")
        
        return response
```

### 4. 实现主控制器

```python
class SalesGPT(ModuleBase):
    """上下文感知的AI销售助手主控制器"""
    
    def __init__(self, llm: ModuleBase, salesperson_name: str = "张销售",
                 salesperson_role: str = "业务发展代表",
                 company_name: str = "优质睡眠",
                 company_business: str = "优质睡眠是一家高端床垫公司...",
                 company_values: str = "优质睡眠的使命是通过提供最佳的睡眠解决方案...",
                 conversation_purpose: str = "了解他们是否希望通过购买高端床垫来改善睡眠质量。",
                 conversation_type: str = "电话",
                 verbose: bool = True):
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
        self.current_conversation_stage = "1"
        
        # 初始化组件
        self.stage_analyzer = SalesStageAnalyzer(llm, verbose=verbose)
        self.sales_conversation_agent = SalesConversationAgent(llm, verbose=verbose)
        
        # 对话阶段定义
        self.conversation_stages = {
            "1": "介绍阶段：开始对话，介绍自己和公司。保持礼貌和专业的语调。",
            "2": "资格确认：确认潜在客户是否是合适的人选，确保他们有购买决策权。",
            "3": "价值主张：简要解释产品/服务如何使潜在客户受益，突出独特卖点。",
            "4": "需求分析：通过开放式问题了解潜在客户的需求和痛点。",
            "5": "解决方案展示：基于潜在客户的需求，展示产品/服务作为解决方案。",
            "6": "异议处理：处理潜在客户对产品/服务的任何异议，提供证据支持。",
            "7": "成交：通过提议下一步行动来要求销售，如演示、试用或与决策者会面。"
        }
    
    def seed_agent(self):
        """初始化销售代理"""
        if self.verbose:
            print(f"{self.salesperson_name}: (等待用户输入...)")
    
    def determine_conversation_stage(self):
        """确定当前对话应该处于的阶段"""
        conversation_text = "\n".join(self.conversation_history)
        self.current_conversation_stage = self.stage_analyzer(conversation_text)
        if self.verbose:
            print(f"当前对话阶段: {self.conversation_stages[self.current_conversation_stage]}")
    
    def human_step(self, human_input: str):
        """处理人类输入"""
        human_input = human_input + "<END_OF_TURN>"
        self.conversation_history.append(human_input)
        if self.verbose:
            print(f"用户: {human_input.replace('<END_OF_TURN>', '')}")
    
    def step(self):
        """执行销售代理的一步对话"""
        conversation_text = "\n".join(self.conversation_history)
        
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

### 5. 主函数实现

```python
def main():
    """主函数：演示SalesGPT的使用"""
    print("=== 上下文感知AI销售助手演示 ===\n")
    
    # 设置LLM
    llm = lazyllm.OnlineChatModule()
    
    # 配置销售代理
    config = {
        "salesperson_name": "李销售",
        "salesperson_role": "业务发展代表",
        "company_name": "优质睡眠",
        "company_business": "优质睡眠是一家高端床垫公司，为客户提供最舒适和支撑性的睡眠体验。",
        "company_values": "优质睡眠的使命是通过提供最佳的睡眠解决方案来帮助人们获得更好的睡眠。",
        "conversation_purpose": "了解他们是否希望通过购买高端床垫来改善睡眠质量。",
        "conversation_type": "电话"
    }
    
    # 创建销售代理
    sales_agent = SalesGPT(llm, **config)
    
    # 初始化代理
    print("初始化销售代理...")
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()
    
    # 开始对话循环
    print("\n开始销售对话...")
    while True:
        sales_agent.step()
        
        user_input = input("\n请输入您的回复 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            print("对话结束，感谢使用！")
            break
        
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()

if __name__ == "__main__":
    main()
```

### 实际测试

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


## 核心特性

### 1. 上下文感知

系统能够根据对话历史自动识别当前应该处于的销售阶段，并相应地调整回复策略。

### 2. 模块化设计

使用LazyLLM的ModuleBase类实现模块化设计，便于扩展和维护。

### 3. 灵活配置

支持自定义销售人员信息、公司信息、对话目的等参数。

### 4. 自然对话流程

模拟真实的销售对话过程，包括介绍、需求分析、价值主张、异议处理、成交等阶段。

## 扩展建议

1. **多轮对话优化**：可以添加更复杂的对话状态管理
2. **个性化推荐**：基于客户需求提供个性化的产品推荐
3. **情感分析**：集成情感分析功能，更好地理解客户情绪
4. **多模态支持**：支持语音、图像等多模态输入
5. **数据持久化**：保存对话历史和分析结果

## 总结

本教程展示了如何使用LazyLLM框架构建一个上下文感知的AI销售助手。通过模块化设计和阶段化管理，系统能够模拟真实的销售对话过程，为潜在客户提供个性化的服务体验。

这种实现方式不仅适用于销售场景，也可以扩展到其他需要上下文感知的对话系统中，如客服、咨询、教育等领域。 