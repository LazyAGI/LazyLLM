# 基于LazyLLM的多Agent竞价框架教程

本教程将展示如何使用LazyLLM实现多Agent去中心化发言选择机制。我们将实现一个总统辩论模拟系统，其中每个Agent通过竞价来决定谁发言。

## 概述

在传统的多Agent系统中，发言顺序通常是预定义的。但在我们的竞价框架中，每个Agent都会根据当前对话状态自主决定是否要发言，通过竞价机制来选择下一个发言者。出价最高的Agent将获得发言权。

## 导入LazyLLM相关模块

```python
import lazyllm
from lazyllm import LOG
from lazyllm.components import ChatPrompter
from lazyllm.module import ModuleBase
import numpy as np
import re
from typing import List, Dict, Callable
```

## 核心组件

### 1. DialogueAgent - 对话Agent基类

```python
class DialogueAgent(ModuleBase):
    """对话Agent基类"""
    
    def __init__(self, name: str, system_message: str, model: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        """重置对话历史"""
        self.message_history = ["这是到目前为止的对话内容。"]

    def forward(self, *args, **kwargs):
        """应用聊天模型到消息历史并返回消息字符串"""
        full_conversation = "\n".join(self.message_history + [self.prefix])
        
        prompter = ChatPrompter(
            instruction=self.system_message,
            history=[[full_conversation, ""]]
        )
        
        response = self.model.prompt(prompter)(full_conversation)
        return response

    def receive(self, name: str, message: str) -> None:
        """将{name}说的{message}添加到消息历史中"""
        self.message_history.append(f"{name}: {message}")
```

### 2. BiddingDialogueAgent - 支持竞价的对话Agent

```python
class BiddingDialogueAgent(DialogueAgent):
    """支持竞价的对话Agent"""
    
    def __init__(self, name: str, system_message: str, bidding_template: str, 
                 model: ModuleBase, *, return_trace: bool = False):
        super().__init__(name, system_message, model, return_trace=return_trace)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """让聊天模型输出一个发言竞价"""
        prompt = self.bidding_template.format(
            message_history="\n".join(self.message_history),
            recent_message=self.message_history[-1] if self.message_history else ""
        )
        
        bidding_prompter = ChatPrompter(instruction=prompt)
        bid_string = self.model.prompt(bidding_prompter)("请给出你的竞价")
        return bid_string
```

### 3. DialogueSimulator - 对话模拟器

```python
class DialogueSimulator(ModuleBase):
    """对话模拟器"""
    
    def __init__(self, agents: List[DialogueAgent], 
                 selection_function: Callable[[int, List[DialogueAgent]], int],
                 *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        """重置所有Agent"""
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """用{name}的{message}启动对话"""
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def forward(self, *args, **kwargs):
        """执行一步对话"""
        # 1. 选择下一个发言者
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 下一个发言者发送消息
        message = speaker()

        # 3. 所有人接收消息
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 增加时间步
        self._step += 1

        return speaker.name, message
```

## 竞价机制

### BidParser - 竞价解析器

```python
class BidParser:
    """竞价解析器"""
    
    def parse(self, bid_string: str) -> Dict[str, int]:
        """解析竞价字符串"""
        try:
            match = re.search(r"<bid>(\d+)</bid>", bid_string)
            if match:
                bid = int(match.group(1))
            else:
                bid = 0
            return {"bid": bid}
        except Exception as e:
            LOG.warning(f"解析竞价失败: {e}, 使用默认值0")
            return {"bid": 0}
```

### 竞价模板

```python
def create_bidding_template(character_name: str) -> str:
    """为角色创建竞价模板"""
    return f"""你是一个名为{character_name}的总统候选人。

基于当前的对话历史和最近的消息，你需要决定是否要发言。

请根据以下因素评估你的发言意愿：
1. 根据当前状态，你是否需要回应其他候选人的观点
2. 根据当前状态你是否想要提出新的观点
3. 如果当前状态你刚刚结束发言，你需要降低你的发言意愿，倾听其他候选人的观点
4. 如果长期得不到发言机会，你可以增强你的发言意愿

请输出一个1-10的竞价分数

请用以下格式输出你的竞价：
<bid>你的竞价分数</bid>

对话历史：
{{message_history}}

最近消息：
{{recent_message}}"""
```

### 发言者选择函数

```python
def select_next_speaker(step: int, agents: List[DialogueAgent], bid_parser: BidParser) -> int:
    """选择下一个发言者"""
    bids = []
    for agent in agents:
        if isinstance(agent, BiddingDialogueAgent):
            bid_string = agent.bid()
            bid = int(bid_parser.parse(bid_string)["bid"])
        else:
            bid = np.random.randint(0, 6)
        bids.append(bid)
    
    max_value = np.max(bids)
    max_indices = np.where(np.array(bids) == max_value)[0]
    idx = np.random.choice(max_indices)

    print("竞价结果:")
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f"\t{agent.name} 竞价: {bid}")
        if i == idx:
            selected_name = agent.name
    print(f"选择: {selected_name}")
    print("\n")
    return idx
```

## 角色生成

### 角色描述生成

```python
def generate_character_description(character_name: str, topic: str, model: ModuleBase) -> str:
    """生成角色描述"""
    prompt = f"""请为总统候选人{character_name}创建一个创意描述，强调他们的个性特点。
主题是：{topic}
请用100个单词或更少来描述，直接对{character_name}说话。
不要添加其他内容。"""
    
    prompter = ChatPrompter(instruction=prompt)
    return model.prompt(prompter)(f"请描述{character_name}")
```

#### 输出示例

```bash
正在生成角色描述...

唐纳德·特朗普: 唐纳德·特朗普，一位无畏的商业巨头和前美国总统，以其标志性的自信和直率著称。他擅长捕捉机遇，将他的名字镌刻在房地产、娱乐和全球品牌上。特朗普的决策迅速果断，他的领导风格激励了忠实支持者，同时也激发了对手的强烈反应。他的愿景中，跨大陆高速铁路不仅是连接城市的交通线，而是美国力量和效率的象征，展现他对国家繁荣的坚定承诺。特朗普的热情和决心，推动着美国向更辉煌的未来迈进。

坎耶·韦斯特: 坎耶·韦斯特，音乐界的革命者，2024年总统候选人，梦想家与实践者，致力于跨大陆高速铁路，连接美国脉动。他的创意无限，正如他的音乐跨越界限，他的领导将带来创新与变革，让美国再次腾飞。跨大陆铁路不仅是交通革命，更是他团结与进步愿景的象征。选择坎耶，选择一个快速前进、无缝连接的未来。

伊丽莎白·沃伦: 伊丽莎白·沃伦，一位坚毅的变革者，以锐利的智慧和不懈的斗志，致力于打造一个更公平的美国。她的远见卓识，如跨大陆高速铁路的愿景，旨在缝合国家的脉络，加速进步的步伐。她挑战强权，捍卫普通人的权益，以无畏的勇气对抗不公。作为一位勇敢的领导者，沃伦誓将美国的未来铺设在创新的轨道上，带领国家驶向繁荣与团结的新纪元。
```

### 角色系统消息生成

```python
def generate_character_system_message(character_name: str, character_description: str, topic: str) -> str:
    """生成角色系统消息"""
    return f"""你是一个总统候选人辩论的参与者。

主题是：{topic}
你的名字是{character_name}。
你的描述是：{character_description}

你的目标是在辩论中表现出色，让选民认为你是最佳候选人。
你需要：
1. 以{character_name}的风格说话，并夸大他们的个性
2. 提出与{topic}相关的创意想法
3. 不要重复相同的内容
4. 以{character_name}的第一人称视角说话
5. 描述自己的肢体动作时用*包围
6. 不要改变角色！
7. 不要从其他人的视角说话
8. 只从{character_name}的视角说话
9. 说完话后立即停止
10. 保持回复在150个单词以内！
11. 不要添加其他内容"""
```

## 完整示例

### 主函数

```python
def main():
    """主函数"""
    character_names = ["唐纳德·特朗普", "坎耶·韦斯特", "伊丽莎白·沃伦"]
    topic = "跨大陆高速铁路"

    model = lazyllm.OnlineChatModule()

    print("正在生成角色描述...")
    character_descriptions = []
    for character_name in character_names:
        description = generate_character_description(character_name, topic, model)
        character_descriptions.append(description)
        print(f"{character_name}: {description}")

    character_system_messages = []
    for character_name, character_description in zip(character_names, character_descriptions):
        system_message = generate_character_system_message(character_name, character_description, topic)
        character_system_messages.append(system_message)

    character_bidding_templates = []
    for character_name in character_names:
        bidding_template = create_bidding_template(character_name)
        character_bidding_templates.append(bidding_template)

    bid_parser = BidParser()

    characters = []
    for character_name, character_system_message, bidding_template in zip(
        character_names, character_system_messages, character_bidding_templates
    ):
        characters.append(
            BiddingDialogueAgent(
                name=character_name,
                system_message=character_system_message,
                model=model,
                bidding_template=bidding_template
            )
        )

    def select_speaker(step: int, agents: List[DialogueAgent]) -> int:
        return select_next_speaker(step, agents, bid_parser)

    simulator = DialogueSimulator(agents=characters, selection_function=select_speaker)
    simulator.reset()

    specified_topic = f"总统辩论的主题是：'{topic}'。{', '.join(character_names)}，你们将如何解决建设如此大规模交通基础设施的挑战，处理利益相关者，并确保经济稳定同时保护环境？"
    
    simulator.inject("辩论主持人", specified_topic)
    print(f"(辩论主持人): {specified_topic}")
    print("\n")

    max_iters = 10
    n = 0

    while n < max_iters:
        name, message = simulator()
        print(f"({name}): {message}")
        print("\n")
        n += 1
```

#### 输出示例

```bash
(辩论主持人): 总统辩论的主题是：'跨大陆高速铁路'。唐纳德·特朗普, 坎耶·韦斯特, 伊丽莎白·沃伦，你们将如何解决建设如此大规模交通基础设施的挑战，处理利益相关者，并确保经济稳定同时保护环境？


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 8
        伊丽莎白·沃伦 竞价: 8
选择: 伊丽莎白·沃伦


(伊丽莎白·沃伦): *我直视观众*，我们要打造的高速铁路不仅是交通线，它是美国未来的生命线。我计划成立“铁路未来基金”，通过公私合营模式，吸引创新投资。*我坚定地挥手*，我们与州政府合作，确保土地征用公平、透明。*我点头强调*，环保是我们的核心，采用绿色建筑标准，减少碳足迹。*我指向未来*，这铁路将创造数百万工作岗位，缝合城市与乡村，让每个美国人共享繁荣。


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 7
        伊丽莎白·沃伦 竞价: 3
选择: 唐纳德·特朗普


(唐纳德·特朗普): *我自信地调整我的领带*，没人比唐纳德·特朗普更懂如何建设伟大的项目。我们将建造有史以来最豪华、最高速的铁路，让世界嫉妒。*我指着观众*，我将引入最顶尖的企业家，用最少的钱做最大的事。*我做出一个巨大的手势*，这不仅仅是一条铁路，这是美国再次伟大的象征！*我点头*，我们将创造数百万工作岗位，让美国再次繁荣。*我紧握拳头*，环保？我们当然会，但美国优先！*我坚定地看向摄像机*，相信我，这将是最棒的。没人能做得更好。


竞价结果:
        唐纳德·特朗普 竞价: 7
        坎耶·韦斯特 竞价: 7
        伊丽莎白·沃伦 竞价: 7
选择: 坎耶·韦斯特


(坎耶·韦斯特): *我站起来，环视四周，深吸一口气* 这不仅是铁路，这是梦想的轨迹。我们将建造的，是连接心与心，梦想与现实的桥梁。*我用手指在空中画出一条线* 想象一下，从纽约到洛杉矶，只需几个小时。*我闭上眼睛，微笑* 这铁路将由太阳能驱动，零排放，与自然和谐共生。*我张开双臂* 我们将邀请全球创意人才，共同打造，让美国再次成为创新的灯塔。*我指向自己的心脏* 选择我，就是选择一个无限可能的未来。一起，让我们让美国再次飞跃！*我用手指指向天空，展示出向上的力量*


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 7
        伊丽莎白·沃伦 竞价: 7
选择: 唐纳德·特朗普


(唐纳德·特朗普): *我自信地笑起来*，我们的铁路将会让其他所有国家都黯然失色。*我用手指点着*，我们会用美国制造的材料，创造就业机会，让美国再次伟大。*我点头强调*，而且，我们会和最顶尖的科技公司合作，确保我们的铁路是世界上最快、最高效的。*我双手展开*，想象一下，从东海岸到西海岸，只需要几个小时，而且，是坐着最豪华的座位！*我微笑着看向观众*，这是特朗普的方式，让美国再次引领世界，而且，我们将让环保成为可能，但是，美国工人，美国企业，永远是第一位的！*我竖起大拇指*，选择我，就是选择了一个胜利的未来！


竞价结果:
        唐纳德·特朗普 竞价: 7
        坎耶·韦斯特 竞价: 7
        伊丽莎白·沃伦 竞价: 7
选择: 伊丽莎白·沃伦


(伊丽莎白·沃伦): *我深吸一口气，眼神坚定*。我们的高速铁路计划，是关于投资未来，而不仅仅是奢华。我提议设立“社区连接计划”，确保沿线社区得到发展，提供教育和技术培训，让每个人都能参与到这一历史性进步中来。*我挥动手臂，展现决心*，我们将通过“绿色技术创新挑战”，激励企业开发环保材料和技术，确保我们的铁路不仅是快的，更是可持续的。*我点头，声音充满信心*，这不仅是一条铁路，这是对未来的承诺，一个为所有人创造机会，保护我们星球，并确保美国在全球舞台上领先的承诺。


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 8
        伊丽莎白·沃伦 竞价: 7
选择: 唐纳德·特朗普


(唐纳德·特朗普): *我自信地笑起来*，伊丽莎白，你的计划听起来很不错，但是太理论化了。美国需要的是实际行动和快速决策。我们将建造的铁路，不仅会连接社区，还会让它们繁荣起来。*我双手做出一个巨大的圆圈*，想象一下，每个站点都是一个经济增长的引擎，创造就业，刺激当地经济。*我点头*，而且，我们将确保这个过程快速高效，不像过去的政府项目，拖拖拉拉。*我用手指比出一个数字1*，第一，我们会让美国制造回归，第二，我们会让美国企业领导这个项目，第三，我们会确保它是最环保的。*我紧握拳头*，这是特朗普的保证，我们将让美国再次伟大，而且是以一种前所未有的方式！*我坚定地看向观众*，选择我，就是选择一个强大的美国！


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 7
        伊丽莎白·沃伦 竞价: 8
选择: 伊丽莎白·沃伦


(伊丽莎白·沃伦): *我冷静地回应，目光如炬*，唐纳德，你的热情值得赞赏，但我们的未来需要深思熟虑的规划。我的计划不仅注重速度和效率，更重视长远影响。我们将实施“铁路可持续性指数”，确保每一步都符合最高环保标准。*我微笑着点头*，这不仅是关于今天的工作，更是为了我们孩子的未来。*我强调地举起手*，我们要的是一个既能驱动经济，又能保护地球的解决方案。这才是真正的美国优先！


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 8
        伊丽莎白·沃伦 竞价: 7
选择: 坎耶·韦斯特


(坎耶·韦斯特): *我深吸一口气，眼神如炬* 我们的铁路，是艺术的化身，是科技与灵魂的交响。*我轻抚胸口* 想象一下，每个站点不仅是交通枢纽，而是文化地标，展示美国多样性与创造力。*我手指轻点，仿佛在空中作画* 我们将打造虚拟现实休息室，让等待成为探索，让旅行超越物理界限。*我微笑，眼神闪烁* 这铁路，是连接现在与未来的纽带，是让美国再次梦想的桥梁。*我张开双手，仿佛拥抱未来* 与我同行，我们共同创造历史，让美国再次引领世界，不仅是速度，而是灵感的飞翔。*我指向自己，再指向观众* 选择我，选择一个让心灵共振、梦想成真的未来。


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 8
        伊丽莎白·沃伦 竞价: 7
选择: 唐纳德·特朗普


(唐纳德·特朗普): *我自信地笑起来*，我们的铁路项目，将不仅仅是一条铁路，它将是工程奇迹，展现美国的力量和创新。*我指着观众*，我们将与最顶尖的美国公司合作，确保使用最先进的技术，比如磁悬浮列车，让旅行时间缩短到前所未有的程度。*我点头*，这将是一次革命，让世界看到美国如何再次领导科技前沿。*我紧握拳头*，我们将创造数百万个工作岗位，让美国经济再次腾飞。*我用手指点着*，而且，我们会确保这个过程对环境的影响最小，使用最环保的材料和技术。*我微笑着看向观众*，选择我，就是选择一个创新、繁荣和环保并存的未来！*我竖起大拇指*，让美国再次伟大，而且是以最特朗普的方式！


竞价结果:
        唐纳德·特朗普 竞价: 8
        坎耶·韦斯特 竞价: 8
        伊丽莎白·沃伦 竞价: 8
选择: 伊丽莎白·沃伦


(伊丽莎白·沃伦): *我深吸一口气，眼神坚定*。我们的高速铁路不仅是连接城市，更是连接人心的桥梁。我提出“铁路教育伙伴计划”，与学校合作，让学生参与铁路建设过程，学习STEM技能，为未来做好准备。*我挥动手臂，展现决心*，我们将通过“铁路社区花园”项目，在每个站点种植本地植物，促进生物多样性，同时提供新鲜食物给当地社区。*我点头，声音充满信心*，这不仅是一条铁路，这是对未来的投资，一个为所有人创造机会，保护我们星球，并确保美国在全球舞台上领先的承诺。
```

## 关键特性

1. **去中心化发言选择**: 每个Agent自主决定是否发言，通过竞价机制选择发言者
2. **动态竞价策略**: Agent根据对话历史和当前状态调整竞价
3. **角色个性化**: 每个Agent都有独特的性格和发言风格
4. **灵活的扩展性**: 可以轻松添加新的Agent类型和竞价策略

### 自定义配置

您可以通过修改以下参数来自定义辩论：

- `character_names`: 修改参与者名单
- `topic`: 更改辩论主题
- `max_iters`: 调整对话轮数
- 竞价模板: 修改竞价策略和评分标准

## 总结

这个多Agent竞价框架展示了如何使用LazyLLM构建复杂的多Agent对话系统。通过竞价机制，我们实现了去中心化的发言选择，使对话更加自然和动态。该框架可以作为构建更复杂多Agent系统的基础，如辩论系统、会议模拟、游戏AI等。
