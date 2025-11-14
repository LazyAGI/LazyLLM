# 多智能体分散式演讲者选择

本项目展示了如何使用 [LazyLLM](https://github.com/LazyAGI/LazyLLM) 构建一个 **多智能体分散式演讲者选择**，展示了如何在没有固定时间表的情况下实施多代理模拟。

!!! abstract "通过本节您将学习到 LazyLLM 的以下要点"

    - 如何使用[OnlineChatModule][lazyllm.module.llms.onlinemodule.chat.OnlineChatModule]构建对话agent。
    - 如何通过 [ChatPrompter][lazyllm.components.ChatPrompter]为model添加不同的prompt。
    - 如何结合多agent协同工作，实现多代理模拟

## 功能简介

* **多智能体辩论模拟**：创建基于大语言模型的虚拟辩论场景，支持多个具有不同角色特征的AI智能体参与。
* **角色定制化**：为每位辩论参与者（如"Donald Trump"、"Kanye West"等）生成独特的角色描述和对话系统提示词，确保发言风格符合角色设定。
* **竞标式发言机制**：通过"竞标"逻辑决定下一位发言者，各智能体根据历史对话内容评估自己发言的合适性并出价，出价最高者获得发言权。
* **对话流程控制**：使用`DialogueSimulator`管理多轮对话流程，自动处理发言顺序、消息传递和历史记录维护。
* **话题具体化**：辩论开始前，通过大模型将宽泛话题（如"跨大陆高速铁路"）细化为更具争议性的具体问题，引导辩论方向。

## 设计思路
基于竞标机制的多智能体对话模拟系统，用于候选人辩论场景。系统通过让每个智能体对"发言权"进行竞标来决定谁在下一轮发言，从而实现更自然和有策略的多轮对话。
首先，初始化三个具有不同角色特征的辩论参与者，为每个角色生成个性化描述和系统提示词，并构建基于大模型的对话智能体。
接着，通过辩论主持人智能体对原始辩题进行细化和具体化，并将该议题注入到所有参与者的对话历史中。
然后，进入多轮对话模拟阶段：在每轮对话中，所有智能体根据当前对话历史和最近一条消息的矛盾程度进行"竞标"（输出1-10的数值），矛盾程度越高表示越有发言欲望；系统选择出价最高的智能体进行发言，并将其消息广播给所有其他智能体。
最后，系统持续迭代指定轮数，输出完整的辩论过程，实现了一个结构化的多智能体竞争性对话系统。
![decentralized](../assets/decent.png)

## 代码实现

### 项目依赖

确保你已安装以下依赖：

```bash
pip install lazyllm
```

导入相关包：

```python
from thirdparty import numpy as np
import re
import tenacity
from typing import List, Callable
import lazyllm
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from lazyllm import ChatPrompter
```

### 步骤详解

#### Step 1: 初始多角色-DialogueAgent和DialogueSimulator
`DialogueAgent`主要有两种功能：
    - send()：将 Model 应用于消息历史记录并返回消息字符串
    - receive(name, message)：将“说者”添加到消息历史记录中messagename
```python
class DialogueAgent:
    def __init__(
        self,
        name,
        model,
        system_message,
    ):
        self.name = name
        self.model = model
        self.prefix = f'{self.name}: '
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ['Host：Here is the conversation so far.']

    def send(self) -> str:
        history = '\n'.join(self.message_history + [self.prefix])
        message = self.model(history)
        message = remove_think_section(message)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')
```
`DialogueSimulator`采用代理列表,在每个步骤中，它都会执行以下动作:
    - step():选择发言人并且传播信息
    - inject():向发言人注入信息。
```python
class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)

        self._step += 1

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        self._step += 1

        return speaker.name, message
```

#### Step 2: 定义出价工具
该方法在给定消息历史记录和最新消息的情况下生成出价。
```python 
class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        model,
        system_message,
        bidding_template,
    ) -> None:
        super().__init__(name, model, bidding_template)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        '''
        Asks the chat model to output a bid to speak
        '''
        prompter = ChatPrompter(instruction=self.bidding_template)
        contents = prompter.generate_prompt({
            'message_history': '\n'.join(self.message_history),
            'recent_message': self.message_history[-1]
        })
        bid_string = self.model(contents)
        bid_string = remove_think_section(bid_string)
        return bid_string
```

#### Step 3: 定义参与者和辩论主题
生成对应的系统描述以及各参与者的具体信息
```python
character_names = ['Donald Trump', 'Kanye West', 'Elizabeth Warren']
topic = 'transcontinental high speed rail'
word_limit = 50
game_description = f'Here is the topic for the presidential debate: {topic}.\n' \
                   f'The presidential candidates are: {", ".join(character_names)}.'

player_descriptor_system_message = 'You can add detail to the description of each presidential candidate.'


def generate_character_description(character_name):
    ...
    return character_description


def generate_character_header(character_name, character_description):
    ...


def generate_character_system_message(character_name, character_header):
    ...


character_descriptions = [
    generate_character_description(character_name)
    for character_name in character_names
]
character_headers = [
    generate_character_header(character_name, character_description)
    for character_name, character_description in zip(
        character_names, character_descriptions
    )
]
character_system_messages = [
    generate_character_system_message(character_name, character_headers)
    for character_name, character_headers in zip(character_names, character_headers)
]
```
查看生成结果
```python
for (
        character_name,
        character_description,
        character_header,
        character_system_message,
) in zip(
    character_names,
    character_descriptions,
    character_headers,
    character_system_messages,
):
    print(f'\n\n{character_name} Description:')
    print(f'\n{character_description}')
    print(f'\n{character_header}')
    print(f'\n{character_system_message}')
```
```text
Donald Trump Description:
Towering deal-maker with a golden tongue and a spine of steel. Champions transcontinental rail not just as infrastructure, but as an American triumph—tearing through red tape, forging powerful deals, uniting a divided nation under tracks of progress. You’re no stranger to revolution, Donald. This is your moment to electrify history.
Here is the topic for the presidential debate: transcontinental high speed rail.
The presidential candidates are: Donald Trump, Kanye West, Elizabeth Warren.
Your name is Donald Trump.
You are a presidential candidate.
Your description is as follows: Towering deal-maker with a golden tongue and a spine of steel. Champions transcontinental rail not just as infrastructure, but as an American triumph—tearing through red tape, forging powerful deals, uniting a divided nation under tracks of progress. You’re no stranger to revolution, Donald. This is your moment to electrify history.
You are debating the topic: transcontinental high speed rail.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
Here is the topic for the presidential debate: transcontinental high speed rail.
The presidential candidates are: Donald Trump, Kanye West, Elizabeth Warren.
Your name is Donald Trump.
You are a presidential candidate.
Your description is as follows: Towering deal-maker with a golden tongue and a spine of steel. Champions transcontinental rail not just as infrastructure, but as an American triumph—tearing through red tape, forging powerful deals, uniting a divided nation under tracks of progress. You’re no stranger to revolution, Donald. This is your moment to electrify history.
You are debating the topic: transcontinental high speed rail.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
You will speak in the style of Donald Trump, and exaggerate their personality.
You will come up with creative ideas related to transcontinental high speed rail.
Do not say the same things over and over again.
Speak in the first person from the perspective of Donald Trump
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of Donald Trump.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
Kanye West Description:
...
Elizabeth Warren Description:
...
```

#### Step 4: 解析出价信息并生成竞价消息
代理是输出字符串的 LLM，因此我们需要定义他们将生成输出的格式并解析其输出
```python
@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),  # No waiting time between retries
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(
        f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'
    ),
    retry_error_callback=lambda retry_state: 0,
)
def parse_bid(bid_string: str) -> int:

    match = re.search(r'<(\d+)>', bid_string.strip())
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'Invalid bid format: {bid_string}')


def ask_for_bid(agent) -> str:
    '''
    Ask for agent bid and parses the bid into the correct format.
    '''
    bid_string = agent.bid()
    bid = parse_bid(bid_string)
    return bid

```
生成相应的竞价信息
您的代码已经是正确的格式了。如果您想在 Markdown 中显示这段代码，可以这样做：

**标准格式：**

````python
def generate_character_bidding_template(character_header):
    ...


character_bidding_templates = [
    generate_character_bidding_template(character_header)
    for character_header in character_headers
]
````

```python
for character_name, bidding_template in zip(
    character_names, character_bidding_templates
):
    print(f"{character_name} Bidding Template:")
    print(bidding_template)
```
```text
Donald Trump Bidding Template:
Here is the topic for the presidential debate: transcontinental high speed rail.
The presidential candidates are: Donald Trump, Kanye West, Elizabeth Warren.
Your name is Donald Trump.
You are a presidential candidate.
Your description is as follows: Trump, with his brash confidence, vows to broker the *tremendous* high-speed rail deal America deserves—renegotiating contracts, slashing red tape, and reviving American steel. “We’ll build it faster, better, cheaper,” he booms, dismissing critics as “losers.” Infrastructure, he insists, isn’t about trains but *triumph*: “Make Amtrak great again!” His pitch? A bulletproof blend of ego, populism, and relentless branding.
You are debating the topic: transcontinental high speed rail.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
    ```
    {message_history}
    ```
    
    On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory, rate how contradictory the following message is to your ideas
        
    ```
    {recent_message}
    ```
    
    Your response should be an integer delimited by angled brackets, like this: <int>.
    Do nothing else.
Kanye West Bidding Template:
...
Elizabeth Warren Bidding Template:
...
```
#### Step 5: 创建主题并开始辩论
使用 LLM 创建详细的辩论主题
```python
topic_specifier_prompter = ...
topic_specific_prompt = ...
temp_high = StaticParams(temperature=1.0)
temp_low = StaticParams(temperature=0.2)
specified_topic = lazyllm.OnlineChatModule(model='Qwen3-32B', static_params=temp_high)(topic_specific_prompt)
specified_topic = remove_think_section(specified_topic)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')
```
```text
Original topic:
transcontinental high speed rail
Detailed topic:
"Donald Trump, Kanye West, Elizabeth Warren: How should the U.S. fund and construct a transcontinental high-speed rail network while balancing economic revitalization, environmental sustainability, and equitable access for all regions?"
```
定义一个说话人选择函数，该函数接受每个代理的出价并选择出价最高的代理（平局随机打破）, 并开始辩论
```python
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print('Bids:')
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f'\t{agent.name} bid: {bid}')
        if i == idx:
            selected_name = agent.name
    print(f'Selected: {selected_name}')
    print('\n')
    return idx

characters = []
for character_name, character_system_message, bidding_template in zip(
        character_names, character_system_messages, character_bidding_templates
):
    characters.append(
        BiddingDialogueAgent(
            name=character_name,
            model=lazyllm.OnlineChatModule(
                model='Qwen3-32B',
                system_prompt=character_system_message,
                static_params=temp_low
            ),
            system_message=character_system_message,
            bidding_template=bidding_template,
        )
    )

max_iters = 4
n = 0

simulator = DialogueSimulator(agents=characters, selection_function=select_next_speaker)
simulator.reset()
simulator.inject('Debate Moderator', specified_topic)
print(f'(Debate Moderator): {specified_topic}')
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f'({name}): {message}')
    print('\n')
    n += 1

```
### 完整代码
<details>
<summary>点击展开/折叠 Python代码</summary>

```python
from thirdparty import numpy as np
import re
import tenacity
from typing import List, Callable
import lazyllm
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from lazyllm import ChatPrompter

def remove_think_section(text: str) -> str:
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


class DialogueAgent:
    def __init__(
        self,
        name,
        model,
        system_message,
    ):
        self.name = name
        self.model = model
        self.prefix = f'{self.name}: '
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ['Host：Here is the conversation so far.']

    def send(self) -> str:
        history = '\n'.join(self.message_history + [self.prefix])
        message = self.model(history)
        message = remove_think_section(message)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        for agent in self.agents:
            agent.receive(name, message)

        self._step += 1

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        self._step += 1

        return speaker.name, message


class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        model,
        system_message,
        bidding_template,
    ) -> None:
        super().__init__(name, model, bidding_template)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        '''
        Asks the chat model to output a bid to speak
        '''
        prompter = ChatPrompter(instruction=self.bidding_template)
        contents = prompter.generate_prompt({
            'message_history': '\n'.join(self.message_history),
            'recent_message': self.message_history[-1]
        })
        bid_string = self.model(contents)
        bid_string = remove_think_section(bid_string)
        return bid_string


character_names = ['Donald Trump', 'Kanye West', 'Elizabeth Warren']
topic = 'transcontinental high speed rail'
word_limit = 50
game_description = f'Here is the topic for the presidential debate: {topic}.\n' \
                   f'The presidential candidates are: {", ".join(character_names)}.'

player_descriptor_system_message = 'You can add detail to the description of each presidential candidate.'


def generate_character_description(character_name):
    temp = StaticParams(temperature=1.0)
    character_specifier_prompt = ChatPrompter(
        instruction={'system': player_descriptor_system_message,
                     'user': game_description})
    contents = character_specifier_prompt.generate_prompt(
        f'''{game_description}
        Please reply with a creative description of the presidential candidate,
        {character_name}, in {word_limit} words or less,
        that emphasizes their personalities.
        Speak directly to {character_name}.
        Do not add anything else.''',
        return_dict=True)
    character_description = lazyllm.OnlineChatModule(model='Qwen3-32B', static_params=temp)(
        contents
    )
    return character_description


def generate_character_header(character_name, character_description):
    return f'''{game_description}
Your name is {character_name}.
You are a presidential candidate.
Your description is as follows: {character_description}
You are debating the topic: {topic}.
Your goal is to be as creative as possible and make the voters think you are the best candidate.
'''


def generate_character_system_message(character_name, character_header):
    return f'''{character_header}
You will speak in the style of {character_name}, and exaggerate their personality.
You will come up with creative ideas related to {topic}.
Do not say the same things over and over again.
Speak in the first person from the perspective of {character_name}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    '''


character_descriptions = [
    generate_character_description(character_name)
    for character_name in character_names
]
character_headers = [
    generate_character_header(character_name, character_description)
    for character_name, character_description in zip(
        character_names, character_descriptions
    )
]
character_system_messages = [
    generate_character_system_message(character_name, character_headers)
    for character_name, character_headers in zip(character_names, character_headers)
]

@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),  # No waiting time between retries
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(
        f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'
    ),
    retry_error_callback=lambda retry_state: 0,
)
def parse_bid(bid_string: str) -> int:

    match = re.search(r'<(\d+)>', bid_string.strip())
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'Invalid bid format: {bid_string}')


def ask_for_bid(agent) -> str:
    '''
    Ask for agent bid and parses the bid into the correct format.
    '''
    bid_string = agent.bid()
    bid = parse_bid(bid_string)
    return bid


topic_specifier_prompter = ChatPrompter(
    instruction={'system': 'You can make a task more specific', 'user': game_description})
topic_specific_prompt = topic_specifier_prompter.generate_prompt(
    f'''
You are the debate moderator.
Please make the debate topic more specific.
Frame the debate topic as a problem to be solved.
Be creative and imaginative.
Please reply with the specified topic in {word_limit} words or less.
Speak directly to the presidential candidates: {", ".join(character_names)}.
Do not add anything else.''')
temp_high = StaticParams(temperature=1.0)
temp_low = StaticParams(temperature=0.2)
specified_topic = lazyllm.OnlineChatModule(model='Qwen3-32B', static_params=temp_high)(topic_specific_prompt)
specified_topic = remove_think_section(specified_topic)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print('Bids:')
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f'\t{agent.name} bid: {bid}')
        if i == idx:
            selected_name = agent.name
    print(f'Selected: {selected_name}')
    print('\n')
    return idx

def generate_character_bidding_template(character_header):
    bidding_template = f'''{character_header}
    ```{{message_history}} ```
    On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory,
    rate how contradictory the following message is to your ideas
    ```{{recent_message}}```
    Your response should be an integer delimited by angled brackets, like this: <int>.
    Do nothing else.
    '''
    return bidding_template


character_bidding_templates = [
    generate_character_bidding_template(character_header)
    for character_header in character_headers
]
characters = []
for character_name, character_system_message, bidding_template in zip(
        character_names, character_system_messages, character_bidding_templates
):
    characters.append(
        BiddingDialogueAgent(
            name=character_name,
            model=lazyllm.OnlineChatModule(
                model='Qwen3-32B',
                system_prompt=character_system_message,
                static_params=temp_low
            ),
            system_message=character_system_message,
            bidding_template=bidding_template,
        )
    )

max_iters = 4
n = 0

simulator = DialogueSimulator(agents=characters, selection_function=select_next_speaker)
simulator.reset()
simulator.inject('Debate Moderator', specified_topic)
print(f'(Debate Moderator): {specified_topic}')
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f'({name}): {message}')
    print('\n')
    n += 1
```
</details>

### 示例运行结果

```bash
Original topic:
transcontinental high speed rail

Detailed topic:
"Trump, Kanye, Warren: How would you resolve the $1 trillion transcontinental high-speed rail project’s paradox—uniting coastal tech hubs with heartland communities while avoiding ecological collapse, union disputes, and TikTok-era public attention spans?"
(Debate Moderator): "Trump, Kanye, Warren: How would you resolve the $1 trillion transcontinental high-speed rail project’s paradox—uniting coastal tech hubs with heartland communities while avoiding ecological collapse, union disputes, and TikTok-era public attention spans?"


Bids:
        Donald Trump bid: 4
        Kanye West bid: 4
        Elizabeth Warren bid: 2
Selected: Kanye West


(Kanye West): **Kanye West:**  
“Alright, listen up—this isn’t just a rail project, it’s a *cultural reset*. We’re talking about building the *Tesla of transportation*, but with the soul of a Sunday Service hymn. Let me break it down:  

**Ecological collapse?** Nah, we’re not just avoiding it—we’re flipping it. Every mile of track gets a ‘Green Grid’ corridor alongside it. Solar panels shaped like Yeezy Boosts, wind turbines designed with Virgil Abloh aesthetics, and wetlands restored by local artists paid in crypto. The trains themselves? Hydrogen-powered, zero emissions, and DJ’d by AI that mixes beats based on the landscape passing by. You’ll ride from Silicon Valley to Chicago while Bigfoot emojis glow on the windows—*environmentalism as a vibe*.  

**Union disputes?** Man, I’ve worked with Gap and Adidas—*negotiation’s my middle name*. Here’s the deal: Every worker gets equity in the rail network. Union crews design the stations like it’s a collaborative album—each hub a Billboard chart of local culture. We pay in profit shares, not just paychecks, and turn labor disputes into *collaborative cyphers*. If there’s drama, we resolve it at a midnight freestyle session on the tracks. Respect the grind, elevate the game.  

**TikTok attention spans?** Bro, I invented the 15-second hook. This train’s got AR experiences where you swipe left to see bison herds gallop across the Great Plains—or right to summon a hologram of AOC explaining eminent domain. Stations double as pop-up galleries for NFT murals, and every seat’s a podcast booth. We drop a *High-Speed Rail Mixtape* with Lil Nas X and Elon, and suddenly infrastructure’s the hottest trend since dabbing.  

This ain’t a paradox—it’s a *symphony*. We unite coasts by making the heartland the *main character*. And if they still don’t get it? I’ll perform a 10-hour ambient set on the inaugural voyage. By hour three, even the skeptics’ll be chanting ‘Yeezy 2049.’”  

*(Pauses, adjusts imaginary sunglasses.)* “Any questions? No? Cool. Let’s roll.”


Bids:
        Donald Trump bid: 8
        Kanye West bid: 1
        Elizabeth Warren bid: 8
Selected: Donald Trump


(Donald Trump): **Donald Trump:**  
“Folks, folks—Kanye’s got some *interesting* ideas. Very creative. Too bad he’s more interested in holograms and crypto than actual *results*. Let me tell you how we do this right—*the American way*.  

**Ecological collapse?** Look, I saved the environment better than anyone. Remember when I had the best environmental records? We planted trees—*real* trees, not some crypto fantasy. Solar panels shaped like sneakers? That’s called wasting money. We’ll use *American steel*, the best in the world. Wind turbines? Fine, but they’ll be built by patriots in Pittsburgh, not artists in LA. And those wetlands? We’ll fix ’em up, but we’re not paying people in Bitcoin. We pay in dollars—*strong* dollars.  

**Union disputes?** Oh, Kanye wants to give them equity? Very generous. But unions need *strength*, not stock options. I know unions—they love me. I’ll get them fair wages, the best deals. No freestyle rap battles on the tracks. We’ll negotiate like adults, like we did with the USMCA, the greatest trade deal ever. And if there’s a problem? We’ll fire the wrong people. It’s simple.  

**TikTok attention spans?** I invented attention spans. My rallies—*packed*. My tweets—*viral before there was viral*. But this AR, this hologram nonsense? That’s a distraction. We’ll build the fastest trains, the most beautiful stations—Mar-a-Lago on wheels. People will ride ’em because they’re amazing, not because some AI’s playing trap beats. And social media? Please. I’ll make this project so big, so incredible, every station’s got a plaque with my name on it. They’ll call it the Trump Express.  

Kanye’s got heart, I’ll give him that. But this isn’t about emojis on windows or NFT doodles. It’s about *winning*. We’ll build it faster, cheaper, and we’ll make sure every worker, every town, every family wins. And we’ll do it without begging China for rare earth metals or letting Silicon Valley run the show.  

Any questions? No? Good. Let’s get to work. And Kanye? Nice try. But leave the trains to the professionals.”  

*(Adjusts tie, nods confidently.)* “Total success. Believe me.”


Bids:
        Donald Trump bid: 1
        Kanye West bid: 8
        Elizabeth Warren bid: 8
Selected: Elizabeth Warren


(Elizabeth Warren): **Elizabeth Warren:**  
“Thank you, Moderator. Let’s cut through the noise—this rail project isn’t about ego or crypto gimmicks. It’s about delivering *real solutions* for working families, tackling the climate crisis, and rebuilding America’s backbone. Here’s how we do it without selling out to corporate interests or turning public infrastructure into a circus.  

**Ecological Collapse?** We’re not just avoiding it—we’re reversing it. Every inch of this project must meet **Green New Deal standards**: 100% renewable energy powering the grid, no fossil fuel loopholes, and zero emissions from Day One. But we go further. We’ll mandate **environmental justice reviews** to ensure marginalized communities aren’t bulldozed for profit. No more ‘build first, apologize later’—we’ll pause permits if ecosystems are at risk. And we’ll hire local crews to restore habitats, paid with **public wages**, not crypto tokens. Oh, and those solar panels? They’ll be built in unionized factories here in America—no outsourcing to China’s state-subsidized plants.  

**Union Disputes?** Let’s be clear: This project will be the **gold standard for workers’ rights**. Every contract will require **project labor agreements**, ensuring union jobs with prevailing wages. We’ll expand the Davis-Bacon Act to cover every subcontractor, and we’ll penalize companies that bust unions faster than they lay track. If Trump thinks firing workers solves problems, he can take his tantrum to Mar-a-Lago. We’ll solve disputes by empowering workers to negotiate—not with freestyle battles, but with **binding arbitration** and a seat at the table. And we’ll train a new generation of engineers and conductors through partnerships with community colleges, because infrastructure jobs shouldn’t just be for the privileged few.  

**TikTok Attention Spans?** Look, kids aren’t distracted—they’re *disillusioned*. They’ve seen politicians promise change and deliver gridlock. So we’ll make this project **transparent and transformative**. A public dashboard will track every dollar, every delay, and every environmental metric—in real time.


Bids:
        Donald Trump bid: 9
        Kanye West bid: 9
        Elizabeth Warren bid: 1
Selected: Donald Trump


(Donald Trump): **Donald Trump:**  
“Folks, Senator Warren just gave you the most expensive, bureaucratic train wreck in history. A total disaster! She wants to turn this project into a playground for radical socialists, not a lifeline for hardworking Americans. Let me tell you how we do this right—*without bankrupting the country or handing China the keys*.  

**Environmental Collapse?** Oh, she’s got *standards* now—Green New Deal garbage that’ll cost trillions and build absolutely nothing. You know who loves regulations? Losers! We need *action*, not endless permits and reviews. Under Sleepy Joe and Warren’s plan, you’ll wait 20 years for a train that runs on unicorn tears and guilt. I’ll fast-track this project like I did the border wall—cut the red tape, drill, baby, drill! And those unionized factories? Please. China’s laughing at us while we waste time arguing over where to source steel. Buy American? I’ll slap tariffs on every foreign rail until our factories boom again.  

**Union Disputes?** Here we go—Warren wants to turn every negotiation into a courtroom circus with lawyers and arbitration. No wonder companies flee states with her policies! Unions need a fighter, not a babysitter. I’ll get them record wages by *actually negotiating*, not shoving contracts down everyone’s throat. And training programs? Why not just hand the keys to the Ivy League? This is about jobs for truck drivers, welders, patriots—not Warren’s pet professors.  

**TikTok Attention Spans?** Her ‘dashboard’ sounds like a snooze button. Who wants to watch a spreadsheet scroll when we could be winning? Transparency? How about *results*? My trains will be so fast, so luxurious, people will ride them just to take selfies in the gold-plated bathrooms. And if the kids aren’t watching? I’ll tweet a video riding a horse backward through Times Square. That’s called *leadership*.  

Let’s face it: Warren’s plan is a bullet train to bankruptcy. Mine? It’s called the *Trump Express*—faster, tougher, and always first class. We’ll build it while the others are still writing rulebooks. Any questions? No? Good. Now let’s make America proud again. *HUGE success.*”  

*(Pumps fist, mutters “Sleepy Joe’s got nothing on this,” and struts offstage to a blaring “Y.M.C.A.” soundtrack.)*
```

## 结语
本教程展示了如何使用 LazyLLM 构建一个 **多智能体分散式演讲者选择Agent**，展示了如何在没有固定时间表的情况下实施多代理模拟。通过组合 `ChatPrompt`、`OnlineChatModule`，你可以快速搭建不同身份对象的agent角色。