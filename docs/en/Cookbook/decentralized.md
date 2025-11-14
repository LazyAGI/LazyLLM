# Multi-Agent Decentralized Speaker Selection

This project demonstrates how to build a **multi-agent decentralized speaker selection system** using [LazyLLM](https://github.com/LazyAGI/LazyLLM), showcasing multi-agent simulation without a fixed schedule.

!!! abstract "In this section, you will learn the following key features of LazyLLM"

    - How to build conversational agents using [OnlineChatModule][lazyllm.module.llms.onlinemodule.chat.OnlineChatModule].
    - How to apply different prompts to the model via [ChatPrompter][lazyllm.components.ChatPrompter].
    - How to orchestrate multiple agents to enable collaborative multi-agent simulation.

## Feature Overview

* **Multi-Agent Debate Simulation**: Creates a virtual debate scenario powered by large language models, allowing multiple AI agents with distinct personality traits to participate.
* **Customizable Roles**: Generates unique character descriptions and system prompts for each participant (e.g., "Donald Trump", "Kanye West"), ensuring their speaking style aligns with their assigned persona.
* **Bidding-Based Speaker Selection**: Implements a "bidding" mechanism to determine the next speaker—each agent evaluates its willingness to speak based on the conversation history and bids (a score from 1 to 10). The highest bidder gains the floor.
* **Dialogue Flow Control**: Uses `DialogueSimulator` to manage multi-turn dialogue flow, automatically handling speaker sequencing, message broadcasting, and history maintenance.
* **Topic Specification**: Before the debate begins, a large language model refines a broad topic (e.g., "transcontinental high speed rail") into a more concrete and debatable question to guide the discussion.

## Design Concept

This is a bidding-based multi-agent dialogue simulation system designed for candidate debate scenarios. The system allows each agent to "bid" for the right to speak in the next turn, enabling more natural and strategic multi-turn conversations.

First, three debate participants with distinct character traits are initialized. Each receives a personalized description and system prompt, and a large-model-based dialogue agent is constructed for each.

Next, a moderator agent refines the original debate topic into a more specific and actionable question, which is then injected into all participants’ conversation histories.

Then, the multi-turn simulation begins: in each round, all agents "bid" (outputting a score from 1 to 10) based on how contradictory the latest message is to their own stance—higher contradiction indicates stronger motivation to respond. The system selects the agent with the highest bid to speak and broadcasts its message to all other agents.

Finally, the system iterates for a predefined number of rounds and outputs the complete debate transcript, implementing a structured, competitive multi-agent dialogue system.

![decentralized](../assets/decent.png)

## Code Implementation

### Project Dependencies

Make sure you have installed the following dependencies:

```bash
pip install lazyllm
```

Import necessary packages:

```python
from thirdparty import numpy as np
import re
import tenacity
from typing import List, Callable
import lazyllm
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from lazyllm import ChatPrompter
```

### Step-by-Step Implementation

#### Step 1: Initialize Multi-Role Dialogue Agents and DialogueSimulator

`DialogueAgent` provides two main functionalities:  
- `send()`: Applies the model to the message history and returns a message string.  
- `receive(name, message)`: Adds the speaker and their message to the conversation history.

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
        self.message_history = ['Host: Here is the conversation so far.']

    def send(self) -> str:
        history = '\n'.join(self.message_history + [self.prefix])
        message = self.model(history)
        message = remove_think_section(message)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')
```

`DialogueSimulator` takes a list of agents and, at each step, performs the following actions:  
- `step()`: Selects the next speaker and broadcasts their message.  
- `inject()`: Injects a message into the conversation (e.g., from a moderator).

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

#### Step 2: Define the Bidding Mechanism

This method generates a bid given the conversation history and the most recent message.

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

#### Step 3: Define Participants and Debate Topic

Generate system descriptions and detailed profiles for each participant.

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

Inspect the generated content:

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
#### Step 4: Parse Bid Information and Generate Bidding Templates

Agents are LLMs that output strings, so we need to define the format they will generate and parse their output.

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

Generate the corresponding bidding templates:

```python
def generate_character_bidding_template(character_header):
    bidding_template = f'''{character_header}
    ```
    {{message_history}}
    ```
    On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory,
    rate how contradictory the following message is to your ideas
    ```
    {{recent_message}}
    ```
    Your response should be an integer delimited by angled brackets, like this: <int>.
    Do nothing else.
    '''
    return bidding_template


character_bidding_templates = [
    generate_character_bidding_template(character_header)
    for character_header in character_headers
]
```

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
#### Step 5: Create the Debate Topic and Launch the Simulation

Use an LLM to generate a refined, actionable debate topic.

```python
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
```

**Example Output:**
```text
Original topic:
transcontinental high speed rail

Detailed topic:
"Donald Trump, Kanye West, Elizabeth Warren: How should the U.S. fund and construct a transcontinental high-speed rail network while balancing economic revitalization, environmental sustainability, and equitable access for all regions?"
```

Define a speaker selection function that collects bids from all agents and selects the one with the highest bid (breaking ties randomly), then launch the debate simulation.

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

### Full Code
<details>
<summary>Click to expand/collapse Python code</summary>

```python
from thirdparty import numpy as np
import re
import tenacity
from typing import List, Callable
import lazyllm
from lazyllm.module.llms.onlinemodule.base.onlineChatModuleBase import StaticParams
from lazyllm import ChatPrompter

def remove_think_section(text: str) -> str:
    cleaned_text = re.sub(r'\<think\>.*?\<\/think\>\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

class DialogueAgent:
    def __init__(self, name, model, system_message):
        self.name = name
        self.model = model
        self.prefix = f'{self.name}: '
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ['Host: Here is the conversation so far.']

    def send(self) -> str:
        history = '\n'.join(self.message_history + [self.prefix])
        message = self.model(history)
        message = remove_think_section(message)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')

class DialogueSimulator:
    def __init__(self, agents: List[DialogueAgent], selection_function: Callable[[int, List[DialogueAgent]], int]):
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
    def __init__(self, name, model, system_message, bidding_template):
        super().__init__(name, model, bidding_template)
        self.bidding_template = bidding_template

    def bid(self) -> str:
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
        instruction={'system': player_descriptor_system_message, 'user': game_description})
    contents = character_specifier_prompt.generate_prompt(
        f'''{game_description}
        Please reply with a creative description of the presidential candidate,
        {character_name}, in {word_limit} words or less,
        that emphasizes their personalities.
        Speak directly to {character_name}.
        Do not add anything else.''',
        return_dict=True)
    character_description = lazyllm.OnlineChatModule(model='Qwen3-32B', static_params=temp)(contents)
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
Speak in the first person from the perspective of {character_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    '''

character_descriptions = [generate_character_description(name) for name in character_names]
character_headers = [
    generate_character_header(name, desc) for name, desc in zip(character_names, character_descriptions)
]
character_system_messages = [
    generate_character_system_message(name, header) for name, header in zip(character_names, character_headers)
]

@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'),
    retry_error_callback=lambda retry_state: 0,
)
def parse_bid(bid_string: str) -> int:
    match = re.search(r'<(\d+)>', bid_string.strip())
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f'Invalid bid format: {bid_string}')

def ask_for_bid(agent) -> int:
    bid_string = agent.bid()
    bid = parse_bid(bid_string)
    return bid

def generate_character_bidding_template(character_header):
    bidding_template = f'''{character_header}
    ```{{message_history}}```
    On the scale of 1 to 10, where 1 is not contradictory and 10 is extremely contradictory,
    rate how contradictory the following message is to your ideas
    ```{{recent_message}}```
    Your response should be an integer delimited by angled brackets, like this: <int>.
    Do nothing else.
    '''
    return bidding_template

character_bidding_templates = [generate_character_bidding_template(header) for header in character_headers]

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
for name, msg, template in zip(character_names, character_system_messages, character_bidding_templates):
    characters.append(
        BiddingDialogueAgent(
            name=name,
            model=lazyllm.OnlineChatModule(model='Qwen3-32B', system_prompt=msg, static_params=temp_low),
            system_message=msg,
            bidding_template=template,
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

### Example Output

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

## Conclusion

This tutorial demonstrates how to use LazyLLM to build a **multi-agent decentralized speaker selection agent**, showing how to implement multi-agent simulation without a fixed schedule. By combining `ChatPrompter` and `OnlineChatModule`, you can quickly set up agent roles with different identities.