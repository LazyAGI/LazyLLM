# Multi-Agent Speaker Selection with a Director

In complex multi-role dialogue systems, dynamically selecting speakers while maintaining rhythm and topic consistency is the key to achieving “intelligent dialogue orchestration.” This section demonstrates how to build a multi-agent dialogue system with “director-style scheduling” based on the LazyLLM framework, enabling different roles to interact naturally and coherently within a given topic.

In this example, the system introduces a *Director (Director Agent)* to control the speaking order and conversation termination logic, while other *DialogueAgents (Regular Dialogue Agents)* participate as controlled roles. The director determines the next suitable speaker through the language model and probabilistically decides when to end the conversation, thus enabling a flexible multi-agent collaboration mechanism.

!!! abstract "In this section, you will learn the following key points of LazyLLM:"

    - How to define multi-role dialogue agents (`DialogueAgent`) with memory and role settings.  
    - How to use [ChatPrompter][lazyllm.components.prompter.ChatPrompter] to build reusable prompt templates.  
    - How to control conversation turns and speaker selection through `DirectorDialogueAgent`.  
    - How to use [OnlineChatModule][lazyllm.module.OnlineChatModule] to simulate realistic multi-agent conversations.  

## Design Idea

To build a multi-role intelligent dialogue system, we need not only models that can answer questions but also roles that can “interact, collaborate, and communicate with rhythm.” This design takes LazyLLM as the core framework and adopts a modular approach, enabling the system to handle role generation, dialogue control, and rhythm orchestration.

First, we define the basic `DialogueAgent` class to simulate the individual behavior of each role. Each agent has an independent memory (`message_history`) and can generate responses that fit its persona according to the system prompt (`system_message`).

Next, we introduce the `DirectorDialogueAgent` (Director Agent) as the central controller. It not only selects the next speaker but also controls when the conversation should end based on probability. The director’s logic involves three key actions:

- Generating dialogue responses (`_generate_response`)
- Selecting the next speaker (`_choose_next_speaker`)
- Ending the discussion when necessary (`stop`)

To ensure that the director understands context and rhythm, we design multiple `ChatPrompter` templates corresponding to different stages (continuing discussion, selecting speakers, summarizing the ending, etc.). Then, all agents are registered into the system through the `DialogueSimulator`, which manages the speaking order according to predefined rules. The director’s selection function (`select_next_speaker`) decides who speaks at each step, forming a dynamic multi-role rotation.

The overall structure is shown below:

![multi-agent_authoritarian_speaker](../assets/multi-agent_authoritarian_speaker.png)

## Environment Setup

### Install Dependencies

Before using the project, please run the following command to install the required libraries:

```bash
pip install lazyllm tenacity typing
```

### Environment Variables

This workflow involves using an online large model. You need to set your API key (using Qwen as an example):

```bash
export LAZYLLM_QWEN_API_KEY="sk-******"
```

> ❗ Note: For details on how to obtain an API key, please refer to the [official documentation](docs.lazyllm.ai/).

### Import Dependencies

```python
import re
import random
import functools
import tenacity
from collections import OrderedDict
from typing import List, Callable

from lazyllm import OnlineChatModule, ChatPrompter
```

## Code Implementation

### Building a Multi-Agent Dialogue Simulator

`DialogueAgent` is a basic dialogue agent class used to simulate a single role within a multi-agent system.

The class includes the essential elements of a role: name (`name`), system message (`system_message`), language model (`model`), and dialogue history (`message_history`). It supports sending and receiving messages, allowing the agent to maintain contextual memory across multiple turns.

```python
class DialogueAgent:
    '''Regular dialogue role Agent'''
    def __init__(self, name, system_message, model):
        self.name = name
        self.model = model
        self.prefix = f'{self.name}: '
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ['Host：Here is the conversation so far.']

    def send(self) -> str:
        structured_history = []
        for i in range(1, len(self.message_history), 2):
            if i + 1 < len(self.message_history):
                parts = self.message_history[i].split(':', 1)
                user_msg = parts[0]
                ai_msg = parts[1]
                structured_history.append([user_msg, ai_msg])

        history = structured_history + [[self.prefix, '']]
        message = self.model(self.system_message, llm_chat_history=history)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')
```

**Main Functions**

- `__init__()`：Initializes the agent, including its name, system configuration, and language model.
- `reset()`：Resets the dialogue history, typically called at the start of a new topic.
- `send()`：Structures the conversation history and calls the model to generate the agent’s current response.
- `receive()`：Receives messages from other agents and appends them to the dialogue history.

`DialogueSimulator` coordinates the dialogue flow among multiple `DialogueAgent`s and acts as the “conversation scheduler” for the multi-agent system.

It maintains a turn counter `_step`, decides the next speaker using a provided selection function, and broadcasts the chosen message to all agents, enabling turn-based multi-role conversations.

```python
class DialogueSimulator:
    '''Dialogue simulator for managing multi-role interactions'''
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

**Main Functions**

- Track turns: Keeps count of the current step in the conversation.
- Select speaker: Dynamically determines the next speaking agent using the external `selection_function`.
- Broadcast message: Sends the current speaker's response to all participants, ensuring context consistency.

### Extracting Integers from Text

This relies on regular expressions to match text and provides a unified format specification and usage method.

```python
class IntegerOutputParser:
    '''Parser for extracting integers from text'''
    def __init__(self, regex: str, output_keys: List[str], default_output_key: str):
        self.pattern = re.compile(regex)
        self.output_keys = output_keys
        self.default_output_key = default_output_key

    def parse(self, text: str):
        match = self.pattern.search(text)
        if not match:
            raise ValueError(f'No match found for regex {self.pattern.pattern} in text: {text}')
        groups = match.groups()
        if len(groups) != len(self.output_keys):
            raise ValueError(
                f'Expected {len(self.output_keys)} groups, but found {len(groups)}'
            )
        result = {}
        for key, value in zip(self.output_keys, groups):
            try:
                result[key] = int(value)
            except ValueError:
                raise ValueError(f"Matched value for key '{key}' is not a valid integer: {value}")
        return result

    def get_format_instructions(self) -> str:
        return 'Your response should be an integer delimited by angled brackets, like this: <int>.'

    def __call__(self, text: str):
        parsed = self.parse(text)
        return parsed.get(self.default_output_key)
```

### Building a Privileged Agent

The `DirectorDialogueAgent` is responsible for selecting the next agent to speak.

To effectively guide the conversation, the following three steps need to be performed:

- Reflect on the current conversation content.
- Choose the next speaking agent.
- Prompt that agent to speak.

While it is possible to perform all three steps in a single LLM call, this approach requires additional parsing code to extract the "next speaker" from the output text. This is not reliable, as the LLM may express its choice in multiple ways, increasing parsing complexity.

Therefore, in `DirectorDialogueAgent`, we explicitly split these steps into three separate LLM calls:

1. First, let the model reflect on the current conversation and generate a response.
2. Next, have the model output a clear index of the next agent (for easier parsing and execution).
3. Finally, pass the agent's name back to the model to generate a prompt that guides the agent to speak.

Additionally, if the model is directly prompted to decide whether to end the conversation, it may terminate immediately. To prevent this, we introduce Bernoulli random sampling to decide whether to stop. Based on the sample, we inject the corresponding prompt into the model, explicitly instructing it to continue or end the conversation, which improves the natural flow and continuity of the dialogue.

```python
class DirectorDialogueAgent(DialogueAgent):
    '''The role of the director, controlling the pace of the dialogue and the speaker'''
    def __init__(
        self,
        name,
        system_message,
        model,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.system_message = system_message
        self.next_speaker = ''
        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = 'Finish the conversation by stating a concluding message and thanking everyone.'
        self.continuation_clause = 'Do not end the conversation. Keep the conversation going by adding your own ideas.'
        self.response_prompt_template = ChatPrompter(
            instruction=f'system:Follow up with an insightful comment.\n{self.prefix}',
            extra_keys=['termination_clause'],
            history=[['{message_history}']]
        )
        self.choice_parser = IntegerOutputParser(
            regex=r'<(\d+)>', output_keys=['choice'], default_output_key='choice'
        )
        self.choose_next_speaker_prompt_template = ChatPrompter(
            instruction=({
                'user':
                'Given the above conversation, select the next speaker by choosing index next to their name:\n'
                '{speaker_names}\n\n'
                f'{self.choice_parser.get_format_instructions()}\n\n'
                'Respond ONLY with the number, no extra words.\n\n'
                'Generated number is not allowed to surpass the total number of {speaker_names}'
                'Do nothing else.'
            }),
            extra_keys=['speaker_names'],
            history=[['{message_history}']]
        )
        self.prompt_next_speaker_prompt_template = ChatPrompter(
            instruction=(
                'user:'
                'The next speaker is {next_speaker}.\n'
                'Prompt the next speaker to speak with an insightful question.\n'
                f'{self.prefix}'
            ),
            extra_keys=['next_speaker'],
            history=[['{message_history}']]
        )
        self.prompt_end_template = ChatPrompter(
            instruction=(
                'user: '
                "Provide a final witty summary that:\n"
                "Recaps the key satirical points about '{topic}'\n"
                'Ends with a memorable punchline\n'
                'Avoids introducing new topics\n'
                '*Use asterisks for physical gestures*\n'
                f'{self.prefix}'
            ),
            history=[['{message_history}']]
        )

    def _generate_response(self):
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability
        print(f'\tStop? {self.stop}\n')
        if self.stop:
            response_model = self.model.share(prompt=self.prompt_end_template)
        else:
            response_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
        self.response = response_model(self.system_message)
        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def _choose_next_speaker(self) -> str:
        speaker_names = '\n'.join(
            [f'{idx}: {name}' for idx, name in enumerate(self.speakers)]
        )
        choice_model = self.model.share(prompt=self.choose_next_speaker_prompt_template)
        choice_string = choice_model(
            self.system_message,
            speaker_names=speaker_names,
            message_history='\n'.join(
                self.message_history + [self.prefix] + [self.response]
            )
        )
        choice = int(self.choice_parser.parse(choice_string)['choice'])
        return choice

    def select_next_speaker(self):
        return self.chosen_speaker_id

    def send(self) -> str:
        self.response = self._generate_response()
        if self.stop:
            message = self.response
        else:
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print(f'\tNext speaker: {self.next_speaker}\n')
            message_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
            message = message_model(
                self.system_message, message_history=self.message_history
            )
            message = ' '.join([self.response, message])
        return message
```

### Discussion Topic and Role Setup

Define the episode's discussion topic, the director's name, and each agent's brief introduction and location.  
This information will be used later to generate system prompts and role background descriptions.

```python
# Set topic and role information
topic = 'The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze'
director_name = 'Jon Stewart'
word_limit = 50

agent_summaries = OrderedDict({
    'Jon Stewart': ('Host of the Daily Show', 'New York'),
    'Samantha Bee': ('Hollywood Correspondent', 'Los Angeles'),
    'Aasif Mandvi': ('CIA Correspondent', 'Washington D.C.'),
    'Ronny Chieng': ('Average American Correspondent', 'Cleveland, Ohio'),
})

agent_summary_string = '\n- '.join([''] + [f'{n}: {r}, located in {l}' for n, (r, l) in agent_summaries.items()])
conversation_description = f'This is a Daily Show episode discussing: {topic}.\nIt features {agent_summary_string}.'
agent_descriptor_system_message = 'You can add detail to the description of each person.'
```

### Generating Agent System Information

Use helper functions to generate detailed descriptions, role headers, and system messages for each agent.
The system message will be used as the prompt during agent initialization, determining their language style and behavioral tendencies.

```python
def generate_agent_description(agent_name, agent_role, agent_location):
    instruction = f'{agent_descriptor_system_message}\n'
    inputs = (
        f'{conversation_description}\n'
        f'Please reply with a creative description of {{agent_name}}, who is a {{agent_role}} in {{agent_location}}, '
        f'that emphasizes their particular role and location.\n'
        f'Speak directly to {{agent_name}} in {{word_limit}} words or less.\n'
        'Do not add anything else.'
    )
    prompter = ChatPrompter({'system': instruction})
    chat = OnlineChatModule().prompt(prompter)
    agent_description = chat(inputs)
    return agent_description


def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f'''{conversation_description}

    Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

    Your description is as follows: {agent_description}

    You are discussing the topic: {topic}.

    Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.
    '''


def generate_agent_system_message(agent_name, agent_header):
    return f'''{agent_header}
    You will speak in the style of {agent_name}, and exaggerate your personality.
    Do not say the same things over and over again.
    Speak in the first person from the perspective of {agent_name}
    For describing your own body movements, wrap your description in '*'.
    Do not change roles!
    Do not speak from the perspective of anyone else.
    Speak only from the perspective of {agent_name}.
    Stop speaking the moment you finish speaking from your perspective.
    Never forget to keep your response to {word_limit} words!
    Do not add anything else.
    '''


# Generate agent system information
agent_descriptions = [
    generate_agent_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    generate_agent_header(name, role, location, desc)
    for (name, (role, location)), desc in zip(agent_summaries.items(), agent_descriptions)
]
agent_system_messages = [
    generate_agent_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]

# Output agent information
for name, desc, header, sys_msg in zip(agent_summaries, agent_descriptions, agent_headers, agent_system_messages):
    print(f'\n{name} Description:\n{desc}\n\nHeader:\n{header}\n\nSystem Message:\n{sys_msg}')
```

### Generating a More Specific Discussion Topic

Use `ChatPrompter` to build a prompt template and have the language model expand the original topic into a more specific question form.  
Here, `OnlineChatModule` is used for online generation to ensure the output topic is concise and creative.

```python
# Generate topic
topic_specifier_prompt = ChatPrompter({'system:You can make a task more specific'})
topic_content = f'''{conversation_description}

        Please elaborate on the topic. 
        Frame the topic as a single question to be answered.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Do not add anything else.'''
chat_model = OnlineChatModule().prompt(topic_specifier_prompt)
specified_topic = chat_model(topic_content)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')
```

### Initializing the Director and Agent Roles

Use the generated system information to build the director agent (`DirectorDialogueAgent`) and multiple ordinary agents (`DialogueAgent`).

The director is responsible for controlling speaking order and termination conditions, while the other roles participate in the discussion according to their settings.

```python
# Initialize director and agents
director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=OnlineChatModule(),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.2,
)
agents = [director] + [
    DialogueAgent(name, sys_msg, OnlineChatModule())
    for name, sys_msg in zip(list(agent_summaries.keys())[1:], agent_system_messages[1:])
]
```

### Starting the Multi-Agent Dialogue Simulation

Finally, use `DialogueSimulator` to start the full multi-role dialogue simulation.  
The system first injects a question from the audience, then the director decides the next speaker and controls the turns until the termination condition is met.

```python
def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx


# Run the simulator
simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject('Audience member', specified_topic)
print(f'(Audience member): {specified_topic}\n')

while True:
    name, message = simulator.step()
    print(f'({name}): {message}\n')
    if director.stop:
        break
```

> The `select_next_speaker()` function determines who speaks next in a multi-turn dialogue: odd-numbered turns are assigned to a fixed agent, while even-numbered turns allow the director agent to dynamically choose the next speaker. This makes the conversation flow more like the rhythm and pacing of a real show.

## Full Code

The complete code is shown below:

<details>
<summary>Click to expand full code</summary>

```python
import re
import random
import functools
import tenacity
from collections import OrderedDict
from typing import List, Callable

from lazyllm import OnlineChatModule, ChatPrompter


# ======================
# Base class definition
# ======================

class DialogueAgent:
    '''Regular dialogue role Agent'''
    def __init__(
        self,
        name,
        system_message,
        model,
    ):
        self.name = name
        self.model = model
        self.prefix = f'{self.name}: '
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ['Host：Here is the conversation so far.']

    def send(self) -> str:
        structured_history = []
        for i in range(1, len(self.message_history), 2):
            if i + 1 < len(self.message_history):
                parts = self.message_history[i].split(':', 1)
                user_msg = parts[0]
                ai_msg = parts[1]
                structured_history.append([user_msg, ai_msg])

        history = structured_history + [[self.prefix, '']]
        message = self.model(self.system_message, llm_chat_history=history)
        return message

    def receive(self, name: str, message: str) -> None:
        self.message_history.append(f'{name}: {message}')


class DialogueSimulator:
    '''Dialogue simulator for managing multi-role interactions'''
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


class IntegerOutputParser:
    '''Parser for extracting integers from text'''
    def __init__(self, regex: str, output_keys: List[str], default_output_key: str):
        self.pattern = re.compile(regex)
        self.output_keys = output_keys
        self.default_output_key = default_output_key

    def parse(self, text: str):
        match = self.pattern.search(text)
        if not match:
            raise ValueError(f'No match found for regex {self.pattern.pattern} in text: {text}')
        groups = match.groups()
        if len(groups) != len(self.output_keys):
            raise ValueError(
                f'Expected {len(self.output_keys)} groups, but found {len(groups)}'
            )
        result = {}
        for key, value in zip(self.output_keys, groups):
            try:
                result[key] = int(value)
            except ValueError:
                raise ValueError(f"Matched value for key '{key}' is not a valid integer: {value}")
        return result

    def get_format_instructions(self) -> str:
        return 'Your response should be an integer delimited by angled brackets, like this: <int>.'

    def __call__(self, text: str):
        parsed = self.parse(text)
        return parsed.get(self.default_output_key)


class DirectorDialogueAgent(DialogueAgent):
    '''The role of the director, controlling the pace of the dialogue and the speaker'''
    def __init__(
        self,
        name,
        system_message,
        model,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.system_message = system_message
        self.next_speaker = ''
        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = 'Finish the conversation by stating a concluding message and thanking everyone.'
        self.continuation_clause = 'Do not end the conversation. Keep the conversation going by adding your own ideas.'
        self.response_prompt_template = ChatPrompter(
            instruction=f'system:Follow up with an insightful comment.\n{self.prefix}',
            extra_keys=['termination_clause'],
            history=[['{message_history}']]
        )
        self.choice_parser = IntegerOutputParser(
            regex=r'<(\d+)>', output_keys=['choice'], default_output_key='choice'
        )
        self.choose_next_speaker_prompt_template = ChatPrompter(
            instruction=({
                'user':
                'Given the above conversation, select the next speaker by choosing index next to their name:\n'
                '{speaker_names}\n\n'
                f'{self.choice_parser.get_format_instructions()}\n\n'
                'Respond ONLY with the number, no extra words.\n\n'
                'Generated number is not allowed to surpass the total number of {speaker_names}'
                'Do nothing else.'
            }),
            extra_keys=['speaker_names'],
            history=[['{message_history}']]
        )
        self.prompt_next_speaker_prompt_template = ChatPrompter(
            instruction=(
                'user:'
                'The next speaker is {next_speaker}.\n'
                'Prompt the next speaker to speak with an insightful question.\n'
                f'{self.prefix}'
            ),
            extra_keys=['next_speaker'],
            history=[['{message_history}']]
        )
        self.prompt_end_template = ChatPrompter(
            instruction=(
                'user: '
                "Provide a final witty summary that:\n"
                "Recaps the key satirical points about '{topic}'\n"
                'Ends with a memorable punchline\n'
                'Avoids introducing new topics\n'
                '*Use asterisks for physical gestures*\n'
                f'{self.prefix}'
            ),
            history=[['{message_history}']]
        )

    def _generate_response(self):
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability
        print(f'\tStop? {self.stop}\n')
        if self.stop:
            response_model = self.model.share(prompt=self.prompt_end_template)
        else:
            response_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
        self.response = response_model(self.system_message)
        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f'ValueError occurred: {retry_state.outcome.exception()}, retrying...'
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def _choose_next_speaker(self) -> str:
        speaker_names = '\n'.join(
            [f'{idx}: {name}' for idx, name in enumerate(self.speakers)]
        )
        choice_model = self.model.share(prompt=self.choose_next_speaker_prompt_template)
        choice_string = choice_model(
            self.system_message,
            speaker_names=speaker_names,
            message_history='\n'.join(
                self.message_history + [self.prefix] + [self.response]
            )
        )
        choice = int(self.choice_parser.parse(choice_string)['choice'])
        return choice

    def select_next_speaker(self):
        return self.chosen_speaker_id

    def send(self) -> str:
        self.response = self._generate_response()
        if self.stop:
            message = self.response
        else:
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print(f'\tNext speaker: {self.next_speaker}\n')
            message_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
            message = message_model(
                self.system_message, message_history=self.message_history
            )
            message = ' '.join([self.response, message])
        return message


# ======================
# Definition of auxiliary function
# ======================

def generate_agent_description(agent_name, agent_role, agent_location):
    instruction = f'{agent_descriptor_system_message}\n'
    inputs = (
        f'{conversation_description}\n'
        f'Please reply with a creative description of {{agent_name}}, who is a {{agent_role}} in {{agent_location}}, '
        f'that emphasizes their particular role and location.\n'
        f'Speak directly to {{agent_name}} in {{word_limit}} words or less.\n'
        'Do not add anything else.'
    )
    prompter = ChatPrompter({'system': instruction})
    chat = OnlineChatModule().prompt(prompter)
    agent_description = chat(inputs)
    return agent_description


def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f'''{conversation_description}

Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

Your description is as follows: {agent_description}

You are discussing the topic: {topic}.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.
'''


def generate_agent_system_message(agent_name, agent_header):
    return f'''{agent_header}
You will speak in the style of {agent_name}, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of {agent_name}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {agent_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    '''


def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx


# ======================
# Main logic execution part
# ======================

# Set topic and role information
topic = 'The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze'
director_name = 'Jon Stewart'
word_limit = 50

agent_summaries = OrderedDict({
    'Jon Stewart': ('Host of the Daily Show', 'New York'),
    'Samantha Bee': ('Hollywood Correspondent', 'Los Angeles'),
    'Aasif Mandvi': ('CIA Correspondent', 'Washington D.C.'),
    'Ronny Chieng': ('Average American Correspondent', 'Cleveland, Ohio'),
})

agent_summary_string = '\n- '.join([''] + [f'{n}: {r}, located in {l}' for n, (r, l) in agent_summaries.items()])
conversation_description = f'This is a Daily Show episode discussing: {topic}.\nIt features {agent_summary_string}.'
agent_descriptor_system_message = 'You can add detail to the description of each person.'

# Generate agent system information
agent_descriptions = [
    generate_agent_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    generate_agent_header(name, role, location, desc)
    for (name, (role, location)), desc in zip(agent_summaries.items(), agent_descriptions)
]
agent_system_messages = [
    generate_agent_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]

# Output agent information
for name, desc, header, sys_msg in zip(agent_summaries, agent_descriptions, agent_headers, agent_system_messages):
    print(f'\n{name} Description:\n{desc}\n\nHeader:\n{header}\n\nSystem Message:\n{sys_msg}')

# Generate topic
topic_specifier_prompt = ChatPrompter({'system:You can make a task more specific'})
topic_content = f'''{conversation_description}

        Please elaborate on the topic. 
        Frame the topic as a single question to be answered.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Do not add anything else.'''
chat_model = OnlineChatModule().prompt(topic_specifier_prompt)
specified_topic = chat_model(topic_content)

print(f'Original topic:\n{topic}\n')
print(f'Detailed topic:\n{specified_topic}\n')

# Initialize director and agents
director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=OnlineChatModule(),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.2,
)
agents = [director] + [
    DialogueAgent(name, sys_msg, OnlineChatModule())
    for name, sys_msg in zip(list(agent_summaries.keys())[1:], agent_system_messages[1:])
]

# Run the simulator
simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject('Audience member', specified_topic)
print(f'(Audience member): {specified_topic}\n')

while True:
    name, message = simulator.step()
    print(f'({name}): {message}\n')
    if director.stop:
        break
```
</details>

## Demonstration

```bash
(Audience member): How did competitive sitting, mocking the extreme of sedentary lifestyles, ironically become a viral fitness trend, discussed humorously by Jon Stewart, Samantha Bee, Aasif Mandvi, and Ronny Chieng on The Daily Show?


	Stop? False

	Next speaker: Ronny Chieng

(Jon Stewart): Jon Stewart: *leans in, eyes wide with faux excitement* Folks, we've gone from CrossFit to... Competitive Sitting! Because why stand when you can be the laziest fitness guru? Samantha, how's Hollywood taking this sitting pretty trend?

*leans back, grinning* Jon Stewart: *Leans in, eyes wide with mock seriousness.* "Competitive sitting? The fitness world has truly hit rock bottom... or the couch, rather. But, seriously, what's next? Competitive napping?"

*Turns to camera, raises an eyebrow.* "Samantha, how's Hollywood staying fit with this new trend?"


(Ronny Chieng): *leans back in chair, arms crossed* 

Folks, competitive sitting is the ultimate workout for the rest of us! Why run a marathon when you can master the art of doing nothing? In Cleveland, we're not just sitting; we're training for the gold in lounging!


	Stop? False

	Next speaker: Aasif Mandvi

(Jon Stewart): Jon Stewart: *leans in* Folks, we've gone from CrossFit to...Competitive Sitting? Next up, Samantha, how's Hollywood staying fit without moving? Jon Stewart: *leans in, eyes wide* So, the couch has become an Olympic event! Samantha, how's Hollywood taking this sitting pretty revolution?


(Aasif Mandvi): *leans in, eyes wide* 

Folks, in D.C., we sit so much, we've turned it into an Olympic sport! Competitive sitting: because why chase terrorists when you can just... sit on them?


	Stop? False

	Next speaker: Ronny Chieng

(Jon Stewart): Jon Stewart: *Leans back in his chair, grinning* Folks, we've gone from CrossFit to... Competitive Sitting! Who knew laziness would be the next Olympic sport? Samantha, how's Hollywood taking this sitting pretty trend?

### next_speaker Jon Stewart: 
So, we've turned sitting into a sport? Next, they'll monetize breathing! Samantha, what's Hollywood doing to one-up this absurdity?

*Looks at camera with a raised eyebrow.*


(Ronny Chieng): *leans back in chair, arms crossed* Folks, competitive sitting in Cleveland? We're not just lazy; we're champions! Forget treadmills, we're training for the gold in lounging. Why stand when you can conquer the world from your couch?


	Stop? False

	Next speaker: Ronny Chieng

(Jon Stewart): Jon Stewart: *leans in, eyes wide with mock seriousness* Folks, we've gone from CrossFit to... Competitive Sitting! Who knew laziness could be an Olympic sport? Samantha, how's Hollywood taking this sitting pretty trend?

### next_speaker: Samantha Bee Jon Stewart: 
Ladies and gentlemen, who needs a gym when you can master the art of doing nothing? Competitive sitting—because why stand when you can win by just sitting pretty? Samantha, what's Hollywood's take on this sedentary revolution?

*leans back in chair, smirking*


(Ronny Chieng): *leans back, arms crossed* Folks, in Cleveland, we turned sitting into a sport! Why run when you can rule from your recliner? Forget Fitbits, we're winning gold in lounging. Competitive sitting: the ultimate lazy Olympics!


	Stop? False

	Next speaker: Ronny Chieng

(Jon Stewart): Jon Stewart: 
Ladies and gentlemen, we've gone from CrossFit to just... sitting. How did this happen? Samantha Bee, what's Hollywood's take on turning sloth into sport?

 Jon Stewart: *leans in, eyes wide with mock seriousness* Folks, we've gone from CrossFit to... Competitive Sitting! Who knew laziness would be the next Olympic sport? Samantha, how's Hollywood staying fit without moving?

Samantha Bee: 



(Ronny Chieng): *leans back, arms crossed* Folks, in Cleveland, we turned sitting into a sport! Why run when you can rule from your recliner? Forget Fitbits, we're winning gold in lounging. Competitive sitting: the ultimate lazy Olympics!


	Stop? False

	Next speaker: Ronny Chieng

(Jon Stewart): Jon Stewart: 
*leans in with a smirk* Folks, we've gone from CrossFit to just... sitting. What's next, competitive napping? Samantha, how's Hollywood staying fit without moving?

### next_speaker: Samantha Bee *leans in, eyes wide with mock seriousness* So, we've turned sloth into sport. What's next, competitive napping? Samantha, how's Hollywood embracing this... "active" inactivity?



(Ronny Chieng): *leans back in chair, arms crossed* Folks, Cleveland's on the map for competitive sitting! Why jog when you can lounge? We're not just lazy; we're fitness pioneers. Forget gyms, we're winning gold in our La-Z-Boys. Competitive sitting: the next lazy Olympics!


	Stop? True

(Jon Stewart): Ladies and gentlemen, we've gone from CrossFit to just... sitting fit. *leans back, smirks* Laziness, the new black. Bee in LA, Mandvi in DC, Chieng in Cleveland—all sitting pretty. *chuckles* Who knew doing nothing could be so exhausting? *winks* Remember, kids: sit hard, sit often, sit... fashionably!
```

## Summary

This section demonstrated how to use LazyLLM to build a multi-agent dialogue system with "director-style scheduling" capabilities. By using `DirectorDialogueAgent` as the core control unit, the system can intelligently select the next speaker, balance the dialogue rhythm, and end the discussion at the appropriate time.

With a concise modular design and extensible prompt templates, this mechanism is not only suitable for hosting scenarios (such as talk shows or roundtable discussions) but can also be extended to multi-role interactions in business meetings, educational Q&A sessions, and other contexts.

Supported by LazyLLM, the complex process of coordinating multiple agents is simplified into clear logic and controllable behavior flows, making "dialogue orchestration" truly intelligent, flexible, and reusable.
