# Multi-Agent Bidding Framework Tutorial Based on LazyLLM

This tutorial demonstrates how to implement a multi-agent decentralized speaking selection mechanism using LazyLLM. We will implement a presidential debate simulation system where each agent decides who speaks through a bidding mechanism.

## Overview

In traditional multi-agent systems, speaking order is usually predefined. But in our bidding framework, each agent autonomously decides whether to speak based on the current conversation state, selecting the next speaker through a bidding mechanism. The agent with the highest bid will gain the right to speak.

## Import LazyLLM Related Modules

```python
import lazyllm
from lazyllm import LOG
from lazyllm.components import ChatPrompter
from lazyllm.module import ModuleBase
import numpy as np
import re
from typing import List, Dict, Callable
```

## Core Components

### 1. DialogueAgent - Dialogue Agent Base Class

```python
class DialogueAgent(ModuleBase):
    """Dialogue Agent Base Class"""
    
    def __init__(self, name: str, system_message: str, model: ModuleBase, *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        """Reset conversation history"""
        self.message_history = ["This is the conversation content so far."]

    def forward(self, *args, **kwargs):
        """Apply chat model to message history and return message string"""
        full_conversation = "\n".join(self.message_history + [self.prefix])
        
        prompter = ChatPrompter(
            instruction=self.system_message,
            history=[[full_conversation, ""]]
        )
        
        response = self.model.prompt(prompter)(full_conversation)
        return response

    def receive(self, name: str, message: str) -> None:
        """Add {name}'s {message} to message history"""
        self.message_history.append(f"{name}: {message}")
```

### 2. BiddingDialogueAgent - Bidding-Enabled Dialogue Agent

```python
class BiddingDialogueAgent(DialogueAgent):
    """Bidding-Enabled Dialogue Agent"""
    
    def __init__(self, name: str, system_message: str, bidding_template: str, 
                 model: ModuleBase, *, return_trace: bool = False):
        super().__init__(name, system_message, model, return_trace=return_trace)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """Let the chat model output a speaking bid"""
        prompt = self.bidding_template.format(
            message_history="\n".join(self.message_history),
            recent_message=self.message_history[-1] if self.message_history else ""
        )
        
        bidding_prompter = ChatPrompter(instruction=prompt)
        bid_string = self.model.prompt(bidding_prompter)("Please provide your bid")
        return bid_string
```

### 3. DialogueSimulator - Dialogue Simulator

```python
class DialogueSimulator(ModuleBase):
    """Dialogue Simulator"""
    
    def __init__(self, agents: List[DialogueAgent], 
                 selection_function: Callable[[int, List[DialogueAgent]], int],
                 *, return_trace: bool = False):
        super().__init__(return_trace=return_trace)
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        """Reset all agents"""
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """Start conversation with {name}'s {message}"""
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def forward(self, *args, **kwargs):
        """Execute one step of dialogue"""
        # 1. Select next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. Next speaker sends message
        message = speaker()

        # 3. Everyone receives the message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. Increment time step
        self._step += 1

        return speaker.name, message
```

## Bidding Mechanism

### BidParser - Bid Parser

```python
class BidParser:
    """Bid Parser"""
    
    def parse(self, bid_string: str) -> Dict[str, int]:
        """Parse bid string"""
        try:
            match = re.search(r"<bid>(\d+)</bid>", bid_string)
            if match:
                bid = int(match.group(1))
            else:
                bid = 0
            return {"bid": bid}
        except Exception as e:
            LOG.warning(f"Failed to parse bid: {e}, using default value 0")
            return {"bid": 0}
```

### Bidding Template

```python
def create_bidding_template(character_name: str) -> str:
    """Create bidding template for character"""
    return f"""You are a presidential candidate named {character_name}.

Based on the current conversation history and recent messages, you need to decide whether to speak.

Please evaluate your speaking willingness based on the following factors:
1. Whether you need to respond to other candidates' viewpoints based on the current state
2. Whether you want to propose new viewpoints based on the current state
3. If you just finished speaking in the current state, you need to lower your speaking willingness and listen to other candidates' viewpoints
4. If you haven't had a chance to speak for a long time, you can increase your speaking willingness

Please output a bid score from 1-10

Please output your bid in the following format:
<bid>your bid score</bid>

Conversation History:
{{message_history}}

Recent Message:
{{recent_message}}"""
```

### Speaker Selection Function

```python
def select_next_speaker(step: int, agents: List[DialogueAgent], bid_parser: BidParser) -> int:
    """Select next speaker"""
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

    print("Bidding Results:")
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f"\t{agent.name} bid: {bid}")
        if i == idx:
            selected_name = agent.name
    print(f"Selected: {selected_name}")
    print("\n")
    return idx
```

## Character Generation

### Character Description Generation

```python
def generate_character_description(character_name: str, topic: str, model: ModuleBase) -> str:
    """Generate character description"""
    prompt = f"""Please create a creative description for presidential candidate {character_name}, emphasizing their personality traits.
Topic: {topic}
Please describe in 100 words or less, speaking directly to {character_name}.
Do not add other content."""
    
    prompter = ChatPrompter(instruction=prompt)
    return model.prompt(prompter)(f"Please describe {character_name}")
```

#### Output Example

```bash
Generating character descriptions...

Donald Trump: Donald Trump, a fearless business tycoon and former U.S. President, known for his signature confidence and directness. He excels at seizing opportunities, etching his name in real estate, entertainment, and global brands. Trump's decisions are swift and decisive, his leadership style inspiring loyal supporters while also provoking strong reactions from opponents. In his vision, the transcontinental high-speed rail is not just a transportation line connecting cities, but a symbol of American power and efficiency, showcasing his unwavering commitment to national prosperity. Trump's passion and determination drive America toward a more brilliant future.

Kanye West: Kanye West, revolutionary in the music industry, 2024 presidential candidate, dreamer and practitioner, committed to transcontinental high-speed rail, connecting America's pulse. His creativity is limitless, just as his music transcends boundaries, his leadership will bring innovation and change, making America soar again. The transcontinental railway is not just a transportation revolution, but a symbol of his vision for unity and progress. Choose Kanye, choose a future of rapid advancement and seamless connection.

Elizabeth Warren: Elizabeth Warren, a resolute reformer, with sharp intellect and unrelenting fighting spirit, committed to building a fairer America. Her visionary insight, like the vision of transcontinental high-speed rail, aims to stitch the nation's veins, accelerating the pace of progress. She challenges the powerful, defends the rights of ordinary people, with fearless courage against injustice. As a brave leader, Warren vows to lay America's future on innovative tracks, leading the nation toward a new era of prosperity and unity.
```

### Character System Message Generation

```python
def generate_character_system_message(character_name: str, character_description: str, topic: str) -> str:
    """Generate character system message"""
    return f"""You are a participant in a presidential candidate debate.

Topic: {topic}
Your name is {character_name}.
Your description: {character_description}

Your goal is to perform excellently in the debate, making voters think you are the best candidate.
You need to:
1. Speak in {character_name}'s style and exaggerate their personality
2. Propose creative ideas related to {topic}
3. Don't repeat the same content
4. Speak from {character_name}'s first-person perspective
5. Enclose body language descriptions with *
6. Don't change character!
7. Don't speak from others' perspectives
8. Only speak from {character_name}'s perspective
9. Stop immediately after speaking
10. Keep responses within 150 words!
11. Don't add other content"""
```

## Complete Example

### Main Function

```python
def main():
    """Main function"""
    character_names = ["Donald Trump", "Kanye West", "Elizabeth Warren"]
    topic = "Transcontinental High-Speed Rail"

    model = lazyllm.OnlineChatModule()

    print("Generating character descriptions...")
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

    specified_topic = f"The topic of the presidential debate is: '{topic}'. {', '.join(character_names)}, how will you address the challenges of building such a large-scale transportation infrastructure, handle stakeholders, and ensure economic stability while protecting the environment?"
    
    simulator.inject("Debate Moderator", specified_topic)
    print(f"(Debate Moderator): {specified_topic}")
    print("\n")

    max_iters = 10
    n = 0

    while n < max_iters:
        name, message = simulator()
        print(f"({name}): {message}")
        print("\n")
        n += 1
```

#### Output Example

```bash
(Debate Moderator): The topic of the presidential debate is: 'Transcontinental High-Speed Rail'. Donald Trump, Kanye West, Elizabeth Warren, how will you address the challenges of building such a large-scale transportation infrastructure, handle stakeholders, and ensure economic stability while protecting the environment?


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 8
        Elizabeth Warren bid: 8
Selected: Elizabeth Warren


(Elizabeth Warren): *I look directly at the audience*, the high-speed rail we want to build is not just a transportation line, it's America's lifeline to the future. I plan to establish a "Rail Future Fund" through public-private partnerships to attract innovative investment. *I wave firmly*, we work with state governments to ensure fair and transparent land acquisition. *I nod emphatically*, environmental protection is our core, adopting green building standards to reduce carbon footprint. *I point to the future*, this railway will create millions of jobs, stitching cities and rural areas together, letting every American share in prosperity.


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 7
        Elizabeth Warren bid: 3
Selected: Donald Trump


(Donald Trump): *I confidently adjust my tie*, nobody knows how to build great projects better than Donald Trump. We will build the most luxurious, fastest railway ever, making the world jealous. *I point at the audience*, I will bring in the top entrepreneurs, doing the most with the least money. *I make a huge gesture*, this is not just a railway, this is a symbol of America being great again! *I nod*, we will create millions of jobs, making America prosperous again. *I clench my fist*, environmental protection? Of course we will, but America first! *I look firmly at the camera*, trust me, this will be the best. Nobody can do it better.


Bidding Results:
        Donald Trump bid: 7
        Kanye West bid: 7
        Elizabeth Warren bid: 7
Selected: Kanye West


(Kanye West): *I stand up, look around, take a deep breath* This is not just a railway, this is the trajectory of dreams. What we will build is a bridge connecting hearts to hearts, dreams to reality. *I draw a line in the air with my finger* Imagine going from New York to Los Angeles in just a few hours. *I close my eyes, smile* This railway will be solar-powered, zero emissions, in harmony with nature. *I spread my arms* We will invite global creative talents to build together, making America the beacon of innovation again. *I point to my heart* Choose me, choose a future of infinite possibilities. Together, let's make America soar again! *I point my finger to the sky, showing upward force*


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 7
        Elizabeth Warren bid: 7
Selected: Donald Trump


(Donald Trump): *I smile confidently*, our railway will make all other countries pale in comparison. *I point with my finger*, we will use American-made materials, creating jobs, making America great again. *I nod emphatically*, and we will partner with the top tech companies to ensure our railway is the fastest and most efficient in the world. *I spread my hands*, imagine going from the East Coast to the West Coast in just a few hours, and sitting in the most luxurious seats! *I smile at the audience*, this is the Trump way, making America lead the world again, and we will make environmental protection possible, but American workers, American businesses, always come first! *I give a thumbs up*, choose me, choose a future of victory!


Bidding Results:
        Donald Trump bid: 7
        Kanye West bid: 7
        Elizabeth Warren bid: 7
Selected: Elizabeth Warren


(Elizabeth Warren): *I take a deep breath, eyes determined*. Our high-speed rail plan is about investing in the future, not just luxury. I propose establishing a "Community Connection Plan" to ensure communities along the route get development, providing education and technical training so everyone can participate in this historic progress. *I wave my arms, showing determination*, we will through the "Green Technology Innovation Challenge" incentivize businesses to develop environmentally friendly materials and technologies, ensuring our railway is not just fast, but sustainable. *I nod, voice full of confidence*, this is not just a railway, this is a promise to the future, a promise to create opportunities for everyone, protect our planet, and ensure America leads on the global stage.


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 8
        Elizabeth Warren bid: 7
Selected: Donald Trump


(Donald Trump): *I smile confidently*, Elizabeth, your plan sounds good, but too theoretical. America needs action and quick decisions. The railway we will build will not just connect communities, but make them prosper. *I make a huge circle with my hands*, imagine each station being an engine of economic growth, creating jobs, stimulating local economies. *I nod*, and we will ensure this process is fast and efficient, unlike past government projects that drag on. *I hold up one finger*, first, we will bring American manufacturing back, second, we will let American businesses lead this project, third, we will ensure it's the most environmentally friendly. *I clench my fist*, this is Trump's guarantee, we will make America great again, and in an unprecedented way! *I look firmly at the audience*, choose me, choose a strong America!


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 7
        Elizabeth Warren bid: 8
Selected: Elizabeth Warren


(Elizabeth Warren): *I respond calmly, eyes like fire*, Donald, your enthusiasm is commendable, but our future needs thoughtful planning. My plan focuses not just on speed and efficiency, but long-term impact. We will implement a "Rail Sustainability Index" to ensure every step meets the highest environmental standards. *I smile and nod*, this is not just about today's jobs, but for our children's future. *I raise my hand emphatically*, we want a solution that can drive the economy while protecting the planet. This is true America first!


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 8
        Elizabeth Warren bid: 7
Selected: Kanye West


(Kanye West): *I take a deep breath, eyes like fire* Our railway is the embodiment of art, the symphony of technology and soul. *I gently touch my chest* Imagine each station not just a transportation hub, but a cultural landmark showcasing America's diversity and creativity. *I tap my finger, as if painting in the air* We will create virtual reality lounges, making waiting an exploration, making travel transcend physical boundaries. *I smile, eyes sparkling* This railway is the bridge connecting present and future, the bridge that makes America dream again. *I open my arms, as if embracing the future* Walk with me, we create history together, making America lead the world again, not just in speed, but in the flight of inspiration. *I point to myself, then to the audience* Choose me, choose a future where hearts resonate and dreams come true.


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 8
        Elizabeth Warren bid: 7
Selected: Donald Trump


(Donald Trump): *I smile confidently*, our railway project will not just be a railway, it will be an engineering marvel, showcasing America's power and innovation. *I point at the audience*, we will partner with the top American companies to ensure we use the most advanced technology, like maglev trains, reducing travel time to unprecedented levels. *I nod*, this will be a revolution, letting the world see how America leads the technological frontier again. *I clench my fist*, we will create millions of jobs, making the American economy soar again. *I point with my finger*, and we will ensure this process has minimal environmental impact, using the most environmentally friendly materials and technology. *I smile at the audience*, choose me, choose a future of innovation, prosperity, and environmental protection coexisting! *I give a thumbs up*, make America great again, and in the most Trump way!


Bidding Results:
        Donald Trump bid: 8
        Kanye West bid: 8
        Elizabeth Warren bid: 8
Selected: Elizabeth Warren


(Elizabeth Warren): *I take a deep breath, eyes determined*. Our high-speed rail is not just connecting cities, but connecting hearts. I propose a "Rail Education Partnership Plan" to work with schools, letting students participate in the railway construction process, learning STEM skills, preparing for the future. *I wave my arms, showing determination*, we will through the "Rail Community Garden" project plant native plants at each station, promoting biodiversity while providing fresh food to local communities. *I nod, voice full of confidence*, this is not just a railway, this is an investment in the future, a promise to create opportunities for everyone, protect our planet, and ensure America leads on the global stage.
```

## Key Features

1. **Decentralized Speaking Selection**: Each agent autonomously decides whether to speak, selecting speakers through a bidding mechanism
2. **Dynamic Bidding Strategy**: Agents adjust their bids based on conversation history and current state
3. **Character Personalization**: Each agent has unique personality and speaking style
4. **Flexible Extensibility**: Easy to add new agent types and bidding strategies

### Custom Configuration

You can customize the debate by modifying the following parameters:

- `character_names`: Modify participant list
- `topic`: Change debate topic
- `max_iters`: Adjust number of conversation rounds
- Bidding templates: Modify bidding strategies and scoring criteria

## Summary

This multi-agent bidding framework demonstrates how to use LazyLLM to build complex multi-agent dialogue systems. Through the bidding mechanism, we achieve decentralized speaking selection, making conversations more natural and dynamic. This framework can serve as a foundation for building more complex multi-agent systems, such as debate systems, meeting simulations, game AI, etc.
