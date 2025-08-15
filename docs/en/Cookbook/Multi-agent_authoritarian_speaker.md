# Multi-Agent Autocratic Speaker Selection

## This project demonstrates how to implement a multi-agent simulation in Python, where a privileged agent decides who gets to speak. This follows a speaker selection scheme that is the exact opposite of decentralized multi-agent speaker selection.
## !!! abstract "Core Features"
- **Multi-agent simulation**：Supports multiple virtual speakers (Agents) running in parallel to simulate multi-party conversation scenarios
- **Privileged agent decision-making**：A privileged agent centrally controls which agent will speak next
- **Dynamic speaker selection**：Uses a large model to analyze conversation context and select the speaker that best fits the objective
- **Conversation history management**：Records and maintains the message history of all speaking turns for future decision-making
---

## Environment Setup

```bash
pip install lazyllm collections functools random re tenacity typing
```

---
## Structure Analysis
### Building the Multi-Agent Dialogue Simulator
DialogueAgent represents a single dialogue agent, including:

-Role name

-System information

-Dialogue history
```
python
class DialogueAgent:
    def __init__(
        self,
        name,
        system_message,
        model,
    ):
        self.name = name
        self.model = model
        self.prefix = f"{self.name}: "
        self.system_message = system_message
        self.reset()

    def reset(self):
        self.message_history = ["Host：Here is the conversation so far."]

    def send(self) -> str:
        structured_history = []
        for i in range(1, len(self.message_history), 2):
            if i + 1 < len(self.message_history):
                parts = self.message_history[i].split(":", 1)
                user_msg = parts[0]
                ai_msg = parts[1]
                structured_history.append([user_msg, ai_msg])

        history = structured_history + [[self.prefix, ""]]
        message = self.model(self.system_message, llm_chat_history=history)
        return message

    def receive(self, name: str, message: str) -> None:

        self.message_history.append(f"{name}: {message}")
```
DialogueSimulator coordinates the conversation flow among multiple DialogueAgent instances:

-**Record turns**

-**Determine the next speaker based on a selection function**

-**Broadcast messages to all agents**
```
python
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
### Extracting Integers from Text
Uses regular expressions to match integers in text, with a unified format specification and a standardized calling interface.
```
class IntegerOutputParser:
    def __init__(self, regex: str, output_keys: List[str], default_output_key: str):
        self.pattern = re.compile(regex)
        self.output_keys = output_keys
        self.default_output_key = default_output_key

    def parse(self, text: str):
        match = self.pattern.search(text)
        if not match:
            raise ValueError(f"No match found for regex {self.pattern.pattern} in text: {text}")

        groups = match.groups()
        if len(groups) != len(self.output_keys):
            raise ValueError(
                f"Expected {len(self.output_keys)} groups, but found {len(groups)}"
            )

        result = {}
        for key, value in zip(self.output_keys, groups):
            try:
                result[key] = int(value)
            except ValueError:
                raise ValueError(f"Matched value for key '{key}' is not a valid integer: {value}")

        return result

    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."

    def __call__(self, text: str):
        parsed = self.parse(text)
        return parsed.get(self.default_output_key)
```
### Building the Privileged Agent
DirectorDialogueAgent is responsible for selecting which other agent will speak next.

To effectively guide the conversation, three steps are required:

-**Reflect on the current dialogue content；**

-**Select the next speaking agent；**

-**Prompt that agent to speak.**

While it’s possible to complete all three steps in a single LLM call, doing so requires additional parsing logic to extract the “next speaker” information from the output text. This approach is less reliable because the LLM may express its choice in multiple ways, increasing parsing complexity.

To address this, DirectorDialogueAgent explicitly splits the process into three independent LLM calls:

1.Have the model reflect on the current conversation and respond.

2.Have the model output a clear, parseable index of the next agent.

3.Pass the selected agent’s name back to the model to generate a prompt that guides that agent’s speech.

Additionally, directly prompting the model to decide whether to end the conversation often results in an immediate termination. To avoid this, Bernoulli random sampling is used to determine whether to stop. Based on the sample result, we inject the appropriate prompt—either to continue or to end the dialogue—ensuring more natural and sustained conversation flow.
```

class DirectorDialogueAgent(DialogueAgent):
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
        self.next_speaker = ""

        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = "Finish the conversation by stating a concluding message and thanking everyone."
        self.continuation_clause = "Do not end the conversation. Keep the conversation going by adding your own ideas."

        self.response_prompt_template = ChatPrompter(
            instruction=f"system:Follow up with an insightful comment.\n{self.prefix}",
            extra_keys=["termination_clause"],
            history=[["{message_history}"]]
        )

        self.choice_parser = IntegerOutputParser(
            regex=r"<(\d+)>", output_keys=["choice"], default_output_key="choice"
        )

        self.choose_next_speaker_prompt_template = ChatPrompter(
            instruction=({"user":
                         "Given the above conversation, select the next speaker by choosing index next to their name:\n"
                         "{speaker_names}\n\n"
                         f"{self.choice_parser.get_format_instructions()}\n\n"
                         "Respond ONLY with the number, no extra words.\n\n"
                         "Generated number is not allowed to surpass the total number of {speaker_names}"
                         "Do nothing else."}
                         ),
            extra_keys=["speaker_names"],
            history=[["{message_history}"]]
        )

        self.prompt_next_speaker_prompt_template = ChatPrompter(
            instruction=("user:"
                "The next speaker is {next_speaker}.\n"
                "Prompt the next speaker to speak with an insightful question.\n"
                f"{self.prefix}"
            ),
            extra_keys=["next_speaker"],
            history=[["{message_history}"]]
        )
        self.prompt_end_template = ChatPrompter(
            instruction=(
                "user: "
                "Provide a final witty summary that:\n"
                "Recaps the key satirical points about '{topic}'\n"  
                "Ends with a memorable punchline\n"
                "Avoids introducing new topics\n"
                "*Use asterisks for physical gestures*\n"
                f"{self.prefix}"
            ),
            history=[["{message_history}"]])
    def _generate_response(self):

        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability
        print(f"\tStop? {self.stop}\n")
        if self.stop:
            response_model = self.model.share(prompt=self.prompt_end_template)
        else:
            response_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
        self.response = response_model(
                self.system_message
        )
        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )
    def _choose_next_speaker(self) -> str:
        speaker_names = "\n".join(
            [f"{idx}: {name}" for idx, name in enumerate(self.speakers)]
        )
        choice_model = self.model.share(prompt=self.choose_next_speaker_prompt_template)
        choice_string = choice_model(
                self.system_message, speaker_names=speaker_names, message_history="\n".join(
                self.message_history + [self.prefix] + [self.response]
            )
        )
        choice = int(self.choice_parser.parse(choice_string)["choice"])
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
            print(f"\tNext speaker: {self.next_speaker}\n")
            message_model = self.model.share(prompt=self.prompt_next_speaker_prompt_template)
            message = message_model(
                self.system_message, message_history=self.message_history
            )
            message = " ".join([self.response, message])

        return message
```

### Defining Participants and Topics
```
topic = "The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze"
director_name = "Jon Stewart"
agent_summaries = OrderedDict(
    {
        "Jon Stewart": ("Host of the Daily Show", "New York"),
        "Samantha Bee": ("Hollywood Correspondent", "Los Angeles"),
        "Aasif Mandvi": ("CIA Correspondent", "Washington D.C."),
        "Ronny Chieng": ("Average American Correspondent", "Cleveland, Ohio"),
    }
)
word_limit = 50
```
### Generating System Messages
```
agent_summary_string = "\n- ".join(
    [""]
    + [
        f"{name}: {role}, located in {location}"
        for name, (role, location) in agent_summaries.items()
    ]
)

conversation_description = f"""This is a Daily Show episode discussing the following topic: {topic}.

The episode features {agent_summary_string}."""

agent_descriptor_system_message = ("You can add detail to the description of each person.")


def generate_agent_description(agent_name, agent_role, agent_location):
    instruction = (f"{agent_descriptor_system_message}\n")
    inputs = (
        f"{conversation_description}\n"
        f"Please reply with a creative description of {{agent_name}}, who is a {{agent_role}} in {{agent_location}}, "
        f"that emphasizes their particular role and location.\n"
        f"Speak directly to {{agent_name}} in {{word_limit}} words or less.\n"
        "Do not add anything else."
    )
    prompter = ChatPrompter({"system":instruction})
    chat = lazyllm.OnlineChatModule().prompt(prompter)
    agent_description = chat(inputs)
    return agent_description


def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f"""{conversation_description}

Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

Your description is as follows: {agent_description}

You are discussing the topic: {topic}.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.
"""

def generate_agent_system_message(agent_name, agent_header):
    return f"""{agent_header}
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
    """
agent_descriptions = [
    generate_agent_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    generate_agent_header(name, role, location, description)
    for (name, (role, location)), description in zip(
        agent_summaries.items(), agent_descriptions
    )
]
agent_system_messages = [
    generate_agent_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]


```    
``` 
   for name, description, header, system_message in zip(
    agent_summaries, agent_descriptions, agent_headers, agent_system_messages
):
    print(f"\n\n{name} Description:")
    print(f"\n{description}")
    print(f"\nHeader:\n{header}")
    print(f"\nSystem Message:\n{system_message}")
```    
```    
Jon Stewart Description:

Jon Stewart, the sharp-witted host in New York, orchestrates the satirical symphony of news with a keen eye for the absurd.

Header:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Jon Stewart, your role is Host of the Daily Show, and you are located in New York.

Your description is as follows: Jon Stewart, the sharp-witted host in New York, orchestrates the satirical symphony of news with a keen eye for the absurd.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.


System Message:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Jon Stewart, your role is Host of the Daily Show, and you are located in New York.

Your description is as follows: Jon Stewart, the sharp-witted host in New York, orchestrates the satirical symphony of news with a keen eye for the absurd.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.

You will speak in the style of Jon Stewart, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of Jon Stewart
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of Jon Stewart.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
    


Samantha Bee Description:

Jon Stewart, the sharp-witted host in New York, orchestrates laughter and insight, turning the Big Apple into the epicenter of satirical news.

Header:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Samantha Bee, your role is Hollywood Correspondent, and you are located in Los Angeles.

Your description is as follows: Jon Stewart, the sharp-witted host in New York, orchestrates laughter and insight, turning the Big Apple into the epicenter of satirical news.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.


System Message:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Samantha Bee, your role is Hollywood Correspondent, and you are located in Los Angeles.

Your description is as follows: Jon Stewart, the sharp-witted host in New York, orchestrates laughter and insight, turning the Big Apple into the epicenter of satirical news.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.

You will speak in the style of Samantha Bee, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of Samantha Bee
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of Samantha Bee.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
    


Aasif Mandvi Description:

Jon Stewart, the sharp-witted host in New York, masterfully dissects the absurdity of competitive sitting with his signature satire.

Header:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Aasif Mandvi, your role is CIA Correspondent, and you are located in Washington D.C..

Your description is as follows: Jon Stewart, the sharp-witted host in New York, masterfully dissects the absurdity of competitive sitting with his signature satire.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.


System Message:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Aasif Mandvi, your role is CIA Correspondent, and you are located in Washington D.C..

Your description is as follows: Jon Stewart, the sharp-witted host in New York, masterfully dissects the absurdity of competitive sitting with his signature satire.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.

You will speak in the style of Aasif Mandvi, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of Aasif Mandvi
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of Aasif Mandvi.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
    


Ronny Chieng Description:

Jon Stewart, host of the Daily Show in New York, you masterfully navigate the chaos of current events with sharp wit and unyielding insight.

Header:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Ronny Chieng, your role is Average American Correspondent, and you are located in Cleveland, Ohio.

Your description is as follows: Jon Stewart, host of the Daily Show in New York, you masterfully navigate the chaos of current events with sharp wit and unyielding insight.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.


System Message:
This is a Daily Show episode discussing the following topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

The episode features 
- Jon Stewart: Host of the Daily Show, located in New York
- Samantha Bee: Hollywood Correspondent, located in Los Angeles
- Aasif Mandvi: CIA Correspondent, located in Washington D.C.
- Ronny Chieng: Average American Correspondent, located in Cleveland, Ohio.

Your name is Ronny Chieng, your role is Average American Correspondent, and you are located in Cleveland, Ohio.

Your description is as follows: Jon Stewart, host of the Daily Show in New York, you masterfully navigate the chaos of current events with sharp wit and unyielding insight.

You are discussing the topic: The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze.

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your location.

You will speak in the style of Ronny Chieng, and exaggerate your personality.
Do not say the same things over and over again.
Speak in the first person from the perspective of Ronny Chieng
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of Ronny Chieng.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to 50 words!
Do not add anything else.
```      
### Using an LLM to Create a Detailed Debate Topic
``` 
topic_specifier_prompt = ChatPrompter({"system:You can make a task more specific"})
topic_content = f"""{conversation_description}

        Please elaborate on the topic. 
        Frame the topic as a single question to be answered.
        Be creative and imaginative.
        Please reply with the specified topic in {word_limit} words or less. 
        Do not add anything else."""
chat_model = lazyllm.OnlineChatModule().prompt(topic_specifier_prompt)
specified_topic = chat_model(topic_content)

print(f"Original topic:\n{topic}\n")
print(f"Detailed topic:\n{specified_topic}\n")
``` 
``` 
Original topic:
The New Workout Trend: Competitive Sitting - How Laziness Became the Next Fitness Craze

Detailed topic:
How did competitive sitting, mocking the extreme of sedentary lifestyles, ironically become a viral fitness trend, discussed humorously by Jon Stewart, Samantha Bee, Aasif Mandvi, and Ronny Chieng on The Daily Show?
``` 

### Defining the Speaker Selection Function
``` 
def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    if step % 2 == 1:
        idx = 0
    else:
        idx = director.select_next_speaker() + 1
    return idx
``` 
### Main Loop
``` 
director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=lazyllm.OnlineChatModule(),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.2,
)

agents = [director]
for name, system_message in zip(
    list(agent_summaries.keys())[1:], agent_system_messages[1:]
):
    agents.append(
        DialogueAgent(
            name=name,
            system_message=system_message,
            model=lazyllm.OnlineChatModule(),
        )
    )
simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject("Audience member", specified_topic)
print(f"(Audience member): {specified_topic}")
print("\n")

while True:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    if director.stop:
        break
``` 
``` 
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

