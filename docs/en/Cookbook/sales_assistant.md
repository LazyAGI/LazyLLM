# Context-Aware AI Sales Assistant

This example demonstrates how to use LazyLLM to quickly build an intelligent Sales Assistant with conversation understanding and stage awareness.  

The assistant can automatically determine the current stage of a sales conversation based on context and generate professional responses accordingly.

!!! abstract "In this section, you will learn how to build a context-aware sales assistant, including the following key points:"

    - How to use [OnlineChatModule][lazyllm.module.OnlineChatModule] to build the core language understanding and generation module for sales dialogues.
    - How to implement `SalesStageAnalyzer` to automatically identify conversation stages such as “Introduction,” “Needs Analysis,” and “Objection Handling.”
    - How to implement `SalesConversationAgent` to generate natural sales dialogue based on stage and conversation history.
    - How to use Sale`sGPT as the main controller to realize the complete cycle of “Stage Analysis → Response Generation → Dialogue Update.”
    - How to launch an interactive demonstration with the main function `main()` for an end-to-end sales simulation experience.

## Design Concept

To build a smart sales assistant that dynamically adjusts its responses based on context, the system must both understand sales logic and grasp conversation context.  

Thus, the design of `SalesGPT` revolves around two core goals: stage recognition and intelligent sales response generation.

First, we use LazyLLM’s `OnlineChatModule` as the core language model to understand the conversation and generate natural, professional sales replies.

Then, the system is divided into two key modules:

- `SalesStageAnalyzer`: Determines which sales stage the conversation is currently in (e.g., Introduction, Needs Analysis, Objection Handling).
- `SalesConversationAgent`: Generates contextually appropriate responses for each stage and appends `<END_OF_TURN>` to control conversation flow.

Finally, the main controller `SalesGPT` integrates both modules, maintaining dialogue history, recognizing stages, and generating replies in a complete Analyze → Generate → Update loop.

Overall workflow:

![sales_assistant](../assets/sales_assistant.png)

## Environment Setup

### Install Dependencies

Run the following command to install required libraries:

```bash
pip install lazyllm
```

### Import Dependencies

```python
from lazyllm import OnlineChatModule
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
```

### Environment Variables

This process requires an online LLM. Set your API key (using Qwen as an example):

```bash
export LAZYLLM_QWEN_API_KEY = "sk-******"
```

> ❗ Note: Refer to the [official documentation](docs.lazyllm.ai/) for API key application details.

## Code Implementation

### Sales Stage Analyzer

In a sales conversation, each stage represents a different relationship development phase between salesperson and client.

For instance, the “Introduction” phase focuses on building trust, while later stages like “Objection Handling” or “Closing” emphasize persuasion and action.

Thus, we need a module to automatically identify the current stage so the assistant can respond appropriately.

The following `SalesStageAnalyzer` class is designed for this purpose:

```python
class SalesStageAnalyzer(ModuleBase):
    '''Sales conversation stage analyzer that identifies the current stage.'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # Define sales conversation stages
        self.conversation_stages = {
            '1': 'Introduction: Start the conversation, introduce yourself and your company. Be polite and professional.',
            '2': 'Qualification: Confirm if the lead is a suitable prospect and has decision-making authority.',
            '3': 'Value Proposition: Explain how the product/service benefits the prospect and highlight unique selling points.',
            '4': 'Needs Analysis: Use open-ended questions to understand the client’s needs and pain points.',
            '5': 'Solution Presentation: Present your product/service as a solution based on the client’s needs.',
            '6': 'Objection Handling: Address any concerns and provide supporting evidence.',
            '7': 'Closing: Propose next steps like a demo, trial, or meeting with decision-makers.'
        }

        # Stage analysis prompt
        stage_analyzer_prompt = '''You are a sales assistant helping a salesperson determine which stage the conversation should move to next.

        Conversation history:
        ===
        {conversation_history}
        ===

        Determine the next conversation stage from these options:
        1. Introduction
        2. Qualification
        3. Value Proposition
        4. Needs Analysis
        5. Solution Presentation
        6. Objection Handling
        7. Closing

        Reply with a single number (1–7) only.  
        If there is no conversation history, output 1.  
        Do not include any extra text.'''

        self.prompter = ChatPrompter(instruction=stage_analyzer_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, conversation_history: str) -> str:
        '''Analyze conversation and return the current stage.'''
        response = self.llm({'conversation_history': conversation_history})
        stage_id = ''.join(filter(str.isdigit, response.strip()))
        if not stage_id or int(stage_id) not in range(1, 8):
            stage_id = '1'

        if self.verbose:
            print(f'Stage analysis result: {stage_id} - {self.conversation_stages[stage_id]}')

        return stage_id
```

**Context Caching / Session Isolation**

LazyLLM supports module-level context caching.

Using `.used_by()`, the same LLM instance is separated by caller ID so that:

- Each module caches its own prompt–response pairs independently.
- Contexts are isolated, ensuring clean session management.

> 💡 Note: `ModuleBase` is the core base class in LazyLLM.
> All modules (dialogue generators, analyzers, retrievers, etc.) inherit from it and share unified interfaces and life cycles.
> See [API Reference](https://docs.lazyllm.ai/en/stable/API%20Reference/module/#lazyllm.module.ModuleBase) for details.

### Sales Conversation Agent

Once we know the sales stage, we need an agent that generates natural replies suitable for that phase. The `SalesConversationAgent` module allows the AI to act like a salesperson, producing appropriate sales talk based on stage and context.

```python
class SalesConversationAgent(ModuleBase):
    '''Sales conversation agent that generates replies based on the current stage'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # Sales conversation prompt template
        sales_conversation_prompt = '''Your name is {salesperson_name}, and you are a {salesperson_role}.
        You work at {company_name}. The company's business is: {company_business}
        The company values are: {company_values}
        The purpose of contacting the potential client is: {conversation_purpose}
        The type of contact is: {conversation_type}

        If asked where you got the user's contact information, say it was obtained from public records.
        Keep responses short to maintain user attention. Do not generate lists—just answer directly.
        You must respond based on the previous conversation history and the current conversation stage.
        Generate only one reply at a time! End each reply with '<END_OF_TURN>' to allow the user to respond.

        Current conversation stage: {conversation_stage}
        Conversation history:
        {conversation_history}
        {salesperson_name}:'''

        self.prompter = ChatPrompter(instruction=sales_conversation_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, salesperson_name: str, salesperson_role: str, company_name: str,
                company_business: str, company_values: str, conversation_purpose: str,
                conversation_type: str, conversation_stage: str, conversation_history: str) -> str:
        
        # Generate sales conversation reply
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

The `SalesConversationAgent` module serves the following purposes:

- Receives the current sales stage (e.g., *Needs Analysis* or *Closing*);
- Generates a natural, concise, and logically appropriate response based on the salesperson’s role, company background, and conversation goal;
- Adapts responses dynamically to maintain realistic sales interactions.

Compared with the `SalesStageAnalyzer`, this module focuses on *how to say* rather than *which stage to move to*.

Together, they form a complete and adaptive sales dialogue loop.

> 💡 Tip: When combined with the “Sales Stage Analyzer” from the previous section, the system can achieve a full intelligent sales simulation with *automatic stage recognition + smart response generation*.

### Main Controller

In the previous sections, we implemented:

- The *Stage Analyzer* (`SalesStageAnalyzer`): determines which stage the current sales conversation is in;
- The *Sales Conversation Agent* (`SalesConversationAgent`): generates contextually appropriate sales replies based on the stage.

Next, we will integrate them into a unified main controller to build a context-aware, interactive Sales AI Assistant.

```python
class SalesGPT(ModuleBase):
    '''Context-aware AI Sales Assistant main controller'''

    def __init__(
        self,
        llm: ModuleBase,
        salesperson_name: str = 'Zhang Sales',
        salesperson_role: str = 'Business Development Representative',
        company_name: str = 'Premium Sleep',
        company_business: str = 'Premium Sleep is a high-end mattress company...',
        company_values: str = 'The mission of Premium Sleep is to provide the best sleep solutions...',
        conversation_purpose: str = 'To learn whether they wish to improve sleep quality by purchasing a premium mattress.',
        conversation_type: str = 'phone',
        verbose: bool = True
    ):
        super().__init__()

        # Salesperson information
        self.salesperson_name = salesperson_name
        self.salesperson_role = salesperson_role
        self.company_name = company_name
        self.company_business = company_business
        self.company_values = company_values
        self.conversation_purpose = conversation_purpose
        self.conversation_type = conversation_type
        self.verbose = verbose

        # Initialize state
        self.conversation_history = []
        self.current_conversation_stage = '1'

        # Initialize components
        self.stage_analyzer = SalesStageAnalyzer(llm, verbose=verbose)
        self.sales_conversation_agent = SalesConversationAgent(llm, verbose=verbose)

        # Conversation stage definitions
        self.conversation_stages = {
            '1': 'Introduction: Start the conversation, introduce yourself and the company. Maintain a polite and professional tone.',
            '2': 'Qualification: Confirm whether the potential customer is the right contact and has decision-making authority.',
            '3': 'Value Proposition: Briefly explain how the product/service benefits the potential customer and highlight unique selling points.',
            '4': 'Needs Analysis: Use open-ended questions to understand the potential customer’s needs and pain points.',
            '5': 'Solution Presentation: Present the product/service as a solution based on the customer’s needs.',
            '6': 'Objection Handling: Address any objections about the product/service and provide supporting evidence.',
            '7': 'Closing: Request a next step in the sales process, such as a demo, trial, or meeting with a decision-maker.'
        }

    def seed_agent(self):
        '''Initialize the sales agent'''
        if self.verbose:
            print(f'{self.salesperson_name}: (Waiting for user input...)')

    def determine_conversation_stage(self):
        '''Determine which stage the current conversation should be in'''
        conversation_text = '\n'.join(self.conversation_history)
        self.current_conversation_stage = self.stage_analyzer(conversation_text)
        if self.verbose:
            print(f'Current conversation stage: {self.conversation_stages[self.current_conversation_stage]}')

    def human_step(self, human_input: str):
        '''Handle human input'''
        human_input = human_input + '<END_OF_TURN>'
        self.conversation_history.append(human_input)
        if self.verbose:
            print(f'User: {human_input.replace('<END_OF_TURN>', '')}')

    def step(self):
        '''Execute one step of the sales agent’s conversation'''
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

`SalesGPT` is responsible for managing the state and flow control of the sales conversation. Its main responsibilities include:

- Managing conversation history;
- Dynamically determining the current sales stage;
- Letting the AI generate appropriate sales responses at each stage;
- Enabling alternating interactions between human input and model responses.

In other words, it acts as a *“sales director”* —
ensuring that *each sales stage transitions naturally and the AI always says the right thing.*

### Main Function

Now, let's demonstrate how to run a complete `SalesGPT` intelligent sales assistant through a main function.

The main function accomplishes the following tasks:

- Create an LLM instance: Load the conversational model to generate sales dialogue and intelligent responses;
- Configure sales agent parameters: Define salesperson identity, company information, and conversation goals;
- Initialize the sales agent: Provide the model with background knowledge and stage awareness;
- Start the interaction loop: Simulate a realistic customer conversation process.

```python
def main():
    '''Main function: Demonstration of SalesGPT usage'''
    print('=== Context-Aware AI Sales Assistant Demo ===\n')

    # Set up the LLM
    llm = OnlineChatModule()

    # Configure the sales agent
    config = {
        'salesperson_name': 'Li Sales',
        'salesperson_role': 'Business Development Representative',
        'company_name': 'Premium Sleep',
        'company_business': 'Premium Sleep is a high-end mattress company that provides customers with the most comfortable and supportive sleep experience.',
        'company_values': 'The mission of Premium Sleep is to help people achieve better rest by offering the best sleep solutions.',
        'conversation_purpose': 'To learn whether they wish to improve sleep quality by purchasing a premium mattress.',
        'conversation_type': 'phone'
    }

    # Create the sales agent
    sales_agent = SalesGPT(llm, **config)

    # Initialize the agent
    print('Initializing sales agent...')
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()

    # Start the conversation loop
    print('\nStarting sales conversation...')
    while True:
        sales_agent.step()
        user_input = input("\nPlease enter your reply (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print('Conversation ended. Thank you for using SalesGPT!')
            break
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()

if __name__ == '__main__':
    main()
```

## Full Code

The complete code is shown below:

<details>
<summary>Click to expand full code</summary>

```python
from lazyllm import OnlineChatModule
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter


class SalesStageAnalyzer(ModuleBase):
    '''Sales conversation stage analyzer that identifies the current stage.'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # Define sales conversation stages
        self.conversation_stages = {
            '1': 'Introduction: Start the conversation, introduce yourself and your company. Be polite and professional.',
            '2': 'Qualification: Confirm if the lead is a suitable prospect and has decision-making authority.',
            '3': 'Value Proposition: Explain how the product/service benefits the prospect and highlight unique selling points.',
            '4': 'Needs Analysis: Use open-ended questions to understand the client’s needs and pain points.',
            '5': 'Solution Presentation: Present your product/service as a solution based on the client’s needs.',
            '6': 'Objection Handling: Address any concerns and provide supporting evidence.',
            '7': 'Closing: Propose next steps like a demo, trial, or meeting with decision-makers.'
        }

        # Stage analysis prompt
        stage_analyzer_prompt = '''You are a sales assistant helping a salesperson determine which stage the conversation should move to next.

        Conversation history:
        ===
        {conversation_history}
        ===

        Determine the next conversation stage from these options:
        1. Introduction
        2. Qualification
        3. Value Proposition
        4. Needs Analysis
        5. Solution Presentation
        6. Objection Handling
        7. Closing

        Reply with a single number (1–7) only.  
        If there is no conversation history, output 1.  
        Do not include any extra text.'''

        self.prompter = ChatPrompter(instruction=stage_analyzer_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, conversation_history: str) -> str:
        '''Analyze conversation and return the current stage.'''
        response = self.llm({'conversation_history': conversation_history})
        stage_id = ''.join(filter(str.isdigit, response.strip()))
        if not stage_id or int(stage_id) not in range(1, 8):
            stage_id = '1'

        if self.verbose:
            print(f'Stage analysis result: {stage_id} - {self.conversation_stages[stage_id]}')

        return stage_id


class SalesConversationAgent(ModuleBase):
    '''Sales conversation agent that generates replies based on the current stage'''

    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

        # Sales conversation prompt template
        sales_conversation_prompt = '''Your name is {salesperson_name}, and you are a {salesperson_role}.
        You work at {company_name}. The company's business is: {company_business}
        The company values are: {company_values}
        The purpose of contacting the potential client is: {conversation_purpose}
        The type of contact is: {conversation_type}

        If asked where you got the user's contact information, say it was obtained from public records.
        Keep responses short to maintain user attention. Do not generate lists—just answer directly.
        You must respond based on the previous conversation history and the current conversation stage.
        Generate only one reply at a time! End each reply with '<END_OF_TURN>' to allow the user to respond.

        Current conversation stage: {conversation_stage}
        Conversation history:
        {conversation_history}
        {salesperson_name}:'''

        self.prompter = ChatPrompter(instruction=sales_conversation_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)

    def forward(self, salesperson_name: str, salesperson_role: str, company_name: str,
                company_business: str, company_values: str, conversation_purpose: str,
                conversation_type: str, conversation_stage: str, conversation_history: str) -> str:
        
        # Generate sales conversation reply
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


class SalesGPT(ModuleBase):
    '''Context-aware AI Sales Assistant main controller'''

    def __init__(
        self,
        llm: ModuleBase,
        salesperson_name: str = 'Zhang Sales',
        salesperson_role: str = 'Business Development Representative',
        company_name: str = 'Premium Sleep',
        company_business: str = 'Premium Sleep is a high-end mattress company...',
        company_values: str = 'The mission of Premium Sleep is to provide the best sleep solutions...',
        conversation_purpose: str = 'To learn whether they wish to improve sleep quality by purchasing a premium mattress.',
        conversation_type: str = 'phone',
        verbose: bool = True
    ):
        super().__init__()

        # Salesperson information
        self.salesperson_name = salesperson_name
        self.salesperson_role = salesperson_role
        self.company_name = company_name
        self.company_business = company_business
        self.company_values = company_values
        self.conversation_purpose = conversation_purpose
        self.conversation_type = conversation_type
        self.verbose = verbose

        # Initialize state
        self.conversation_history = []
        self.current_conversation_stage = '1'

        # Initialize components
        self.stage_analyzer = SalesStageAnalyzer(llm, verbose=verbose)
        self.sales_conversation_agent = SalesConversationAgent(llm, verbose=verbose)

        # Conversation stage definitions
        self.conversation_stages = {
            '1': 'Introduction: Start the conversation, introduce yourself and the company. Maintain a polite and professional tone.',
            '2': 'Qualification: Confirm whether the potential customer is the right contact and has decision-making authority.',
            '3': 'Value Proposition: Briefly explain how the product/service benefits the potential customer and highlight unique selling points.',
            '4': 'Needs Analysis: Use open-ended questions to understand the potential customer’s needs and pain points.',
            '5': 'Solution Presentation: Present the product/service as a solution based on the customer’s needs.',
            '6': 'Objection Handling: Address any objections about the product/service and provide supporting evidence.',
            '7': 'Closing: Request a next step in the sales process, such as a demo, trial, or meeting with a decision-maker.'
        }

    def seed_agent(self):
        '''Initialize the sales agent'''
        if self.verbose:
            print(f'{self.salesperson_name}: (Waiting for user input...)')

    def determine_conversation_stage(self):
        '''Determine which stage the current conversation should be in'''
        conversation_text = '\n'.join(self.conversation_history)
        self.current_conversation_stage = self.stage_analyzer(conversation_text)
        if self.verbose:
            print(f'Current conversation stage: {self.conversation_stages[self.current_conversation_stage]}')

    def human_step(self, human_input: str):
        '''Handle human input'''
        human_input = human_input + '<END_OF_TURN>'
        self.conversation_history.append(human_input)
        if self.verbose:
            print(f'User: {human_input.replace('<END_OF_TURN>', '')}')

    def step(self):
        '''Execute one step of the sales agent’s conversation'''
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


def main():
    '''Main function: Demonstration of SalesGPT usage'''
    print('=== Context-Aware AI Sales Assistant Demo ===\n')

    # Set up the LLM
    llm = OnlineChatModule()

    # Configure the sales agent
    config = {
        'salesperson_name': 'Li Sales',
        'salesperson_role': 'Business Development Representative',
        'company_name': 'Premium Sleep',
        'company_business': 'Premium Sleep is a high-end mattress company that provides customers with the most comfortable and supportive sleep experience.',
        'company_values': 'The mission of Premium Sleep is to help people achieve better rest by offering the best sleep solutions.',
        'conversation_purpose': 'To learn whether they wish to improve sleep quality by purchasing a premium mattress.',
        'conversation_type': 'phone'
    }

    # Create the sales agent
    sales_agent = SalesGPT(llm, **config)

    # Initialize the agent
    print('Initializing sales agent...')
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()

    # Start the conversation loop
    print('\nStarting sales conversation...')
    while True:
        sales_agent.step()
        user_input = input("\nPlease enter your reply (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print('Conversation ended. Thank you for using SalesGPT!')
            break
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()

if __name__ == '__main__':
    main()
```
</details>

## Demonstration

Next, let's take a look at the actual running effect of `SalesGPT`.

```bash
=== Context-Aware AI Sales Assistant Demo ===

Initializing sales agent...
Li Sales: (Waiting for user input...)
Conversation stage analysis result: 1 - Introduction: Start the conversation, introduce yourself and your company. Maintain a polite and professional tone.  
Current stage: Introduction: Start the conversation, introduce yourself and your company. Maintain a polite and professional tone. Your greeting should be welcoming and clearly state the reason for contacting the potential customer.

Starting sales conversation...
Li Sales: Hello! This is Li Sales from Quality Sleep. We specialize in high-end mattresses and sleep solutions that help people improve their sleep quality. May I ask if you’ve been looking for ways to enhance your sleep experience recently?

Please enter your reply (type 'quit' to exit): Yes  
User: Yes  
Conversation stage analysis result: 4 - Needs Analysis: Understand the potential customer's needs and pain points through open-ended questions.  
Current stage: Needs Analysis: Understand the potential customer's needs and pain points through open-ended questions. Listen carefully to their response and take notes.  
Li Sales: That’s great to hear that you’re interested in improving your sleep experience. May I ask what kind of sleep issues you’re currently facing, or what specific improvements you hope to achieve?

Please enter your reply (type 'quit' to exit): I have trouble sleeping at night and often suffer from insomnia.  
User: I have trouble sleeping at night and often suffer from insomnia.  
Conversation stage analysis result: 4 - Needs Analysis: Understand the potential customer's needs and pain points through open-ended questions.  
Current stage: Needs Analysis: Understand the potential customer's needs and pain points through open-ended questions. Listen carefully to their response and take notes.  
Li Sales: I understand. Having trouble sleeping and experiencing insomnia can really affect your quality of life. Have you tried using any special types of mattresses or pillows to help with this issue?

Please enter your reply (type 'quit' to exit): Not yet.  
User: Not yet.  
Conversation stage analysis result: 3 - Value Proposition: Briefly explain how the product/service benefits the potential customer and highlight unique selling points.  
Current stage: Value Proposition: Briefly explain how the product/service benefits the potential customer and highlight unique selling points.  
Li Sales: In that case, let me introduce how our products can help you. Our mattresses use advanced materials and design to provide personalized body support, reduce tossing and turning, and help you fall asleep faster and sleep more deeply. In addition, our pillows are made from memory foam that adapts to your head and neck shape for better comfort. Many of our customers have reported significant improvements in their sleep quality. Would you like to learn more?

Please enter your reply (type 'quit' to exit): Tell me more about your products and prices.  
User: Tell me more about your products and prices.  
Conversation stage analysis result: 5 - Solution Presentation: Present your product/service as the solution based on the potential customer’s needs.  
Current stage: Solution Presentation: Present your product/service as the solution based on the potential customer’s needs.  
Li Sales: Of course. Our product line includes three main series: Classic, Luxury, and Custom.

1. **Classic Series**: Suitable for customers seeking comfort and support. Prices range from ¥5,000 to ¥10,000.  
2. **Luxury Series**: Uses premium materials and advanced technology for an ultimate sleep experience. Prices range from ¥10,000 to ¥20,000.  
3. **Custom Series**: Fully personalized solutions tailored to your body and needs. Prices start at ¥20,000.  

All our mattresses and pillows come with warranty coverage to ensure worry-free use. If you’re interested, we can arrange a free sleep consultation to better understand your needs and provide tailored recommendations. How does that sound?

Please enter your reply (type 'quit' to exit): Too expensive.  
User: Too expensive.  
Conversation stage analysis result: 6 - Objection Handling: Address any concerns or objections from the potential customer with supporting evidence.  
Current stage: Objection Handling: Address any concerns or objections from the potential customer with supporting evidence or testimonials.  
Li Sales: I understand your concern — high-end mattresses do cost more than standard ones. However, considering that our products can significantly improve your sleep quality and reduce insomnia, it’s truly a worthwhile investment in the long run. Quality sleep has a huge positive impact on health and productivity, helping you perform better in daily life and work.  

Moreover, we offer a free trial period and warranty service to ensure you have full confidence before making a decision. If you’d like, we can schedule a free sleep consultation to give you more personalized advice. What do you think?

Please enter your reply (type 'quit' to exit): Sounds good.  
User: Sounds good.  
Conversation stage analysis result: 7 - Closing: Propose the next step, such as a demo, trial, or meeting with a decision-maker.  
Current stage: Closing: Propose the next step, such as a demo, trial, or meeting with a decision-maker. Make sure to summarize the discussion and restate the benefits.  
Li Sales: Excellent! I’ll schedule a free sleep consultation for you right away. May I ask which day this week would be convenient for you? Our specialist will contact you to learn more about your needs and provide tailored recommendations. Looking forward to helping you improve your sleep quality!

Please enter your reply (type 'quit' to exit): Tomorrow.  
User: Tomorrow.  
Conversation stage analysis result: 7 - Closing: Propose the next step, such as a demo, trial, or meeting with a decision-maker.  
Current stage: Closing: Propose the next step, such as a demo, trial, or meeting with a decision-maker. Make sure to summarize the discussion and restate the benefits.  
Li Sales: Thank you very much! I’ll schedule your sleep consultation for tomorrow. Our specialist will reach out to ensure you get the most suitable sleep solution. If you have any other questions or need further assistance, please don’t hesitate to let me know. Have a great night!

Please enter your reply (type 'quit' to exit): quit  
Conversation ended. Thank you for using!
```

## Summary

This tutorial demonstrated how to build a context-aware AI Sales Assistant using the LazyLLM framework. Through modular design and stage-based management, the system can simulate realistic sales conversations and provide personalized service experiences for potential customers.

This implementation approach is not limited to sales scenarios — it can be extended to other context-aware dialogue systems such as customer service, consulting, and education.
