# Context-Aware AI Sales Assistant

This tutorial introduces how to implement a context-aware AI sales assistant using the LazyLLM framework, which can automatically adjust its behavior and response strategies based on conversation stages.

## Overview

SalesGPT is a context-aware AI sales assistant that can:

1. **Identify Conversation Stages**: Automatically analyze which sales stage the current conversation is in
2. **Dynamic Strategy Adjustment**: Adjust response content and strategies based on conversation stages
3. **Natural Conversation Flow**: Simulate real sales conversation processes

## Architecture Design

### Core Components

1. **SalesStageAnalyzer (Sales Stage Analyzer)**
   - Analyze conversation history
   - Determine which sales stage should be current

2. **SalesConversationAgent (Sales Conversation Agent)**
   - Generate corresponding responses based on current stage
   - Maintain salesperson identity and company information

3. **SalesGPT (Main Controller)**
   - Coordinate various components
   - Manage conversation history and state

### Sales Conversation Stages

The system defines 7 sales conversation stages:

1. **Introduction Stage**: Start conversation, introduce yourself and company
2. **Qualification**: Confirm if potential customer has purchasing decision authority
3. **Value Proposition**: Explain unique value of products/services
4. **Needs Analysis**: Understand customer needs and pain points
5. **Solution Presentation**: Present products as solutions
6. **Objection Handling**: Handle customer concerns and objections
7. **Closing**: Propose next steps to close the deal

## Code Implementation

### 1. Import Required Libraries

```python
import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.components import ChatPrompter
```

### 2. Implement Sales Stage Analyzer

```python
class SalesStageAnalyzer(ModuleBase):
    """Sales conversation stage analyzer for identifying which sales stage the current conversation should be in"""
    
    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        
        # Sales conversation stage definitions
        self.conversation_stages = {
            "1": "Introduction Stage: Start conversation, introduce yourself and company. Maintain polite and professional tone.",
            "2": "Qualification: Confirm if potential customer is the right person, ensure they have purchasing decision authority.",
            "3": "Value Proposition: Briefly explain how products/services benefit potential customers, highlight unique selling points.",
            "4": "Needs Analysis: Understand potential customer needs and pain points through open-ended questions.",
            "5": "Solution Presentation: Based on potential customer needs, present products/services as solutions.",
            "6": "Objection Handling: Handle any objections potential customers have about products/services, provide supporting evidence.",
            "7": "Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers."
        }
        
        # Stage analysis prompt template
        stage_analyzer_prompt = """You are a sales assistant helping sales agents determine which stage a sales conversation should enter.

Here is the conversation history:
===
{conversation_history}
===

Now determine the next immediate conversation stage for the sales agent in the sales conversation based on the conversation history, choose from the following options:
1. Introduction Stage: Start conversation, introduce yourself and company. Maintain polite and professional tone.
2. Qualification: Confirm if potential customer is the right person, ensure they have purchasing decision authority.
3. Value Proposition: Briefly explain how products/services benefit potential customers, highlight unique selling points.
4. Needs Analysis: Understand potential customer needs and pain points through open-ended questions.
5. Solution Presentation: Based on potential customer needs, present products/services as solutions.
6. Objection Handling: Handle any objections potential customers have about products/services, provide supporting evidence.
7. Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers.

Only answer with a number between 1 and 7, indicating which stage the conversation should continue. The answer must be only a number, do not add any other content.
If there is no conversation history, output 1.
Do not answer anything else."""
        
        self.prompter = ChatPrompter(instruction=stage_analyzer_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)
    
    def forward(self, conversation_history: str) -> str:
        """Analyze conversation history and return current sales stage"""
        response = self.llm({"conversation_history": conversation_history})
        stage_id = ''.join(filter(str.isdigit, response.strip()))
        if not stage_id or int(stage_id) not in range(1, 8):
            stage_id = "1"  # Default to introduction stage
        
        if self.verbose:
            print(f"Conversation stage analysis result: {stage_id} - {self.conversation_stages[stage_id]}")
        
        return stage_id
```

### 3. Implement Sales Conversation Agent

```python
class SalesConversationAgent(ModuleBase):
    """Sales conversation agent that generates corresponding responses based on current stage"""
    
    def __init__(self, llm: ModuleBase, verbose: bool = True):
        super().__init__()
        self.verbose = verbose
        
        # Sales conversation prompt template
        sales_conversation_prompt = """You are {salesperson_name}, a {salesperson_role}.
You work at {company_name}. {company_name}'s business is: {company_business}
Company values are: {company_values}
Your purpose for contacting potential customers is: {conversation_purpose}
Your method of contacting potential customers is: {conversation_type}

If asked where you got the user's contact information, say you obtained it from public records.
Keep responses brief to maintain user attention. Do not generate lists, just answer questions.
You must respond based on previous conversation history and current conversation stage.
Generate only one response at a time! After generation is complete, end with '<END_OF_TURN>' to give the user a chance to respond.

Current conversation stage: {conversation_stage}
Conversation history:
{conversation_history}
{salesperson_name}: """
        
        self.prompter = ChatPrompter(instruction=sales_conversation_prompt)
        self.llm = llm.share(prompt=self.prompter).used_by(self._module_id)
    
    def forward(self, salesperson_name: str, salesperson_role: str, company_name: str, 
                company_business: str, company_values: str, conversation_purpose: str,
                conversation_type: str, conversation_stage: str, conversation_history: str) -> str:
        """Generate sales conversation response"""
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

### 4. Implement Main Controller

```python
class SalesGPT(ModuleBase):
    """Context-aware AI sales assistant main controller"""
    
    def __init__(self, llm: ModuleBase, salesperson_name: str = "Zhang Sales",
                 salesperson_role: str = "Business Development Representative",
                 company_name: str = "Quality Sleep",
                 company_business: str = "Quality Sleep is a premium mattress company...",
                 company_values: str = "Quality Sleep's mission is to help people achieve better sleep through providing optimal sleep solutions...",
                 conversation_purpose: str = "Understand if they want to improve sleep quality by purchasing premium mattresses.",
                 conversation_type: str = "Phone",
                 verbose: bool = True):
        super().__init__()
        
        # Set salesperson information
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
        self.current_conversation_stage = "1"
        
        # Initialize components
        self.stage_analyzer = SalesStageAnalyzer(llm, verbose=verbose)
        self.sales_conversation_agent = SalesConversationAgent(llm, verbose=verbose)
        
        # Conversation stage definitions
        self.conversation_stages = {
            "1": "Introduction Stage: Start conversation, introduce yourself and company. Maintain polite and professional tone.",
            "2": "Qualification: Confirm if potential customer is the right person, ensure they have purchasing decision authority.",
            "3": "Value Proposition: Briefly explain how products/services benefit potential customers, highlight unique selling points.",
            "4": "Needs Analysis: Understand potential customer needs and pain points through open-ended questions.",
            "5": "Solution Presentation: Based on potential customer needs, present products/services as solutions.",
            "6": "Objection Handling: Handle any objections potential customers have about products/services, provide supporting evidence.",
            "7": "Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers."
        }
    
    def seed_agent(self):
        """Initialize sales agent"""
        if self.verbose:
            print(f"{self.salesperson_name}: (Waiting for user input...)")
    
    def determine_conversation_stage(self):
        """Determine current conversation stage"""
        conversation_text = "\n".join(self.conversation_history)
        self.current_conversation_stage = self.stage_analyzer(conversation_text)
        if self.verbose:
            print(f"Current conversation stage: {self.conversation_stages[self.current_conversation_stage]}")
    
    def human_step(self, human_input: str):
        """Process human input"""
        human_input = human_input + "<END_OF_TURN>"
        self.conversation_history.append(human_input)
        if self.verbose:
            print(f"User: {human_input.replace('<END_OF_TURN>', '')}")
    
    def step(self):
        """Execute one step of sales agent conversation"""
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

### 5. Main Function Implementation

```python
def main():
    """Main function: Demonstrate SalesGPT usage"""
    print("=== Context-Aware AI Sales Assistant Demo ===\n")
    
    # Set up LLM
    llm = lazyllm.OnlineChatModule()
    
    # Configure sales agent
    config = {
        "salesperson_name": "Li Sales",
        "salesperson_role": "Business Development Representative",
        "company_name": "Quality Sleep",
        "company_business": "Quality Sleep is a premium mattress company providing the most comfortable and supportive sleep experience for customers.",
        "company_values": "Quality Sleep's mission is to help people achieve better sleep through providing optimal sleep solutions.",
        "conversation_purpose": "Understand if they want to improve sleep quality by purchasing premium mattresses.",
        "conversation_type": "Phone"
    }
    
    # Create sales agent
    sales_agent = SalesGPT(llm, **config)
    
    # Initialize agent
    print("Initializing sales agent...")
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()
    
    # Start conversation loop
    print("\nStarting sales conversation...")
    while True:
        sales_agent.step()
        
        user_input = input("\nPlease enter your response (enter 'quit' to exit): ")
        if user_input.lower() == 'quit':
            print("Conversation ended, thank you for using!")
            break
        
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()

if __name__ == "__main__":
    main()
```

### Actual Testing

```bash
=== Context-Aware AI Sales Assistant Demo ===

Initializing sales agent...
Li Sales: (Waiting for user input...)
Conversation stage analysis result: 1 - Introduction Stage: Start conversation, introduce yourself and company. Maintain polite and professional tone.
Current conversation stage: Introduction Stage: Start conversation, introduce yourself and company. Maintain polite and professional tone. Your greeting should be welcoming. Always clearly state the reason for contacting potential customers in the greeting.

Starting sales conversation...
Li Sales: Hello! I'm Li Sales from Quality Sleep Company. We specialize in providing premium mattresses and sleep solutions to help people improve their sleep quality. Are you currently looking for ways to enhance your sleep experience?

Please enter your response (enter 'quit' to exit): Yes
User: Yes
Conversation stage analysis result: 4 - Needs Analysis: Understand potential customer needs and pain points through open-ended questions.
Current conversation stage: Needs Analysis: Understand potential customer needs and pain points through open-ended questions. Listen carefully to their responses and take notes.
Li Sales: Great! I'm glad to hear you're interested in improving your sleep experience. What specific sleep issues are you currently facing, or what improvements do you hope to achieve in your sleep experience?

Please enter your response (enter 'quit' to exit): I can't sleep well at night, easily suffer from insomnia
User: I can't sleep well at night, easily suffer from insomnia
Conversation stage analysis result: 4 - Needs Analysis: Understand potential customer needs and pain points through open-ended questions.
Current conversation stage: Needs Analysis: Understand potential customer needs and pain points through open-ended questions. Listen carefully to their responses and take notes.
Li Sales: I understand, poor sleep at night and insomnia can really affect quality of life. Have you tried using special material mattresses or pillows to improve this situation?

Please enter your response (enter 'quit' to exit): Not yet
User: Not yet
Conversation stage analysis result: 3 - Value Proposition: Briefly explain how products/services benefit potential customers, highlight unique selling points.
Current conversation stage: Value Proposition: Briefly explain how products/services benefit potential customers, highlight unique selling points.
Li Sales: Then let me introduce how our products can help you. Our mattresses use advanced materials and design to provide customized support based on your body, reducing tossing and turning, helping you fall asleep faster and achieve deep sleep. Additionally, our pillows use memory foam that adapts to your head and neck shape, further improving sleep quality. These products have helped many customers significantly improve their sleep issues. Are you interested in learning more?

Please enter your response (enter 'quit' to exit): Tell me about your products and prices
User: Tell me about your products and prices
Conversation stage analysis result: 5 - Solution Presentation: Based on potential customer needs, present products/services as solutions.
Current conversation stage: Solution Presentation: Based on potential customer needs, present products/services as solutions.
Li Sales: Of course. Our product line includes three main series: Classic Series, Luxury Series, and Custom Series.

1. **Classic Series**: Suitable for customers seeking comfort and support. Prices range from 5,000 to 10,000 yuan.

2. **Luxury Series**: Uses higher-end materials and advanced technology, suitable for customers pursuing ultimate sleep experience. Prices range from 10,000 to 20,000 yuan.

3. **Custom Series**: Tailored to your specific needs and physical condition, providing the most personalized sleep solutions. Prices start from 20,000 yuan.

Our mattresses and pillows all come with warranty periods to ensure you can use them with peace of mind. If you're interested, we can arrange a free sleep consultation to better understand your needs and provide more accurate recommendations. What do you think?

Please enter your response (enter 'quit' to exit): Too expensive
User: Too expensive
Conversation stage analysis result: 6 - Objection Handling: Handle any objections potential customers have about products/services, provide supporting evidence.
Current conversation stage: Objection Handling: Handle any objections potential customers have about products/services, prepare to provide evidence or testimonials to support your claims.
Li Sales: I understand your concern, premium mattresses do cost more than regular mattresses. But considering our products can significantly improve sleep quality and reduce insomnia issues, from a long-term perspective, this is actually a very worthwhile investment. Quality sleep has a tremendous positive impact on health and work efficiency, helping you perform better in daily life and work.

Additionally, we offer free trial periods and warranty services to ensure you have full experience and confidence in our products. If you're willing, we can arrange a free sleep consultation to provide more personalized advice. What do you think?

Please enter your response (enter 'quit' to exit): Sure
User: Sure
Conversation stage analysis result: 7 - Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers.
Current conversation stage: Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers. Ensure to summarize discussion content and reiterate benefits.
Li Sales: Excellent! I'll immediately arrange a free sleep consultation for you. Which day this week would be convenient for you? We can arrange for professionals to contact you to further understand your specific needs and provide personalized advice. Looking forward to helping you improve your sleep quality!

Please enter your response (enter 'quit' to exit): Tomorrow
User: Tomorrow
Conversation stage analysis result: 7 - Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers.
Current conversation stage: Closing: Request sales by proposing next steps such as demo, trial, or meeting with decision makers. Ensure to summarize discussion content and reiterate benefits.
Li Sales: Thank you very much, I'll arrange a sleep consultation for you tomorrow. Our professionals will contact you to ensure you get the most suitable sleep solutions. If you have any other questions or need further assistance, please feel free to let me know. Have a good night's sleep!

Please enter your response (enter 'quit' to exit): quit
Conversation ended, thank you for using!
```

## Core Features

### 1. Context Awareness

The system can automatically identify which sales stage the current conversation should be in based on conversation history and adjust response strategies accordingly.

### 2. Modular Design

Uses LazyLLM's ModuleBase class for modular design, making it easy to extend and maintain.

### 3. Flexible Configuration

Supports customizing salesperson information, company information, conversation purpose, and other parameters.

### 4. Natural Conversation Flow

Simulates real sales conversation processes, including introduction, needs analysis, value proposition, objection handling, closing, and other stages.

## Extension Suggestions

1. **Multi-turn Conversation Optimization**: Can add more complex conversation state management
2. **Personalized Recommendations**: Provide personalized product recommendations based on customer needs
3. **Sentiment Analysis**: Integrate sentiment analysis functionality to better understand customer emotions
4. **Multimodal Support**: Support multimodal inputs such as voice, images, etc.
5. **Data Persistence**: Save conversation history and analysis results

## Summary

This tutorial demonstrates how to build a context-aware AI sales assistant using the LazyLLM framework. Through modular design and stage-based management, the system can simulate real sales conversation processes and provide personalized service experiences for potential customers.

This implementation approach is not only applicable to sales scenarios but can also be extended to other conversation systems that require context awareness, such as customer service, consulting, education, and other fields. 