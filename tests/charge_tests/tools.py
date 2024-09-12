import json
import lazyllm
from typing import Literal
from lazyllm import fc_register
import wikipedia

@fc_register("tool")
def get_current_weather(location: str, unit: Literal["fahrenheit", "celsius"] = 'fahrenheit'):
    """
    Get the current weather in a given location

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        unit (str): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register("tool")
def get_n_day_weather_forecast(location: str, num_days: int, unit: Literal["celsius", "fahrenheit"] = 'fahrenheit'):
    """
    Get an N-day weather forecast

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        num_days (int): The number of days to forecast.
        unit (Literal['celsius', 'fahrenheit']): The temperature unit to use. Infer this from the users location.
    """
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius', "num_days": num_days})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit', "num_days": num_days})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius', "num_days": num_days})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

@fc_register("tool")
def multiply_tool(a: int, b: int) -> int:
    """
    Multiply two integers and return the result integer

    Args:
        a (int): multiplier
        b (int): multiplier
    """
    return a * b

@fc_register("tool")
def add_tool(a: int, b: int):
    """
    Add two integers and returns the result integer

    Args:
        a (int): addend
        b (int): addend
    """
    return a + b

@fc_register("tool")
def WikipediaWorker(input: str):
    """
    Worker that search for similar page contents from Wikipedia. Useful when you need to get holistic knowledge \
    about people, places, companies, historical events, or other subjects. The response are long and might \
    contain some irrelevant information. Input should be a search query.

    Args:
        input (str): search query.
    """
    print(f"wikipedia input: {input}")
    try:
        evidence = wikipedia.page(input).content
        evidence = evidence.split("\n\n")[0]
    except wikipedia.PageError:
        evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    except wikipedia.DisambiguationError:
        evidence = f"Could not find [{input}]. Similar: {wikipedia.search(input)}"
    print(f"wikipedia output: {evidence}")
    return evidence

@fc_register("tool")
def LLMWorker(input: str):
    """
    A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. \
    Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.

    Args:
        input (str): instruction
    """
    llm = lazyllm.OnlineChatModule(source="glm", stream=False)
    query = f"Respond in short directly with no extra words.\n\n{input}"
    print(f"llm query: {query}, input: {input}")
    response = llm(query, llm_chat_history=[])
    print(f"llm res: {response}")
    return response
