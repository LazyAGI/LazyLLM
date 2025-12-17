# -*- coding: utf-8 -*-
"""
PPIO (Paiou Cloud) chatbot example.

Before running, set environment variable:
export LAZYLLM_PPIO_API_KEY=your_api_key

Or create config file ~/.lazyllm/config.json:
{
    "ppio_api_key": "your_api_key"
}
"""

import lazyllm

# Method 1: Use default config (read API Key from environment variable or config file)
chat = lazyllm.OnlineChatModule(source='ppio', model='deepseek/deepseek-v3.2')

# Method 2: Specify API Key directly
# chat = lazyllm.OnlineChatModule(
#     source='ppio',
#     model='deepseek/deepseek-v3.2',
#     api_key='your_api_key_here'
# )

if __name__ == '__main__':
    # Start web interface, port range 23466-23470
    lazyllm.WebModule(chat, port=range(23466, 23470)).start().wait()

