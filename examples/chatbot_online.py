import lazyllm
chat = lazyllm.OnlineChatModule()

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=23466).start().wait()
