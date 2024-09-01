import sys
import argparse

# lazyllm run xx.json / xx.dsl / xx.lazyml
# lazyllm run chatbot --model=xx --framework=xx --source=xx
# lazyllm run rag --model=xx --framework=xx --source=xx --documents=''

def run(commands):
    parser = argparse.ArgumentParser(description="lazyllm deploy command")
    parser.add_argument("command", help="model name")

    print('lazyllm run is not ready yet.')
    sys.exit(0)
