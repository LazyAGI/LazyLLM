import sys
try:
    from install import install
    from deploy import deploy
    from run import run
except ImportError:
    from .install import install
    from .deploy import deploy
    from .run import run

def main():
    def exit():
        print('Usage:\n  lazyllm install [full|standard|package_name]\n'
              '  lazyllm deploy modelname\n  lazyllm deploy mcp_server <command> [args ...] [options]\n'
              '  lazyllm run graph.json\n  lazyllm run chatbot\n  lazyllm run rag\n')
        sys.exit(1)

    if len(sys.argv) <= 1: exit()

    commands = sys.argv[2:]
    if sys.argv[1] == 'install':
        install(commands)
    elif sys.argv[1] == 'deploy':
        deploy(commands)
    elif sys.argv[1] == 'run':
        run(commands)
    else:
        exit()

if __name__ == "__main__":
    main()
