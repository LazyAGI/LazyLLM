import sys
try:
    from install import install
    from deploy import deploy
    from run import run
    from skills import skills
    from review import review, review_local
except ImportError:
    from .install import install
    from .deploy import deploy
    from .run import run
    from .skills import skills
    from .review import review, review_local
import logging

def main():
    def exit():
        logging.error('Usage:\n  lazyllm install <extra1> <extra2> <pkg1> ...\n'
                      '  lazyllm deploy modelname\n  lazyllm deploy mcp_server <command> [args ...] [options]\n'
                      '  lazyllm run graph.json\n  lazyllm run chatbot\n  lazyllm run rag\n'
                      '  lazyllm skills init\n  lazyllm skills list\n  lazyllm skills info <name>\n'
                      '  lazyllm skills delete <name>\n  lazyllm skills add <path> [-n NAME] [--dir DIR]\n'
                      '  lazyllm skills import <path> [--dir DIR] [--names a,b,c] [--overwrite]\n'
                      '  lazyllm skills install --agent <name> [--project] [--timeout SEC]\n'
                      '  lazyllm review --pr <number> [--repo owner/name] [--model ...] [--post] ...\n'
                      '  lazyllm review-local [--repo-path .] [--base main] [--output review.json] ...\n')
        sys.exit(1)

    if len(sys.argv) <= 1: exit()

    commands = sys.argv[2:]
    if sys.argv[1] == 'install':
        install(commands)
    elif sys.argv[1] == 'deploy':
        deploy(commands)
    elif sys.argv[1] == 'run':
        run(commands)
    elif sys.argv[1] == 'skills':
        skills(commands)
    elif sys.argv[1] == 'review':
        review(commands)
    elif sys.argv[1] == 'review-local':
        review_local(commands)
    else:
        exit()

if __name__ == '__main__':
    main()
