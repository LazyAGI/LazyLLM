import lazyllm
import argparse
import time

def deploy(commands):
    parser = argparse.ArgumentParser(description="lazyllm deploy command")
    parser.add_argument("model", help="model name")
    parser.add_argument("--framework", help="deploy framework", default="auto",
                        choices=["auto", "vllm", "lightllm", "lmdeploy"])

    args = parser.parse_args(commands)

    t = lazyllm.TrainableModule(args.model).deploy_method(getattr(lazyllm.deploy, args.framework))
    t.start()
    lazyllm.LOG.success(f'LazyLLM TrainableModule launched successfully:\n  URL: {t._url}\n  '
                        f'Framework: {t._deploy_type.__name__}', flush=True)
    while True:
        time.sleep(10)
