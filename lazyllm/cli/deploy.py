import argparse
import time

def deploy(commands):
    import lazyllm
    parser = argparse.ArgumentParser(description="lazyllm deploy command")
    parser.add_argument("model", help="model name")
    parser.add_argument("--framework", help="deploy framework", default="auto",
                        choices=["auto", "vllm", "lightllm", "lmdeploy"])
    parser.add_argument("--chat", help="chat ", default='false',
                        choices=["ON", "on", "1", "true", "True", "OFF", "off", "0", "False", "false"])

    args = parser.parse_args(commands)

    t = lazyllm.TrainableModule(args.model).deploy_method(getattr(lazyllm.deploy, args.framework))
    if args.chat in ["ON", "on", "1", "true", "True"]:
        t = lazyllm.WebModule(t)
    t.start()
    if args.chat in ["ON", "on", "1", "true", "True"]:
        t.wait()
    else:
        lazyllm.LOG.success(f'LazyLLM TrainableModule launched successfully:\n  URL: {t._url}\n  '
                            f'Framework: {t._deploy_type.__name__}', flush=True)
        while True:
            time.sleep(10)
