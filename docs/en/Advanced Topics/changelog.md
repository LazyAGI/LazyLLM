# [0.1.3] - 2024-07-09
## New Features
- **update docs in mkdocs** - [commit](https://github.com/LazyAGI/LazyLLM/commit/75dcd3b373c18ef90d1e6e1f5dde4286b749a99e)
- **convert the onlineChatModule stream output to nonstream** (#61) - [commit](https://github.com/LazyAGI/LazyLLM/commit/1de9ee78d24c09eb85571ba979451d060bcc743b)

## Bug Fixes / Nits
- **Fixed a bug that required explicit settings when using auto-deploy** (#64) - [commit](https://github.com/LazyAGI/LazyLLM/commit/b0d8bac9ec271607307243bea68d9a23a7569d04)
- **add log_once for logger system** (#60) - [commit](https://github.com/LazyAGI/LazyLLM/commit/4bc8cfc0cf819af792493bb4df3366a08d05ea5c)
- **Fixed a bug that caused the test to get stuck due to resources on sco not being released.** (#63) - [commit](https://github.com/LazyAGI/LazyLLM/commit/9c5b339c58932f2fe8387ce88c1f6807d2f4729f)
- **fix: web module unexpected type error** (#58) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ada3e966ff8a40b75caddda36393b16bd3e99307)
- **Support sft via `LLaMA-Factory`** (#51) - [commit](https://github.com/LazyAGI/LazyLLM/commit/c0ba8b9ba3321f9c779730431f051dfd6a1f3379)
- **change default language of README.md to English from Chinese** (#57) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5c21e3d22817521072c5edb636b46d89fd9b2b22)
- **add with for switch and loop** (#55) - [commit](https://github.com/LazyAGI/LazyLLM/commit/8971e9fdb654e2ebbfe10edd609c4c48cba5c4a6)
- **add DocNode and sentence-splitter for RAG** (#48) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ae6c0ca9b89900dac3149bb2fbe6aa2189f4b243)
- **Dev env update** (#52) - [commit](https://github.com/LazyAGI/LazyLLM/commit/1a7556ecf0d816a6289d3c2e518bb091c1514e25)
- **support TrainableModule.share()** (#40) - [commit](https://github.com/LazyAGI/LazyLLM/commit/16c44adc83d48b54b0619abadd58a2aca586daab)
- **fix: rename document** (#50) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d2d63c938edc17cf794936e8b1bad91bb7a2d61d)
- **Update README.md to add rag demo** (#46) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a7089604be2a2a66d9ed136f593ff8ce22f657cc)

# [0.1.3] - 2024-06-22
## New Features
- **Updated the logic of uploading to pipy** (#42) - [commit](https://github.com/LazyAGI/LazyLLM/commit/53b060be2579abb2f858f35e16151f3798df1e16)
- **support keep record for sensecore** (#41) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a595abd8cf84e6e1c3fec19230a49e438f162590)
- **Fixed CI stability** (#39) - [commit](https://github.com/LazyAGI/LazyLLM/commit/75a502c47de17781fedb6e7bd61bb30a39096153)
- **version to v0.1.2** (#38) - [commit](https://github.com/LazyAGI/LazyLLM/commit/946b9ddcc5a89c6115a7c9c2179f6b3e33100370)
- **modify version to v0.1.0** (#36) - [commit](https://github.com/LazyAGI/LazyLLM/commit/10b13f280111214abd4d1c52ee99edb891ce0b8f)
- **upload PyPI and Release** (#35) - [commit](https://github.com/LazyAGI/LazyLLM/commit/46bd2d046af410dd5e1525e89931bb5d143d4bb2)
- **Organize Pytest Examples And Add them to CI** (#29) - [commit](https://github.com/LazyAGI/LazyLLM/commit/3998f72825d3022d67c3f543881259da4d768fbd)
- **add python package manager for lazyllm** (#20) - [commit](https://github.com/LazyAGI/LazyLLM/commit/65d3fd3db5edf26203803182320c556611ecbd94)
- **Fixed the bug caused by the given path not existing in the logging system.** (#25) - [commit](https://github.com/LazyAGI/LazyLLM/commit/727085cb1e540b7caf8087b7f90e9ec6f27ff956)
- **fix bug for online chat formatter** (#24) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d7ac613e49618c58c3e50bc7c6cda69b9bbfa864)
- **Online chat formatter** (#8) - [commit](https://github.com/LazyAGI/LazyLLM/commit/42162021b08bbfb4c2f5c50783a3973e768988ba)
- **add `aslist`, `astuple` and `join` for parallel and warp** (#23) - [commit](https://github.com/LazyAGI/LazyLLM/commit/47af82727a0462e5bc6aaa9d60a898d8a4c850cc)
- **Add auto path for data and model** (#21) - [commit](https://github.com/LazyAGI/LazyLLM/commit/2c0b94e217b938a6c7e46e74def3356b5493007d)
- **exports all examples** (#22) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ee149a2712f1e132d2bbfb32adc2b31a1f570e40)
- **add badges** (#19) - [commit](https://github.com/LazyAGI/LazyLLM/commit/8115afd58dd485e34f3d1dec7e33fe103fb28d23)

## Bug Fixes / Nits
- **update readme** (#17) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5a37b62264b56dc1f11b34b6da6913f28453a3fc)
- **Support simple prompt keys, RAG demo and readme** (#14) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d68bfeb2e009e955b574304fda87d3ee9e2ce358)
- **add best-practice docs for rag** (#16) - [commit](https://github.com/LazyAGI/LazyLLM/commit/52e91e0e93ef7b6b96aebc7797db25513a7e6f16)
- **Deploy doc** (#9) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a4de27f4129e576787b5b6d0f7b69dc6c152f8e2)
- **rtd doc** (#15) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5c3f55b2816a4c914a19ad08f134d9a06350d0e2)
- **add best-practice docs for prompter and module** (#13) - [commit](https://github.com/LazyAGI/LazyLLM/commit/7005de51addea94a164f94d71c330c1f0e9fec57)
- **trainable module init with no arg and examples corrections** (#11) - [commit](https://github.com/LazyAGI/LazyLLM/commit/31b495464ae3f816e1c3bc21880b2905e8c9ba4c)
- **add parallel mode for all prompts** (#10) - [commit](https://github.com/LazyAGI/LazyLLM/commit/6c5d3a578f015e5e99f3f03d5b70d7c9242cb9b5)
- **Nits and bugs** (#7) - [commit](https://github.com/LazyAGI/LazyLLM/commit/eb495d7c5949d55b0d71c7d17081d57c2a19634e)
- **fix readme** (#6) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a7815b08d7edb0a91d2b08ef58570c5275c7989f)