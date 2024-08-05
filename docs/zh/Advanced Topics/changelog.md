# [0.1.3] - 2024-07-09
## 新特性
- **更新 mkdocs 中的文档** - [commit](https://github.com/LazyAGI/LazyLLM/commit/75dcd3b373c18ef90d1e6e1f5dde4286b749a99e)
- **将 onlineChatModule 流输出转换为非流输出** (#61) - [commit](https://github.com/LazyAGI/LazyLLM/commit/1de9ee78d24c09eb85571ba979451d060bcc743b)

## Bug Fixes / Nits
- **修复了在使用自动部署时需要显式设置的 bug** (#64) - [commit](https://github.com/LazyAGI/LazyLLM/commit/b0d8bac9ec271607307243bea68d9a23a7569d04)
- **为日志系统添加 log_once** (#60) - [commit](https://github.com/LazyAGI/LazyLLM/commit/4bc8cfc0cf819af792493bb4df3366a08d05ea5c)
- **修复了由于 sco 上的资源未被释放导致测试卡住的 bug** (#63) - [commit](https://github.com/LazyAGI/LazyLLM/commit/9c5b339c58932f2fe8387ce88c1f6807d2f4729f)
- **修复：web 模块出现意外类型错误** (#58) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ada3e966ff8a40b75caddda36393b16bd3e99307)
- **通过 `LLaMA-Factory` 支持 sft** (#51) - [commit](https://github.com/LazyAGI/LazyLLM/commit/c0ba8b9ba3321f9c779730431f051dfd6a1f3379)
- **将 README.md 的默认语言从中文改为英文** (#57) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5c21e3d22817521072c5edb636b46d89fd9b2b22)
- **为 switch 和 loop 添加 with** (#55) - [commit](https://github.com/LazyAGI/LazyLLM/commit/8971e9fdb654e2ebbfe10edd609c4c48cba5c4a6)
- **为 RAG 添加 DocNode 和 sentence-splitter** (#48) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ae6c0ca9b89900dac3149bb2fbe6aa2189f4b243)
- **开发环境更新** (#52) - [commit](https://github.com/LazyAGI/LazyLLM/commit/1a7556ecf0d816a6289d3c2e518bb091c1514e25)
- **支持 TrainableModule.share()** (#40) - [commit](https://github.com/LazyAGI/LazyLLM/commit/16c44adc83d48b54b0619abadd58a2aca586daab)
- **修复：重命名文档** (#50) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d2d63c938edc17cf794936e8b1bad91bb7a2d61d)
- **更新 README.md，添加 rag 演示** (#46) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a7089604be2a2a66d9ed136f593ff8ce22f657cc)

# [0.1.3] - 2024-06-22
## New Features
- **更新了上传到 pipy 的逻辑** (#42) - [commit](https://github.com/LazyAGI/LazyLLM/commit/53b060be2579abb2f858f35e16151f3798df1e16)
- **支持为 sensecore 保留记录** (#41) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a595abd8cf84e6e1c3fec19230a49e438f162590)
- **修复了 CI 稳定性** (#39) - [commit](https://github.com/LazyAGI/LazyLLM/commit/75a502c47de17781fedb6e7bd61bb30a39096153)
- **版本更新至 v0.1.2** (#38) - [commit](https://github.com/LazyAGI/LazyLLM/commit/946b9ddcc5a89c6115a7c9c2179f6b3e33100370)
- **修改版本至 v0.1.0** (#36) - [commit](https://github.com/LazyAGI/LazyLLM/commit/10b13f280111214abd4d1c52ee99edb891ce0b8f)
- **上传 PyPI 和发布** (#35) - [commit](https://github.com/LazyAGI/LazyLLM/commit/46bd2d046af410dd5e1525e89931bb5d143d4bb2)
- **组织 Pytest 示例并将它们添加到 CI** (#29) - [commit](https://github.com/LazyAGI/LazyLLM/commit/3998f72825d3022d67c3f543881259da4d768fbd)
- **为 lazyllm 添加 python 包管理器** (#20) - [commit](https://github.com/LazyAGI/LazyLLM/commit/65d3fd3db5edf26203803182320c556611ecbd94)
- **修复了由于给定路径在日志系统中不存在而导致的 bug** (#25) - [commit](https://github.com/LazyAGI/LazyLLM/commit/727085cb1e540b7caf8087b7f90e9ec6f27ff956)
- **修复在线聊天格式器的 bug** (#24) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d7ac613e49618c58c3e50bc7c6cda69b9bbfa864)
- **在线聊天格式器** (#8) - [commit](https://github.com/LazyAGI/LazyLLM/commit/42162021b08bbfb4c2f5c50783a3973e768988ba)
- **为 parallel 和 warp 添加 `aslist`，`astuple` 和 `join`** (#23) - [commit](https://github.com/LazyAGI/LazyLLM/commit/47af82727a0462e5bc6aaa9d60a898d8a4c850cc)
- **为数据和模型添加自动路径** (#21) - [commit](https://github.com/LazyAGI/LazyLLM/commit/2c0b94e217b938a6c7e46e74def3356b5493007d)
- **导出所有示例** (#22) - [commit](https://github.com/LazyAGI/LazyLLM/commit/ee149a2712f1e132d2bbfb32adc2b31a1f570e40)
- **添加徽章** (#19) - [commit](https://github.com/LazyAGI/LazyLLM/commit/8115afd58dd485e34f3d1dec7e33fe103fb28d23)

## Bug Fixes / Nits
- **更新 readme** (#17) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5a37b62264b56dc1f11b34b6da6913f28453a3fc)
- **支持简单的提示键，RAG 演示和 readme** (#14) - [commit](https://github.com/LazyAGI/LazyLLM/commit/d68bfeb2e009e955b574304fda87d3ee9e2ce358)
- **为 rag 添加最佳实践文档** (#16) - [commit](https://github.com/LazyAGI/LazyLLM/commit/52e91e0e93ef7b6b96aebc7797db25513a7e6f16)
- **部署文档** (#9) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a4de27f4129e576787b5b6d0f7b69dc6c152f8e2)
- **rtd 文档** (#15) - [commit](https://github.com/LazyAGI/LazyLLM/commit/5c3f55b2816a4c914a19ad08f134d9a06350d0e2)
- **为 prompter 和 module 添加最佳实践文档** (#13) - [commit](https://github.com/LazyAGI/LazyLLM/commit/7005de51addea94a164f94d71c330c1f0e9fec57)
- **无参数初始化可训练模块和示例修正** (#11) - [commit](https://github.com/LazyAGI/LazyLLM/commit/31b495464ae3f816e1c3bc21880b2905e8c9ba4c)
- **为所有提示添加并行模式** (#10) - [commit](https://github.com/LazyAGI/LazyLLM/commit/6c5d3a578f015e5e99f3f03d5b70d7c9242cb9b5)
- **小改动和 bug** (#7) - [commit](https://github.com/LazyAGI/LazyLLM/commit/eb495d7c5949d55b0d71c7d17081d57c2a19634e)
- **修复 readme** (#6) - [commit](https://github.com/LazyAGI/LazyLLM/commit/a7815b08d7edb0a91d2b08ef58570c5275c7989f)