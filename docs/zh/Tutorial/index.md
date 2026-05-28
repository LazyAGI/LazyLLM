# 教程概览

欢迎来到 LazyLLM 教程！

本教程旨在帮助你快速上手和深入了解 LazyLLM 的各项功能。

## 教程目录

### 从数据到大模型

- [第1讲 Transformer 核心与 Self-Attention 深度剖析](from-data-to-llm/chapter1/1.md)
  从注意力机制、位置编码到 Transformer 结构，理解大模型架构的基础。

- [第2讲 LLM 训练范式与数据工程全景](from-data-to-llm/chapter2/2.md)
  建立从数据准备、训练流程到评测反馈的大模型训练全局视角。

- [第3讲 分布式训练技术概览](from-data-to-llm/chapter3/3.md)
  讲解数据并行、模型并行、流水线并行等大规模训练核心技术。

- [第4讲 模型部署与推理加速](from-data-to-llm/chapter4/4.md)
  介绍推理部署、显存优化、服务化与加速策略。

- [第5讲 基于 Agent 的数据处理](from-data-to-llm/chapter5/5.md)
  使用 Agent 自动化完成数据清洗、分析、生成与流程编排。

- [第6讲 基于 LazyLLM 的全流程实践](from-data-to-llm/chapter6/6.md)
  以 LazyLLM 串联数据构建、模型训练与效果评测。

- [第7讲 预训练原理、策略与评测](from-data-to-llm/chapter7/7.md)
  理解预训练目标、训练策略和衡量模型能力的关键指标。

- [第8讲 预训练数据构建全流程](from-data-to-llm/chapter8/8.md)
  覆盖语料采集、清洗、去重、过滤、配比与数据质量控制。

- [第9讲 基于 LazyLLM 的预训练实战](from-data-to-llm/chapter9/9.md)
  通过实践完成预训练数据到模型训练的闭环。

- [第10讲 多模态 LLM 架构与预训练实战](from-data-to-llm/chapter10/10.md)
  解析视觉编码、图文对齐与多模态预训练方法。

- [第11讲 指令微调原理与策略](from-data-to-llm/chapter11/11.md)
  讲解 SFT、LoRA 等指令微调方法及其数据要求。

- [第12讲 通用指令数据构建、合成与蒸馏](from-data-to-llm/chapter12/12.md)
  构建高质量指令数据，并利用合成与蒸馏提升覆盖度。

- [第13讲 基于 LazyLLM 的微调实战](from-data-to-llm/chapter13/13.md)
  使用 LazyLLM 完成微调任务配置、训练和效果验证。

- [第14讲 多模态指令微调与实战](from-data-to-llm/chapter14/14.md)
  面向图文任务构建多模态指令数据并完成微调实践。

- [第15讲 对齐算法原理（RLHF & GRPO）](from-data-to-llm/chapter15/15.md)
  理解偏好优化、强化学习对齐与 GRPO 等主流对齐算法。

- [第16讲 偏好数据构建](from-data-to-llm/chapter16/16.md)
  讲解偏好样本设计、标注、质量控制和对齐训练数据组织。

- [第17讲 模型风险、合规与伦理](from-data-to-llm/chapter17/17.md)
  识别大模型安全风险，理解合规、隐私和伦理治理要求。

- [第18讲 基于 LazyLLM 的对齐实战](from-data-to-llm/chapter18/18.md)
  结合 LazyLLM 完成对齐数据、训练流程和结果评估实践。

- [第19讲 推理与数学能力增强](from-data-to-llm/chapter19/19.md)
  围绕推理链、数学数据和训练策略提升模型复杂问题求解能力。

- [第20讲 代码能力增强](from-data-to-llm/chapter20/20.md)
  构建代码语料和训练任务，增强模型代码理解与生成能力。

- [第21讲 长上下文能力增强](from-data-to-llm/chapter21/21.md)
  介绍长上下文数据构建、位置扩展与长序列评测方法。

- [第22讲 结构化输出与格式对齐](from-data-to-llm/chapter22/22.md)
  让模型稳定输出 JSON、表格、工具参数等可解析结构。

- [第23讲 Agent 能力增强（Tools & Planning）](from-data-to-llm/chapter23/23.md)
  通过工具调用、任务规划和执行反馈增强 Agent 能力。

- [第24讲 行业领域模型实战](from-data-to-llm/chapter24/24.md)
  面向垂直行业构建领域数据、训练方案和落地评测流程。

- [第25讲 RAG 架构原理与数据处理](from-data-to-llm/chapter25/25.md)
  讲解 RAG 系统结构、知识处理、切分、索引与检索数据准备。

- [第26讲 Embedding 模型微调与实战](from-data-to-llm/chapter26/26.md)
  构建检索训练样本并微调 Embedding 模型，提升召回质量。

- [第27讲 Reranker 模型微调与实战](from-data-to-llm/chapter27/27.md)
  通过重排模型微调提升候选文档排序与最终问答效果。

- [第28讲 Agentic RAG 能力增强](from-data-to-llm/chapter28/28.md)
  融合 Agent 与 RAG，实现多跳检索、规划执行和复杂任务增强。

### LazyLLM RAG 教程

- [第1讲 RAG原理解读：让检索增强生成不再是黑盒](1.md)
  透彻讲解 RAG（Retrieval-Augmented Generation）的原理与结构。

- [第2讲 10分钟上手一个最小可用RAG系统](2.md)
  零基础快速构建最小可运行的 RAG 系统。

- [第3讲 大模型怎么玩：用LazyLLM带你理解调用逻辑与Prompt魔法](3.md)
  深入理解大模型的调用过程与 Prompt 的组织方式。

- [第4讲 RAG项目工程化入门：从脚本走向模块化与可维护性](4.md)
  教你如何将 RAG 项目逐步工程化，提升可维护性。

- [第5讲 打造专属Reader组件：轻松解析HTML、PDF等复杂文档格式](5.md)
  自定义文档解析器，实现对多种文档格式的支持。

- [第6讲 检索更准：RAG召回效果优化的底层逻辑与技巧](6.md)
  学习影响召回效果的核心要素与优化策略。

- [第7讲 检索升级实践：亲手打造“更聪明”的文档理解系统！](7.md)
  实战提升系统对复杂文档的理解能力。

- [第8讲 不止是cosine！匹配策略决定你召回的质量](8.md)
  多种向量匹配策略的对比与应用分析。

- [第9讲 微调实践：让大模型和向量模型更懂你的领域](9.md)
  基于私有领域数据对 Embedding 与 LLM 进行微调。

- [第10讲 探索Deepseek：打造思维能力更强的RAG系统](10.md)
  集成 DeepSeek 等强模型提升多步推理能力。

- [第11讲 性能优化指南：从冷启动到响应加速你的RAG](11.md)
  全方位介绍 RAG 系统的启动与响应性能优化技巧。

- [第12讲 实践：用缓存、异步与向量引擎加速你的RAG](12.md)
  手把手教你使用缓存、异步处理、引擎插件等方式提速。

- [第13讲 RAG+多模态：图片、表格通吃的问答系统](13.md)
  构建支持图文混合理解的多模态 RAG 系统。

- [第14讲 实战：构建一个支持复杂学术论文问答的RAG系统](14.md)
  面向论文场景，搭建具备上下文、结构感知能力的系统。

- [第15讲 大视角问答：RAG如何支持跨文档、跨维度总结](15.md)
  实现多文档总结与维度整合型问答。

- [第16讲 实践：打造具备宏观问答与图表生成功能的论文问答的RAG系统](16.md)
  加入宏观总结与图表生成，提升问答系统交互质量。

- [第17讲 企业级RAG：权限、共享与内容安全的全链路方案](17.md)
  企业内部部署场景下的权限管理与安全设计实战。

- [第18讲 高阶RAG：Agentic RAG](18.md)
  探索 Agent + RAG 的融合方式，增强任务执行能力。

- [第19讲： RAG × 知识图谱：从关系结构中召回更准确的内容](19.md)
  打通结构化知识图谱与生成式问答系统的协同路径。

---

准备好开始探索了吗？点击任意一讲，即可深入学习 🚀
