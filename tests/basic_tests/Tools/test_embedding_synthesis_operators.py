"""Test cases for Embedding Synthesis operators"""
import os
import shutil
import json
import tempfile
import pytest
import lazyllm
from lazyllm import config
from lazyllm.tools.data import embedding


class TestEmbeddingSynthesisOperators:
    """Test suite for Embedding Synthesis data operators"""

    def setup_method(self):
        self.root_dir = './test_data_op'
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def model_set(self):
        """Get LLM instance for testing"""
        self.llm = lazyllm.TrainableModule('Qwen2.5-0.5B-Instruct')
        # self.llm = lazyllm.OnlineChatModule()
        return self.llm

    # ==================== EmbeddingQueryGenerator Tests ====================

    def test_query_generator_init(self):
        """Test EmbeddingQueryGenerator initialization"""
        generator = embedding.EmbeddingQueryGenerator(
            num_queries=5,
            lang="zh",
            _save_data=False
        )
        assert generator is not None
        assert generator.num_queries == 5
        assert generator.lang == "zh"

    def test_query_generator_init_english(self):
        """Test EmbeddingQueryGenerator initialization with English"""
        generator = embedding.EmbeddingQueryGenerator(
            num_queries=3,
            lang="en",
            _save_data=False
        )
        assert generator is not None
        assert generator.lang == "en"

    def test_query_generator_with_llm(self):
        """Test EmbeddingQueryGenerator with actual LLM"""
        generator = embedding.EmbeddingQueryGenerator(
            llm=self.model_set().share(),
            num_queries=2,
            lang="zh",
            _concurrency_mode='single',
            _save_data=False
        )

        inputs = [{
            'passage': '''
            龙美术馆是中国知名的私立美术馆，由收藏家刘益谦、王薇夫妇创办。
            龙美术馆目前在上海有两个馆：浦东馆和西岸馆。
            '''
        }]

        res = generator(inputs)

        # 断言：由于 LLM 输出有不确定性，只检查结构和类型
        assert isinstance(res, list)
        for item in res:
            assert 'query' in item
            assert 'pos' in item
            assert isinstance(item['pos'], list)

    # ==================== EmbeddingHardNegativeMiner Tests ====================

    def test_hard_negative_miner_init_random(self):
        """Test EmbeddingHardNegativeMiner initialization with random strategy"""
        miner = embedding.EmbeddingHardNegativeMiner(
            mining_strategy="random",
            num_negatives=5,
            _save_data=False
        )
        assert miner is not None
        assert miner.mining_strategy == "random"
        assert miner.num_negatives == 5

    def test_hard_negative_miner_init_bm25(self):
        """Test EmbeddingHardNegativeMiner initialization with BM25 strategy"""
        miner = embedding.EmbeddingHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=7,
            language="zh",
            _save_data=False
        )
        assert miner is not None
        assert miner.mining_strategy == "bm25"
        assert miner.language == "zh"

    def test_hard_negative_miner_random(self):
        """Test hard negative mining with random strategy"""
        miner = embedding.EmbeddingHardNegativeMiner(
            mining_strategy="random",
            num_negatives=2,
            _concurrency_mode='single',
            _save_data=False
        )

        # 将 corpus 信息通过 passage 字段嵌入到数据中
        # 算子会自动从 passage 字段提取 corpus
        inputs = [
            {"query": "龙美术馆在哪里？", "pos": ["龙美术馆位于上海"], "passage": "龙美术馆位于上海"},
            {"query": "谁创办了龙美术馆？", "pos": ["刘益谦、王薇夫妇创办"], "passage": "刘益谦、王薇夫妇创办"},
            {"query": "故宫在哪里？", "pos": ["故宫位于北京"], "passage": "故宫位于北京"},
            {"query": "国家博物馆在哪里？", "pos": ["中国国家博物馆在天安门广场"], "passage": "中国国家博物馆在天安门广场"},
            {"query": "上海博物馆怎么样？", "pos": ["上海博物馆收藏丰富"], "passage": "上海博物馆收藏丰富"},
        ]

        res = miner(inputs)

        assert isinstance(res, list)
        assert len(res) == 5
        for item in res:
            assert 'neg' in item
            assert isinstance(item['neg'], list)

    def test_hard_negative_miner_bm25(self):
        """Test hard negative mining with BM25 strategy"""
        miner = embedding.EmbeddingHardNegativeMiner(
            mining_strategy="bm25",
            num_negatives=2,
            language="zh",
            _concurrency_mode='single',
            _save_data=False
        )

        # 将 corpus 信息通过 passage 字段嵌入到数据中
        inputs = [
            {"query": "龙美术馆在哪里？", "pos": ["龙美术馆位于上海浦东"], "passage": "龙美术馆位于上海浦东"},
            {"query": "龙美术馆西岸馆在哪？", "pos": ["龙美术馆西岸馆在徐汇区"], "passage": "龙美术馆西岸馆在徐汇区"},
            {"query": "故宫在哪？", "pos": ["故宫博物院在北京"], "passage": "故宫博物院在北京"},
            {"query": "国家博物馆有什么？", "pos": ["中国国家博物馆收藏众多文物"], "passage": "中国国家博物馆收藏众多文物"},
            {"query": "上海博物馆怎么样？", "pos": ["上海博物馆是著名博物馆"], "passage": "上海博物馆是著名博物馆"},
        ]

        res = miner(inputs)

        assert isinstance(res, list)
        assert len(res) == 5
        for item in res:
            assert 'neg' in item
            # BM25 应该返回词汇相似但不是正样本的结果
            assert len(item['neg']) <= 2

    # ==================== EmbeddingDataAugmentor Tests ====================

    def test_data_augmentor_init(self):
        """Test EmbeddingDataAugmentor initialization"""
        augmentor = embedding.EmbeddingDataAugmentor(
            augment_methods=["query_rewrite"],
            num_augments=2,
            lang="zh",
            _save_data=False
        )
        assert augmentor is not None
        assert augmentor.num_augments == 2
        assert augmentor.lang == "zh"

    def test_data_augmentor_with_llm(self):
        """Test EmbeddingDataAugmentor with actual LLM"""
        augmentor = embedding.EmbeddingDataAugmentor(
            llm=self.model_set().share(),
            augment_methods=["query_rewrite"],
            num_augments=1,
            lang="zh",
            _concurrency_mode='single',
            _save_data=False
        )

        inputs = [
            {"query": "龙美术馆在哪里？", "pos": ["龙美术馆位于上海"], "neg": ["故宫在北京"]},
        ]

        res = augmentor(inputs)

        # 断言：检查结构
        assert isinstance(res, list)

    # ==================== EmbeddingDataFormatter Tests ====================

    def test_data_formatter_init_flagembedding(self):
        """Test EmbeddingDataFormatter initialization with flagembedding format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="flagembedding",
            instruction="Represent this sentence: ",
            _save_data=False
        )
        assert formatter is not None
        assert formatter.output_format == "flagembedding"
        assert formatter.instruction == "Represent this sentence: "

    def test_data_formatter_init_sentence_transformers(self):
        """Test EmbeddingDataFormatter initialization with sentence_transformers format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="sentence_transformers",
            _save_data=False
        )
        assert formatter is not None
        assert formatter.output_format == "sentence_transformers"

    def test_data_formatter_init_triplet(self):
        """Test EmbeddingDataFormatter initialization with triplet format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="triplet",
            _save_data=False
        )
        assert formatter is not None
        assert formatter.output_format == "triplet"

    def test_data_formatter_flagembedding(self):
        """Test formatting to flagembedding format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="flagembedding",
            instruction="Query: ",
            _concurrency_mode='single',
            _save_data=False
        )
        inputs = [
            {"query": "什么是AI？", "pos": ["人工智能是..."], "neg": ["天气是..."]},
        ]
        res = formatter(inputs)

        assert len(res) == 1
        assert res[0]["query"] == "什么是AI？"
        assert res[0]["pos"] == ["人工智能是..."]
        assert res[0]["neg"] == ["天气是..."]
        assert res[0]["prompt"] == "Query: "

    def test_data_formatter_sentence_transformers(self):
        """Test formatting to sentence_transformers format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="sentence_transformers",
            _concurrency_mode='single',
            _save_data=False
        )
        inputs = [
            {"query": "Q1", "pos": ["P1"], "neg": ["N1", "N2"]},
        ]
        res = formatter(inputs)

        # Should expand to anchor-positive-negative triplets
        assert len(res) == 2  # 1 pos * 2 neg
        assert all("anchor" in r and "positive" in r and "negative" in r for r in res)

    def test_data_formatter_triplet(self):
        """Test formatting to triplet format"""
        formatter = embedding.EmbeddingDataFormatter(
            output_format="triplet",
            _concurrency_mode='single',
            _save_data=False
        )
        inputs = [
            {"query": "Q1", "pos": ["P1", "P2"], "neg": ["N1"]},
        ]
        res = formatter(inputs)

        # Should expand: 2 pos * 1 neg = 2 triplets
        assert len(res) == 2
        assert all("query" in r and "positive" in r and "negative" in r for r in res)

    def test_data_formatter_with_output_file(self):
        """Test formatting with output file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.jsonl")
            formatter = embedding.EmbeddingDataFormatter(
                output_format="flagembedding",
                output_file=output_file,
                _concurrency_mode='single',
                _save_data=False
            )
            inputs = [
                {"query": "Q1", "pos": ["P1"], "neg": ["N1"]},
            ]
            res = formatter(inputs)

            # Check file was created
            assert os.path.exists(output_file)

            # Check file contents
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            assert len(lines) == 1

    # ==================== EmbeddingTrainTestSplitter Tests ====================

    def test_train_test_splitter_init(self):
        """Test EmbeddingTrainTestSplitter initialization"""
        splitter = embedding.EmbeddingTrainTestSplitter(
            test_size=0.2,
            seed=42,
            _save_data=False
        )
        assert splitter is not None
        assert splitter.test_size == 0.2
        assert splitter.seed == 42

    def test_train_test_splitter_split(self):
        """Test train/test splitting"""
        splitter = embedding.EmbeddingTrainTestSplitter(
            test_size=0.3,
            seed=42,
            _concurrency_mode='single',
            _save_data=False
        )
        inputs = [{"id": i, "query": f"query_{i}"} for i in range(10)]
        res = splitter(inputs)

        # All samples should have split label
        assert len(res) == 10
        assert all("split" in r for r in res)

        # Check split ratio
        train_count = sum(1 for r in res if r["split"] == "train")
        test_count = sum(1 for r in res if r["split"] == "test")
        assert train_count == 7  # 70%
        assert test_count == 3   # 30%

    def test_train_test_splitter_with_output_files(self):
        """Test splitting with output files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_file = os.path.join(temp_dir, "train.json")
            test_file = os.path.join(temp_dir, "test.json")

            splitter = embedding.EmbeddingTrainTestSplitter(
                test_size=0.2,
                seed=42,
                train_output_file=train_file,
                test_output_file=test_file,
                _concurrency_mode='single',
                _save_data=False
            )
            inputs = [{"id": i, "query": f"query_{i}"} for i in range(10)]
            res = splitter(inputs)

            # Check files were created
            assert os.path.exists(train_file)
            assert os.path.exists(test_file)

            # Check file contents
            with open(train_file, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()
            with open(test_file, 'r', encoding='utf-8') as f:
                test_lines = f.readlines()

            assert len(train_lines) == 8  # 80%
            assert len(test_lines) == 2   # 20%

    # ==================== Pipeline Integration Test ====================

    def test_full_pipeline_without_llm(self):
        """Test complete embedding data pipeline without LLM (using mock data)"""
        # Step 1: Mock data (simulating QueryGenerator output)
        # 将 corpus 信息通过 passage 字段嵌入到数据中
        query_pos_data = [
            {"query": "龙美术馆在哪里？", "pos": ["龙美术馆位于上海浦东和徐汇"], "passage": "龙美术馆位于上海浦东和徐汇"},
            {"query": "谁创办了龙美术馆？", "pos": ["刘益谦、王薇夫妇创办了龙美术馆"], "passage": "刘益谦、王薇夫妇创办了龙美术馆"},
            {"query": "龙美术馆有哪些收藏？", "pos": ["龙美术馆收藏了书画、雕塑等艺术品"], "passage": "龙美术馆收藏了书画、雕塑等艺术品"},
            {"query": "故宫在哪里？", "pos": ["故宫博物院位于北京紫禁城"], "passage": "故宫博物院位于北京紫禁城"},
            {"query": "国家博物馆在哪？", "pos": ["中国国家博物馆在天安门广场东侧"], "passage": "中国国家博物馆在天安门广场东侧"},
            {"query": "上海博物馆是什么？", "pos": ["上海博物馆是一座大型古代艺术博物馆"], "passage": "上海博物馆是一座大型古代艺术博物馆"},
        ]

        # Step 2: Mine hard negatives
        miner = embedding.EmbeddingHardNegativeMiner(
            mining_strategy="random",
            num_negatives=2,
            _concurrency_mode='single',
            _save_data=False
        )
        triplet_data = miner(query_pos_data)

        assert len(triplet_data) == 6
        assert all('neg' in item for item in triplet_data)

        # Step 3: Format data
        formatter = embedding.EmbeddingDataFormatter(
            output_format="flagembedding",
            instruction="Query: ",
            _concurrency_mode='single',
            _save_data=False
        )
        formatted_data = formatter(triplet_data)

        assert len(formatted_data) == 6
        assert all('query' in item and 'pos' in item and 'neg' in item for item in formatted_data)

        # Step 4: Split train/test
        splitter = embedding.EmbeddingTrainTestSplitter(
            test_size=0.33,
            seed=42,
            _concurrency_mode='single',
            _save_data=False
        )
        final_data = splitter(formatted_data)

        assert len(final_data) == 6
        train_count = sum(1 for r in final_data if r["split"] == "train")
        test_count = sum(1 for r in final_data if r["split"] == "test")
        assert train_count == 4
        assert test_count == 2
