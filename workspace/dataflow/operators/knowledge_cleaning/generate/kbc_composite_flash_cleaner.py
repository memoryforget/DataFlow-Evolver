from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
from dataflow import get_logger

# 引入基础算子
from dataflow.operators.knowledge_cleaning import (
    FileOrURLToMarkdownConverterFlash,
    KBCChunkGenerator,
    KBCTextCleaner
)

@OPERATOR_REGISTRY.register()
class KBCCompositeCleaningFlashOperator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        # Step 1: FileOrURLToMarkdownConverterFlash 参数
        intermediate_dir: str = "../example_data/KBCleaningPipeline/flash/",
        mineru_model_path: str = None,
        batch_size: int = 4,
        replicas: int = 1,
        num_gpus_per_replica: float = 1.0,
        engine_gpu_util_rate_to_ray_cap: float = 0.9,
        # Step 2: KBCChunkGenerator 参数
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        split_method: str = "token",
        min_tokens_per_chunk: int = 100,
        tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        # Step 3: KBCTextCleaner 参数
        lang: str = "en"
    ):
        super().__init__()
        self.logger = get_logger()
        
        # 实例化 Step 1: FlashMinerU 转换器
        self.step1_converter = FileOrURLToMarkdownConverterFlash(
            intermediate_dir=intermediate_dir,
            mineru_model_path=mineru_model_path,
            batch_size=batch_size,
            replicas=replicas,
            num_gpus_per_replica=num_gpus_per_replica,
            engine_gpu_util_rate_to_ray_cap=engine_gpu_util_rate_to_ray_cap
        )
        
        # 实例化 Step 2: 文本分块器
        self.step2_chunker = KBCChunkGenerator(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
            min_tokens_per_chunk=min_tokens_per_chunk,
            tokenizer_name=tokenizer_name
        )
        
        # 实例化 Step 3: LLM 文本清洗器
        self.step3_cleaner = KBCTextCleaner(
            llm_serving=llm_serving,
            lang=lang
        )

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "复合知识清洗算子（FlashMinerU加速版）：将本地GPU加速的文件转Markdown、文本分块以及LLM文本清洗组合为一个端到端的算子。\n"
                "执行顺序：\n"
                "1. FileOrURLToMarkdownConverterFlash: 使用本地GPU引擎批量将输入源解析为Markdown。\n"
                "2. KBCChunkGenerator: 读取Markdown路径，切分为原始文本块 (raw_chunk)。\n"
                "3. KBCTextCleaner: 使用大模型对原始文本块进行标准化，输出清洗后的文本 (cleaned_chunk)。"
            )
        elif lang == "en":
            return (
                "Composite Knowledge Cleaning Operator (FlashMinerU Accelerated): Combines GPU-accelerated File/URL to Markdown conversion, text chunking, and LLM-based text cleaning into a single pipeline.\n"
                "Execution Flow:\n"
                "1. FileOrURLToMarkdownConverterFlash: Parses input sources to Markdown using local GPU engine.\n"
                "2. KBCChunkGenerator: Reads Markdown and splits text into raw chunks.\n"
                "3. KBCTextCleaner: Standardizes raw chunks using LLM, outputting cleaned chunks."
            )

    def run(self, storage: DataFlowStorage, input_key: str = "source", output_key: str = "cleaned_chunk"):
        self.logger.info("Starting composite knowledge cleaning operator (Flash Version)...")
        
        intermediate_text_path_key = "text_path"
        intermediate_raw_chunk_key = "raw_chunk"
        
        # 步骤 1：文件/URL 转 Markdown (Flash版)
        self.logger.info("Step 1/3: Converting File/URL to Markdown via FlashMinerU...")
        self.step1_converter.run(
            storage=storage.step(),
            input_key=input_key,
            output_key=intermediate_text_path_key
        )
        
        # 步骤 2：文本分块
        self.logger.info("Step 2/3: Generating text chunks...")
        self.step2_chunker.run(
            storage=storage.step(),
            input_key=intermediate_text_path_key,
            output_key=intermediate_raw_chunk_key
        )
        
        # 步骤 3：文本清洗
        self.logger.info("Step 3/3: Cleaning text chunks using LLM...")
        self.step3_cleaner.run(
            storage=storage.step(),
            input_key=intermediate_raw_chunk_key,
            output_key=output_key
        )
        
        self.logger.info(f"Composite operator finished. Final output saved to column: {output_key}")
        
        return [output_key]