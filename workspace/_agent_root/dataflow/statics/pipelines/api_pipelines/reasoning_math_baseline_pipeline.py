"""从零合成数学推理基线流水线（暖启动锚点）。

CondorGenerator(MathCondorPrompt) 从零生成数学题 → ReasoningQuestionFilter 质量过滤
→ ReasoningAnswerGenerator 生成 CoT 推理链 → FormatterFilter + TokenLengthFilter + NgramFilter
最终产出字段：instruction（题目）、generated_cot（CoT 推理链，作为 SFT output）。

此文件由 adapt_baseline_source 正则替换注入运行时 api_url / model_name / key / paths。
"""
from dataflow.core.prompt import DIYPromptABC
from dataflow.operators.text_sft import CondorGenerator
from dataflow.operators.reasoning import (
    ReasoningQuestionFilter,
    ReasoningAnswerGenerator,
    ReasoningAnswerFormatterFilter,
    ReasoningAnswerTokenLengthFilter,
    ReasoningAnswerNgramFilter,
)
from dataflow.prompts.reasoning.math import MathQuestionFilterPrompt, MathAnswerGeneratorPrompt
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class MathCondorPrompt(DIYPromptABC):
    """数学领域 CondorGenerator prompt：生成 [Easy/Medium/Hard] 三道数学题。"""

    def __init__(self):
        self.tag = {
            "Algebra": {
                "Linear Equations": ["一元一次方程", "二元一次方程组", "参数方程"],
                "Quadratic Equations": ["一元二次方程", "韦达定理", "判别式"],
                "Polynomials": ["多项式运算", "因式分解", "余数定理"],
                "Inequalities": ["一元不等式", "绝对值不等式", "不等式组"],
                "Sequences": ["等差数列", "等比数列", "递推数列"],
                "Matrices": ["矩阵运算", "行列式", "线性方程组"],
            },
            "Calculus": {
                "Limits": ["数列极限", "函数极限", "无穷小比较"],
                "Derivatives": ["导数计算", "隐函数求导", "高阶导数"],
                "Integrals": ["不定积分", "定积分", "换元积分法"],
                "Series": ["级数收敛性", "幂级数", "泰勒展开"],
                "Differential Equations": ["一阶常微分方程", "二阶常微分方程", "分离变量法"],
                "Multivariate Calculus": ["偏导数", "重积分", "曲线积分"],
            },
            "Geometry": {
                "Plane Geometry": ["三角形性质", "圆的性质", "相似与全等"],
                "Solid Geometry": ["空间向量", "体积计算", "截面问题"],
                "Analytic Geometry": ["直线方程", "圆锥曲线", "参数方程"],
                "Trigonometry": ["三角恒等式", "三角方程", "反三角函数"],
                "Vectors": ["向量运算", "内积外积", "向量应用"],
                "Coordinate Geometry": ["距离公式", "面积计算", "坐标变换"],
            },
            "Number Theory": {
                "Divisibility": ["整除性", "最大公约数", "最小公倍数"],
                "Primes": ["质数判定", "唯一分解定理", "质数分布"],
                "Modular Arithmetic": ["同余方程", "中国剩余定理", "费马小定理"],
                "Diophantine Equations": ["不定方程", "Pell 方程", "整数解"],
                "Number Sequences": ["Fibonacci 数列", "完全数", "整数分拆"],
                "Cryptography Basics": ["RSA 原理", "离散对数", "欧拉函数"],
            },
            "Probability and Statistics": {
                "Basic Probability": ["古典概型", "条件概率", "全概率公式"],
                "Random Variables": ["离散分布", "连续分布", "期望与方差"],
                "Statistical Inference": ["参数估计", "假设检验", "置信区间"],
                "Combinatorial Probability": ["排列组合概率", "容斥原理", "鸽巢原理"],
                "Stochastic Processes": ["马尔可夫链", "随机游走", "泊松过程"],
                "Bayesian Methods": ["贝叶斯公式", "先验后验", "最大似然"],
            },
            "Combinatorics": {
                "Counting Principles": ["加法原理", "乘法原理", "分类计数"],
                "Permutations and Combinations": ["排列数", "组合数", "多重集排列"],
                "Graph Theory": ["图的连通性", "树", "欧拉路径"],
                "Generating Functions": ["普通生成函数", "指数生成函数", "递推求解"],
                "Pigeonhole Principle": ["鸽巢原理应用", "Ramsey 定理", "极值组合"],
                "Inclusion-Exclusion": ["容斥原理", "错排问题", "集合计数"],
            },
        }
        self.task_types = [
            "Problem Solving",
            "Proof",
            "Computation",
            "Application",
            "Derivation",
            "Estimation",
            "Construction",
        ]

    def build_prompt(self, theme, domain):
        prompt = f"""
Now we need to create high-quality math reasoning SFT data for LLM training. You only need to create \
Questions. I will give you a mathematical theme. You need to create three math Questions of different \
difficulty levels based on this theme.\\
Your Questions must meet the following requirements:\\
1. You must strictly create only three Questions at a time. These three Questions must be in the domain \
of {domain} and the Questions should align with the given mathematical theme of {theme}.\\
2. Each Question must be self-contained with all necessary conditions and constraints clearly stated. \
Include specific numerical values, variable definitions, and any relevant constraints. The problem should \
be unambiguous and solvable.\\
3. Your reply must strictly follow the format below. Your Questions need to be included between \
[Question Start] and [Question End], and the difficulty level should be indicated at the beginning, \
as in the following format:\\

[Easy][Question Start]Question[Question End]

[Medium][Question Start]Question[Question End]

[Hard][Question Start]Question[Question End]

4. Easy = single-step computation or direct formula application; Medium = multi-step reasoning with 2-3 \
logical steps; Hard = requires creative insight, proof, or combining multiple concepts.\\
5. All Questions must require mathematical reasoning to solve — no definition recall or trivia.\\
\quad \\

Now it's your turn. Please provide the three math Questions of different difficulty levels about the \
theme of {theme} for {domain}, according to the requirements.
"""
        return prompt


class ReasoningMathBaseline_APIPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="empty.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-5.5",   # 强基线用更强模型；经框架暖启动时会被 adapt 替换为 pipeline_llm.generation
            key_name_of_api_key="DF_PIPELINE_API_KEY",
            max_workers=100,        # 注意：adapt_baseline_source 不改写 max_workers，此字面量实际生效
        )
        self.generator = CondorGenerator(
            llm_serving=self.llm_serving,
            num_samples=800,        # //3 → ~266 prompt → ~798 题，与 generation.target_samples 对齐
            prompt_template=MathCondorPrompt(),
        )
        self.question_filter = ReasoningQuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. "
            "Follow the user's instructions strictly and output your final judgment "
            "in the required JSON format.",
            llm_serving=self.llm_serving,
            prompt_template=MathQuestionFilterPrompt(),
        )
        self.answer_generator = ReasoningAnswerGenerator(
            llm_serving=self.llm_serving,
            prompt_template=MathAnswerGeneratorPrompt(),
        )
        self.format_filter = ReasoningAnswerFormatterFilter()
        self.token_filter = ReasoningAnswerTokenLengthFilter(
            max_answer_token_length=8192,
            tokenizer_dir="Qwen/Qwen2.5-0.5B-Instruct",
        )
        self.ngram_filter = ReasoningAnswerNgramFilter(
            min_score=0.1,
            max_score=1.0,
            ngrams=5,
        )

    def forward(self):
        self.generator.run(storage=self.storage.step())
        self.question_filter.run(
            storage=self.storage.step(),
            input_key="instruction",
        )
        self.answer_generator.run(
            storage=self.storage.step(),
            input_key="instruction",
            output_key="generated_cot",
        )
        self.format_filter.run(
            storage=self.storage.step(),
            input_key="generated_cot",
        )
        self.token_filter.run(
            storage=self.storage.step(),
            input_key="generated_cot",
        )
        self.ngram_filter.run(
            storage=self.storage.step(),
            input_question_key="instruction",
            input_answer_key="generated_cot",
        )


if __name__ == "__main__":
    model = ReasoningMathBaseline_APIPipeline()
    model.forward()
