"""Toy-corpus SentencePiece BPE comparison. Not GPT/tiktoken outputs.

Requires: pip install sentencepiece==0.2.1
Run: python sentencepiece_compare.py
"""
from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
base = [
    "I like programming with Swift and Python.",
    "Swift programming is fun. I like programming.",
    "func fetchUserProfileAsync() async throws -> UserProfile { return profile }",
    "func fetchUser() async throws -> User { return user }",
    'let programmingLanguage = "Swift"',
    "我喜欢 Swift programming。",
    "你好，世界！今天学习 Tokenization。",
    "在 Swift 中，我们可以编写异步函数，也喜欢编程。",
    "Hello, world! Programming is interesting.",
    "UserProfile fetchUser programming programmer programs",
    "字符串中的空格、标点、emoji 😀 都可能影响分词。",
    "The developer writes a new function named fetchUserProfileAsync.",
]
corpus = ROOT / "toy_corpus.txt"
corpus.write_text("\n".join(base * 40) + "\n", encoding="utf-8")

examples = [
    "我喜欢 Swift programming",
    "Hello, world!",
    "func fetchUser() async throws",
    "fetchUserProfileAsync",
    "programming",
    "我喜欢编程 😀",
]
print("sentencepiece version:", spm.__version__)
print("training lines:", len(base) * 40, " unique templates:", len(base))
for size in (128, 256):
    prefix = str(ROOT / f"toy_bpe_{size}")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=size,
        model_type="bpe",
        character_coverage=1.0,
        hard_vocab_limit=True,
        num_threads=1,
        shuffle_input_sentence=False,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=prefix + ".model")
    print("\nVOCAB", size, "actual", tokenizer.vocab_size())
    for sample in examples:
        pieces = tokenizer.encode(sample, out_type=str)
        ids = tokenizer.encode(sample, out_type=int)
        decoded = tokenizer.decode(ids)
        print(repr(sample), "tokens=", len(ids),
              "pieces=", pieces, "roundtrip=", decoded == sample)
        assert decoded == sample, (sample, decoded)
