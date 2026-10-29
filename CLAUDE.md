# fastbpe

纯 Python 实现的字节级 BPE 分词器（训练 + 编码 + 解码），目标是算法层面的极致性能。

## 硬性约束
- 运行时零第三方依赖（仅标准库），不得引入 C 扩展、Cython、numpy。
- 所有依赖与命令通过 uv 管理：`uv add`、`uv run`，禁止直接用 pip。
- 全量类型注解，`uv run mypy --strict src` 必须通过。
- `uv run ruff check --fix && uv run ruff format` 必须无告警。

## 命令
- 测试：`uv run pytest`
- 覆盖率：`uv run pytest --cov=fastbpe --cov-report=term-missing`
- 基准：`uv run pytest benchmarks/ --benchmark-only`
- 性能剖析：`uv run python -m cProfile -s cumtime scripts/profile_train.py`
- free-threaded 测试（CI 矩阵含 3.14t）：
  `UV_PROJECT_ENVIRONMENT=.venv-3.14t uv run --python 3.14t pytest -q`。
  必须用 `UV_PROJECT_ENVIRONMENT` 指向独立目录：直接 `uv run --python 3.14t` 会把项目的
  主 `.venv` 重建成 3.14t。`.venv-*/` 已在 `.gitignore` 中。
- pre-push 钩子：克隆后执行一次 `uv run pre-commit install`，之后每次推送前自动跑快速测试
  （`uv run pytest -q`，不含 `--runslow`）；完整测试由 CI 跑。

## 工作方式
- TDD：先写失败的测试，再实现。
- 用来判断测试或检查是否通过的命令，不得通过管道截断输出（`| tail`、`| head`、`| grep`
  会让整条命令的退出码变成最后一个命令的，失败会被掩盖）。需要截断时必须先
  `set -o pipefail`，或者用 pytest 自己的 `-q`、`--tb=short` 控制输出量。
  反例：dcfc7d6 就是因为 `pytest ... | tail -1 && git commit` 在测试失败时照样提交了。
  `.claude/hooks/require_pipefail.py`（PreToolUse hook）会拦下含管道 `|` 却没有
  `pipefail` 的 Bash 命令（`||` 不算管道；引号里的 `|`，如 `grep 'a\|b'`，也会被拦，
  加 `set -o pipefail;` 即可）；测试见 `tests/test_claude_hooks.py`。
- 关于外部库（tiktoken、`regex`、标准库 `unicodedata` 等）行为的结论，必须有提交到
  仓库、可重复运行的测试或脚本支撑，不得依据一次性探测下结论；写进 README、TODO、
  注释或提交信息之前先固定成测试。反例：PR #4 曾依据一次探测写下"tiktoken 的选择取决于
  id"，实际只取决于 token 字符串（`tests/test_tiktoken_differences.py`）。
- 任何性能优化必须：(1) 先跑基准记录基线 (2) 优化 (3) 再跑基准，在提交信息里写明提升倍数。
- 无基准证据的"优化"不得提交；优化不得降低可读性到无法维护（必要时写注释说明复杂度）。
- 正确性 oracle：在相同 merges 下，编码结果须与 tiktoken 参考实现一致（见 tests/test_oracle.py）。
- 平局规则：pair 频次相同时，选字节序列字典序更大者（与 GPT-2 训练一致）。

## 架构
- src/fastbpe/pretokenize.py  预分词与并行计数
- src/fastbpe/train.py        增量式合并训练
- src/fastbpe/encode.py       基于 rank 的编码 + 缓存
- src/fastbpe/tokenizer.py    公共 API、特殊 token、序列化

## 性能测量
- cProfile 会严重夸大预分词等细粒度调用的耗时，只用来定位热点；
  耗时占比和优化前后对比一律用 pytest-benchmark / perf_counter 实测。
- 基准必须包含对抗性输入：超长无空格串（base64、DNA、压缩 JS）、
  大量特殊 token、高重复与低重复语料。只在常规语料上验证过的优化不算完成。
