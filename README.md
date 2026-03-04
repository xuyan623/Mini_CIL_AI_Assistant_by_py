# AI Terminal Assistant (Alpine-first)

面向 Alpine Linux 的命令行 AI 助手，统一提供 `chat/file/code/context/backup/config/shell` 七类能力。

## 1. 项目定位

这个项目解决三个核心问题：

1. 把自然语言转成可执行的终端工作流（`ai shell run`）。
2. 把代码操作（检查/注释/优化/解释/生成）做成可追溯、可回滚流程。
3. 把配置、上下文、历史、备份统一收敛到 `root` 内部状态目录。

## 2. 运行环境

1. Python `3.8+`
2. Alpine Linux 优先（Linux/macOS/Windows 也可运行）
3. 可访问模型 API

## 3. 60 秒上手

在 `root` 目录执行：

```bash
cd root
python ai.py -h
```

添加模型配置（二选一）：

```bash
# 交互方式（推荐）
python config.py --add

# 命令行方式
python ai.py config add \
  --profile deepseek \
  --name "DeepSeek" \
  --api-key "sk-xxx" \
  --api-url "https://api.deepseek.com/v1/chat/completions" \
  --model "deepseek-reasoner" \
  --stream off
```

验证：

```bash
python ai.py config current
python ai.py chat "你好"
python ai.py file ls .
```

## 4. 命令总览

主帮助：

```bash
python ai.py -h
```

配置帮助：

```bash
python ai.py config -h
```

七大模块：

1. `chat`：普通对话
2. `file`：文件与目录
3. `code`：代码操作
4. `context`：代码上下文
5. `backup`：备份管理
6. `config`：模型配置
7. `shell`：自然语言分步执行

## 5. 常用命令示例

### 5.1 chat

```bash
python ai.py chat "请解释这个项目"
python ai.py --clear
```

### 5.2 file

```bash
python ai.py file ls .
python ai.py file read ./README.md
python ai.py file search ./README.md "配置"
python ai.py file find "Sam.c" --dir .
python ai.py file rm ./tmp.txt --force
python ai.py file rmdir ./tmp_dir --force
```

### 5.3 code

```bash
python ai.py code check app.py --start 1 --end 120
python ai.py code explain app.py --start 1 --end 120
python ai.py code comment app.py --start 1 --end 120
python ai.py code optimize app.py --start 1 --end 120
python ai.py code generate app.py --start 1 --end 1 --desc "add argparse import"
python ai.py code summarize app.py
```

写入类命令可加 `--yes` 跳过确认：

```bash
python ai.py code comment app.py --start 1 --end 120 --yes
python ai.py code optimize app.py --start 1 --end 120 --yes
python ai.py code generate app.py --start 1 --end 1 --desc "add argparse import" --yes
```

### 5.4 context

```bash
python ai.py context set app.py --start 1 --end 120
python ai.py context add utils.py
python ai.py context list
python ai.py context ask "app.py 如何调用 utils.py?"
python ai.py context clear
```

### 5.5 backup

```bash
python ai.py backup create app.py --keep 5
python ai.py backup status
python ai.py backup status app.py
python ai.py backup list app.py
python ai.py backup restore <backup-file> --target app.py
python ai.py backup clean app.py --keep 3
```

### 5.6 config

```bash
python ai.py config list
python ai.py config current
python ai.py config switch
python ai.py config switch deepseek
python ai.py config stream deepseek on
python ai.py config delete old_profile
python ai.py config export deepseek ./deepseek.profile.json --redact
python ai.py config import ./deepseek.profile.json --profile deepseek_copy
```

### 5.7 shell

```bash
python ai.py shell run "检查并修复 Sam.c"
python ai.py shell run "先找 Sam.c，再备份这个文件"
python ai.py shell run "./mycode 目录里没有 AI 就创建"
```

## 6. `shell run` 行为规则（重要）

`ai shell run` 不是一次性黑盒执行，而是“规划 + 确认 + 执行 + 再规划”的状态机：

1. 先做任务解释与指代解析（例如“这个文件”“它”）。
2. 信息不足时先生成可执行发现步骤（例如 `find/test/sed/wc`）。
3. 每一步都需要用户确认（`y/n`）。
4. 每一步执行后立刻展示 `exit_code/stdout/stderr`。
5. 下一步会基于上一步结果动态生成，而不是机械照抄草稿。
6. 模型输出非 JSON 时会自动修复一次；修复失败则中止。
7. 非交互终端只生成步骤，不执行。
8. `Ctrl+C` 优雅中断，退出码 `130`。
9. 若上一步已得到行数（如 `wc -l`），后续 `ai code ... --end` 会优先复用该数字。
10. 若目标文件被写入（`ai code comment/optimize/generate`），行数缓存自动失效并重算。
11. 纯探测冗余步骤会自动跳过（如重复 `test -f`、已唯一定位后的重复 `find`）。
12. 同一 shell trace 内会优先复用最近成功的模型配置，减少重复 fallback。
13. 规划失败时会输出 profile 级失败摘要（含 `error_preview`）。

附加约束：

1. Alpine 场景默认优先 Unix 路径。
2. Windows 风格路径（如 `C:\...`）不会被优先自动采用。
3. `--execute` 已下线，不再支持。

## 7. 数据与状态文件（全部在 root 内）

运行时文件固定在 `root` 下：

1. `assistant-config/profiles.json`：模型配置
2. `assistant-state/history.json`：历史（`messages/events/planner_traces/entities`）
3. `assistant-state/context.json`：代码上下文
4. `assistant-data/backup_index.json`：备份索引
5. `assistant-data/backups/`：备份文件

## 8. 开发与测试

基础测试：

```bash
cd root
python -m pytest -q
```

覆盖率门禁：

```bash
python -m pytest \
  --cov=ai_assistant \
  --cov-report=term-missing \
  --cov-report=annotate:cov_annotate \
  --cov-fail-under=90
```

当前仓库状态（2026-03-04）：

1. `python -m pytest -q` 可通过
2. 覆盖率门禁 `--cov-fail-under=90` 已启用并可通过

## 9. 常见问题

### Q1: `ai config switch` 不带参数会失败吗？

不会。TTY 下会进入交互选择；非交互环境会提示你显式传 `profile`。

### Q2: 为什么会看到明文 API Key 警告？

表示你把 key 写在配置文件里。建议使用环境变量并避免提交配置到仓库。

### Q3: 为什么 `shell run` 先问确认？

这是默认安全策略，避免自然语言误触发有副作用命令。

### Q4: 为什么有时提示“请补充目标文件名”？

因为当前描述缺少必要参数（例如只说“帮我修改”但没给文件）。

## 10. 安全建议

1. 不要提交真实 API Key。
2. 导出配置时优先使用 `--redact`。
3. 删除命令和写入命令都应先在测试文件验证。
