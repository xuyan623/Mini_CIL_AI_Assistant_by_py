# AI Terminal Assistant (v2)

面向 Alpine Linux 的命令行 AI 助手，支持：

1. 普通对话
2. 文件与目录操作
3. 代码检查/解释/优化/生成
4. 代码上下文问答
5. 备份管理
6. 多模型配置切换
7. Shell 命令生成与安全提示

## 1. 运行要求

1. Python 3.8+
2. 推荐 Alpine Linux（其他 Linux/macOS/Windows 也可运行）
3. 可访问模型 API 的网络环境

## 2. 3 分钟上手

### 第一步：进入目录并看帮助

```bash
cd root
python ai.py -h
```

### 第二步：配置模型

推荐方式（避免把 API Key 留在 shell 历史）：

```bash
python config.py --add
```

命令行方式（便于自动化）：

```bash
python ai.py config add --profile deepseek --name "DeepSeek" \
  --api-key "sk-xxx" \
  --api-url "https://api.deepseek.com/v1/chat/completions" \
  --model "deepseek-reasoner" \
  --stream off
```

### 第三步：验证可用

```bash
python ai.py config current
python ai.py chat "你好"
python ai.py file ls .
```

## 3. 命令速查

### chat 普通对话

```bash
python ai.py chat "请解释一下这个项目是做什么的"
python ai.py --clear
```

### file 文件与目录

```bash
python ai.py file ls .
python ai.py file read ./README.md
python ai.py file search ./README.md "配置"
python ai.py file find "config" --dir .
python ai.py file rm ./tmp.txt --force
python ai.py file rmdir ./tmp_dir --force
```

### code 代码操作

```bash
python ai.py code check app.py --start 1 --end 80
python ai.py code explain app.py --start 1 --end 80
python ai.py code comment app.py --start 10 --end 40
python ai.py code optimize app.py --start 10 --end 40
python ai.py code generate app.py --start 1 --end 1 --desc "add argparse import"
python ai.py code summarize app.py

# 非交互写入（适合 shell 分步自动执行）
python ai.py code comment app.py --start 10 --end 40 --yes
python ai.py code optimize app.py --start 10 --end 40 --yes
python ai.py code generate app.py --start 1 --end 1 --desc "add argparse import" --yes
```

### context 代码上下文

```bash
python ai.py context set app.py --start 1 --end 120
python ai.py context add utils.py
python ai.py context list
python ai.py context ask "app.py 如何调用 utils.py?"
python ai.py context clear
```

### backup 备份管理

```bash
python ai.py backup create app.py --keep 5
python ai.py backup status
python ai.py backup status app.py
python ai.py backup list app.py
python ai.py backup restore <backup-file> --target app.py
python ai.py backup clean app.py --keep 3
```

### config 配置管理

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

### shell Shell 命令生成

```bash
python ai.py shell run "查找大于 100M 的文件"
```

运行流程：

1. 先执行“指代解析”（如“这个文件/它”），若唯一命中则自动回填路径
2. 信息不足时先生成可执行的发现步骤（如 `find`），不会展示占位符命令
3. 再用模型生成首批结构化步骤（必须是 JSON 协议）
4. 若模型输出非 JSON，会自动进行一次修复回合；修复失败则直接中止
5. 询问是否开始执行（`y/n`）
6. 每一步执行前再次确认（`y/n`）
7. 每一步执行后立即输出结果并写入历史
8. 下一步会基于上一步 `stdout/stderr/exit code` 重新规划，不是机械照抄草案

说明：

1. 指代歧义时（候选 >1）会停止并要求你明确路径，不会盲猜执行
2. 在 Alpine 环境下，Windows 风格路径（如 `C:\...`）默认不会自动采用
3. `--execute` 已下线，不再支持
4. 非交互终端只生成步骤，不会执行
5. 当模型返回空内容或失败时，会自动尝试其他已配置模型兜底
6. 执行过程中按 `Ctrl+C` 会优雅取消（退出码 130），不会输出 Python traceback

示例：

```bash
python ai.py shell run "先找 Sam.c 的路径"
python ai.py shell run "帮我备份这个文件"
```

第二条命令会优先解析“这个文件”为上一轮命中的文件实体，再生成备份命令。

旧版流程（仅供理解差异）：

1. 先用模型将自然语言解析为结构化任务，并生成首批步骤草案
2. 询问是否开始执行（`y/n`）
3. 每一步执行前再次确认（`y/n`）
4. 每一步执行后立即输出结果并写入历史
5. 下一步会基于上一步 `stdout/stderr/exit code` 重新规划，不是机械照抄草案

## 4. 让 `ai` 命令可直接使用（可选）

默认用 `python ai.py ...` 即可。  
如果你希望直接输入 `ai ...`：

```bash
cd root
chmod +x ai.py
ln -sf "$(pwd)/ai.py" /usr/local/bin/ai
ai -h
```

## 5. 数据目录说明

运行时数据都在 `root` 目录内：

1. 配置目录：`root/assistant-config`
2. 状态目录：`root/assistant-state`
3. 数据目录：`root/assistant-data`

主要文件：

1. `assistant-config/profiles.json`：模型配置
2. `assistant-state/history.json`：输入输出历史（含 messages/events/planner_traces/entities）
3. `assistant-state/context.json`：代码上下文
4. `assistant-data/backup_index.json`：备份索引

## 6. 常见问题

### Q1: `ai config -h` 运行失败

如果你还没创建 `ai` 命令，请改用：

```bash
python ai.py config -h
```

### Q2: `ai: command not found`

说明系统里还没有 `ai` 可执行入口，按上面的“让 `ai` 命令可直接使用”步骤创建软链接。

### Q3: 提示“当前配置缺少 API Key”

执行：

```bash
python ai.py config current
python config.py --add
python ai.py config switch <你的profile>
```

### Q4: 删除命令没有执行

`file rm` 和 `file rmdir` 会要求二次确认输入 `y`，未确认时会返回“已取消”。

## 7. 安全建议

1. 不要把真实 API Key 提交到 Git。
2. 使用 `.gitignore` 排除本地运行目录与敏感文件。
3. `config export` 建议使用 `--redact` 生成脱敏文件。
