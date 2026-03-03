# AI Terminal Assistant (v2)

Alpine Linux first command-line AI assistant for file operations, code analysis/modification, context Q&A, backup management, and model profile switching.

## Key Changes in v2

1. New `argparse` subcommand CLI.
2. State files are stored under `root/assistant-state` by default.
3. Reliable history persistence with file lock + atomic writes.
4. Backup system rebuilt with `source_id` + backup index to avoid same-name collisions.
5. Profile stream control via `ai.py config stream`.

## Runtime

1. Python 3.8+
2. Alpine Linux recommended
3. Network access required for AI API calls

## Quick Start

```bash
cd root

# show help
python ai.py -h

# chat
python ai.py chat "你好"

# list directory
python ai.py file ls .
```

## New CLI

### Chat

```bash
python ai.py chat "如何读取 CSV？"
python ai.py --clear
```

### File

```bash
python ai.py file ls .
python ai.py file read ./README.md
python ai.py file search ./README.md "backup"
python ai.py file find "config" --dir .
python ai.py file rm ./tmp.txt --force
python ai.py file rmdir ./tmp_dir --force
```

### Code

```bash
python ai.py code check app.py --start 1 --end 80
python ai.py code explain app.py --start 1 --end 80
python ai.py code comment app.py --start 10 --end 40
python ai.py code optimize app.py --start 10 --end 40
python ai.py code generate app.py --start 1 --end 1 --desc "add argparse import"
python ai.py code summarize app.py
```

### Context

```bash
python ai.py context set app.py --start 1 --end 120
python ai.py context add utils.py
python ai.py context list
python ai.py context ask "app.py 如何调用 utils.py?"
python ai.py context clear
```

### Backup

```bash
python ai.py backup create app.py --keep 5
python ai.py backup status
python ai.py backup status app.py
python ai.py backup list app.py
python ai.py backup restore <backup-file> --target app.py
python ai.py backup clean app.py --keep 3
```

### Config

```bash
python ai.py config add --profile deepseek --name "DeepSeek" \
  --api-key "sk-xxx" --api-url "https://api.deepseek.com/v1/chat/completions" \
  --model "deepseek-reasoner" --stream off

python ai.py config list
python ai.py config current
python ai.py config switch deepseek
python ai.py config stream deepseek on
python ai.py config delete old_profile
```

### Shell

```bash
python ai.py shell run "查找大于 100M 的文件"
python ai.py shell run "查找大于 100M 的文件" --execute
```

## Storage Paths

1. Config: `root/assistant-config`
2. State: `root/assistant-state`
3. Data: `root/assistant-data`

## Security Note

1. Avoid committing real API keys.
2. Use `.gitignore` and private local config.
3. Prefer environment-level secret injection in production.
