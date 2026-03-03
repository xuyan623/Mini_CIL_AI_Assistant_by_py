#!/bin/bash
# Bash completion for ai (v2)

_ai_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local top="chat file code context backup config shell"
    local file_sub="ls read search find rm rmdir"
    local code_sub="check comment explain optimize generate summarize"
    local context_sub="set add list ask clear"
    local backup_sub="create status list restore clean"
    local config_sub="add switch list current delete stream export import"
    local shell_sub="run"

    case "${COMP_WORDS[1]}" in
        file)
            COMPREPLY=($(compgen -W "${file_sub}" -- "${cur}"))
            return 0
            ;;
        code)
            COMPREPLY=($(compgen -W "${code_sub}" -- "${cur}"))
            return 0
            ;;
        context)
            COMPREPLY=($(compgen -W "${context_sub}" -- "${cur}"))
            return 0
            ;;
        backup)
            COMPREPLY=($(compgen -W "${backup_sub}" -- "${cur}"))
            return 0
            ;;
        config)
            COMPREPLY=($(compgen -W "${config_sub}" -- "${cur}"))
            return 0
            ;;
        shell)
            COMPREPLY=($(compgen -W "${shell_sub}" -- "${cur}"))
            return 0
            ;;
    esac

    COMPREPLY=($(compgen -W "${top}" -- "${cur}"))
    return 0
}

complete -F _ai_complete ai
