#compdef ai
# zsh completion for ai (v2)

_ai() {
  local -a top file_sub code_sub context_sub backup_sub config_sub shell_sub
  top=(chat file code context backup config shell)
  file_sub=(ls read search find rm rmdir)
  code_sub=(check comment explain optimize generate summarize)
  context_sub=(set add list ask clear)
  backup_sub=(create status list restore clean)
  config_sub=(add switch list current delete stream export import)
  shell_sub=(run)

  if (( CURRENT == 2 )); then
    _describe 'top commands' top
    return
  fi

  case ${words[2]} in
    file) _describe 'file commands' file_sub ;;
    code) _describe 'code commands' code_sub ;;
    context) _describe 'context commands' context_sub ;;
    backup) _describe 'backup commands' backup_sub ;;
    config) _describe 'config commands' config_sub ;;
    shell) _describe 'shell commands' shell_sub ;;
    *) _files ;;
  esac
}

_ai "$@"
