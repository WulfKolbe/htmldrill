#!/usr/bin/env bash
# Resume the Claude session, setting the terminal title first.
# The OSC escape (ESC ]0; <text> BEL) sets the window/tab title; it reaches the
# real TTY because this script runs at your prompt. Override the title with $1.
printf '\033]0;%s\007' "${1:-WulfKolbe/htmldrill}"
claude --resume 62f5d8e0-8cda-40e0-ada5-f87bca3cc4e9 --dangerously-skip-permissions --permission-mode acceptEdits
