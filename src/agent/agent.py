"""`ruai agent`: agge, the multi-agent CLI, written in py0 (run by CPython too).

    python3 agent.py <op> ...           (PYTHONPATH=py0/python for ruai.json / ruai.http)
    ruai py run src/agent/agent.py <op> ...
"""
import sys

from cli import main

main(sys.argv[1:])
