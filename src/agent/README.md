# `ruai agent`: agge in py0

agge is a git-style CLI for teams of LLM agents that write code and books. This directory is
agge rewritten in **py0** (ruai's typed subset of Python), module for module:

| module | contents |
|---|---|
| `agent.py` | entry point |
| `cli.py` | subcommands and option parsing (argparse is outside py0) |
| `repl.py` | interactive chat: slash commands, `@agent`, sessions, `/run` |
| `runtime.py` | the agent loop, tool execution, delegation, the graph executor |
| `graph.py` | graphs: edges, dependencies, validation, templates, mermaid |
| `tools.py` | the text-protocol tools: read / write / edit files, list, search, shell |
| `providers.py` | echo, OpenAI-compatible and Anthropic models (streaming, retries) |
| `project.py` | the `.agge/` directory: config, agents, graphs, sessions, run logs |
| `presets.py` | the built-in agents, teams and graphs (JSON) |
| `ui.py` | terminal colours and progress events |

The same files run under CPython (`ruai.json` and `ruai.http` have stand-ins in
`py0/python/ruai/`):

```bash
PYTHONPATH=py0/python python3 src/agent/agent.py help
ruai py run src/agent/agent.py help        # translated, compiled, cached
ruai agent help                            # the translation built into ruai
```

After changing a module, update the translation that `ruai` embeds (`tests/agent.rs` fails
until it is current) and rebuild:

```bash
ruai py emit src/agent/agent.py --module > src/cli/agent_gen.rs
```

Differences from the Python original: no threads (several `<delegate>` tags in one reply, the
ready nodes of a graph and the items of a map node run one after another; `--workers` is
accepted and ignored), no readline history or tab completion in the chat, and Ctrl-C stops
the program. Configuration, sessions and run logs are agge's (`.agge/`, `~/.agge` or
`$AGGE_HOME`).
