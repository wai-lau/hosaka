from dataclasses import dataclass

_PARAMS = {
    "exag": "exaggeration",
    "cfg": "cfg_weight",
    "temp": "temperature",
    "speed": "speed",
}


@dataclass
class ReplAction:
    kind: str
    value: object = None


def parse_line(line: str) -> ReplAction:
    line = line.strip()
    if not line:
        return ReplAction("speak", "")
    if not line.startswith(":"):
        return ReplAction("speak", line)

    parts = line[1:].split()
    cmd, args = parts[0], parts[1:]

    if cmd in _PARAMS:
        if len(args) != 1:
            return ReplAction("error", f"usage: :{cmd} <number>")
        try:
            return ReplAction("set_param", (_PARAMS[cmd], float(args[0])))
        except ValueError:
            return ReplAction("error", f":{cmd} needs a number")
    if cmd == "voice":
        if not args:
            return ReplAction("error", "usage: :voice <name> [text]")
        return ReplAction("voice", (args[0], " ".join(args[1:])))
    if cmd == "clone":
        return (
            ReplAction("clone", args[0]) if args else ReplAction("error", "usage: :clone <id|path>")
        )
    if cmd == "backend":
        if args and args[0] in ("kokoro", "chatterbox"):
            return ReplAction("backend", args[0])
        return ReplAction("error", "usage: :backend kokoro|chatterbox")
    if cmd == "voices":
        return ReplAction("voices")
    if cmd == "help":
        return ReplAction("help")
    if cmd == "quit":
        return ReplAction("quit_stop") if args == ["--stop"] else ReplAction("quit")
    return ReplAction("error", f"unknown command: :{cmd}")
