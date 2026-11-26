# Answered questions

Decisions Wai made from the open-items doc, newest first.

| When | Question | Wai's answer | Reading taken |
|------|----------|--------------|---------------|
| 2026-11-26 | Should the strip's nomo stop nicole? (The spec said nomo = hosaka-server stopped; always-on kept nicole up in idle.) | "the toggle only has two modes, lomo and homo / when the home server is down, replace this with \"unavailable\"" | No nomo: the strip offers lomo and homo only. The box's emo/idle read as lomo (nicole is always on), and a dead TTS or an unreachable box shows "unavailable" in place of the buttons. Both switches restart hosaka-server, so both ask before cutting listeners. |
| 2026-11-25 | How should nicole stay on all the time, without loading Charlie? (Options: home box on the 1660; droplet CPU Kokoro; both.) | Home box: nicole always loaded on the GTX 1660. | hosaka-server runs in every GPU mode with nicole + GLaDOS; only homo loads Charlie (Chatterbox + RVC). The engine set is fixed at server start from a state file, so switching in or out of homo restarts the server. Qwen still wins: while Qwen holds the GPUs, hosaka-server stays down. Nicole is down when the PC is off. |
