# Graph Report - /home/wai/src/hosaka  (2026-06-27)

## Corpus Check
- 4 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 657 nodes · 1318 edges · 46 communities (33 shown, 13 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RvcEngine|RvcEngine]]
- [[_COMMUNITY_test_piper_proto.py|test_piper_proto.py]]
- [[_COMMUNITY_app.py|app.py]]
- [[_COMMUNITY_test_server.py|test_server.py]]
- [[_COMMUNITY_test_lexicon.py|test_lexicon.py]]
- [[_COMMUNITY_split_fragments()|split_fragments()]]
- [[_COMMUNITY_test_replcmd.py|test_replcmd.py]]
- [[_COMMUNITY_test_rvc_proto.py|test_rvc_proto.py]]
- [[_COMMUNITY_repl.py|repl.py]]
- [[_COMMUNITY_config.py|config.py]]
- [[_COMMUNITY_VoiceLibrary|VoiceLibrary]]
- [[_COMMUNITY_test_audio.py|test_audio.py]]
- [[_COMMUNITY_PacatPlayer|PacatPlayer]]
- [[_COMMUNITY_EngineRegistry|EngineRegistry]]
- [[_COMMUNITY_ChatterboxEngine|ChatterboxEngine]]
- [[_COMMUNITY_glados_fx.py|glados_fx.py]]
- [[_COMMUNITY_FfplayPlayer|FfplayPlayer]]
- [[_COMMUNITY_WinSoundPlayer|WinSoundPlayer]]
- [[_COMMUNITY_app.js|app.js]]
- [[_COMMUNITY_audio.py|audio.py]]
- [[_COMMUNITY_Spy|Spy]]
- [[_COMMUNITY_HangEngine|HangEngine]]
- [[_COMMUNITY_Engine|Engine]]
- [[_COMMUNITY_GatedEngine|GatedEngine]]
- [[_COMMUNITY_smoke_server.sh|smoke_server.sh]]
- [[_COMMUNITY__FakeProc|_FakeProc]]
- [[_COMMUNITY_PCMPlayer|PCMPlayer]]
- [[_COMMUNITY_fetch_rvc_model.sh script|fetch_rvc_model.sh script]]
- [[_COMMUNITY_setup_rvc_venv.sh script|setup_rvc_venv.sh script]]
- [[_COMMUNITY_fetch_glados_model.sh script|fetch_glados_model.sh script]]
- [[_COMMUNITY_setup_bake_venv.sh script|setup_bake_venv.sh script]]
- [[_COMMUNITY_setup_piper_venv.sh script|setup_piper_venv.sh script]]
- [[_COMMUNITY_setup_server_venv.sh script|setup_server_venv.sh script]]
- [[_COMMUNITY_start_server.sh script|start_server.sh script]]
- [[_COMMUNITY_sync_lexicon.sh script|sync_lexicon.sh script]]
- [[_COMMUNITY_update_glados_droplet.sh script|update_glados_droplet.sh script]]
- [[_COMMUNITY_hosaka|hosaka]]

## God Nodes (most connected - your core abstractions)
1. `VoiceLibrary` - 47 edges
2. `EngineRegistry` - 36 edges
3. `split_fragments()` - 29 edges
4. `parse_line()` - 24 edges
5. `_client()` - 24 edges
6. `create_app()` - 23 edges
7. `RvcEngine` - 22 edges
8. `SourceCache` - 20 edges
9. `FfplayPlayer` - 18 edges
10. `PcmCache` - 18 edges

## Surprising Connections (you probably didn't know these)
- `_FakeProc` --uses--> `PacatPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `_FakeProc` --uses--> `WinSoundPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `_FakeProc` --uses--> `FfplayPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `FakeEngine` --uses--> `VoiceLibrary`  [INFERRED]
  tests/test_engines_gpu.py → hosaka/library.py
- `BoomEngine` --uses--> `VoiceLibrary`  [INFERRED]
  tests/test_server.py → hosaka/library.py

## Import Cycles
- None detected.

## Communities (46 total, 13 thin omitted)

### Community 0 - "RvcEngine"
Cohesion: 0.07
Nodes (40): Client for the out-of-process RVC sidecar (.venv-rvc).      RVC converts timbre,, RvcEngine, The sidecar reported a synthesis failure (error frame)., The frame stream was malformed or ended before the end marker., RvcProtocolError, RvcSidecarError, PcmCache, RAM-hot, disk-durable PCM cache (the RvcEngine source cache).      A small in-RA (+32 more)

### Community 1 - "test_piper_proto.py"
Cohesion: 0.08
Nodes (48): PiperEngine, Piper's length_scale is a duration multiplier (higher = slower). Map the     REP, Client for the out-of-process Piper sidecar (.venv-piper).      The server venv, speed_to_length_scale(), encode_request(), pack_audio(), pack_end(), pack_error() (+40 more)

### Community 2 - "app.py"
Cohesion: 0.07
Nodes (36): BaseException, BaseModel, FastAPI, _cardinal(), normalize_times(), Pre-chunk text normalization for spoken forms the engines mishandle.  Engine G2P, Spell 0-59 as words ('forty-five'). Out of range -> str(n)., Render an in-range clock time as a spoken phrase. (+28 more)

### Community 3 - "test_server.py"
Cohesion: 0.08
Nodes (41): _client(), _client_piper_only(), _client_with_piper(), _client_with_rvc(), CudaBoomEngine, _drain_ws(), FakeEngine, PiperFakeEngine (+33 more)

### Community 4 - "test_lexicon.py"
Cohesion: 0.10
Nodes (36): add_entry(), _anchored(), apply_lexicon(), _compile(), Lexicon, load_map(), Custom-pronunciation lexicon: respell words so the TTS lands them right.  A flat, Mtime-cached view of the lexicon file for the server's hot path.      Reloads (a (+28 more)

### Community 5 - "split_fragments()"
Cohesion: 0.10
Nodes (37): _boundary(), _dash_ms(), _pause_marker(), pause_ms(), _ramp_wrap(), Split one dash-free segment into fragments. `start` is the number of     spoken, Split s into pieces no longer than limit, breaking on word boundaries., Emit fragments whose size cap grows geometrically from first_max_chars     to ma (+29 more)

### Community 6 - "test_replcmd.py"
Cohesion: 0.11
Nodes (31): _input_lines(), Yield logical input lines using readline (importing it above wires the     editi, parse_line(), ReplAction, _fake_input(), Drive _input_lines: feed strings to return, exception instances to raise., test_backend_accepts_all_three(), test_backend_unknown_is_error() (+23 more)

### Community 7 - "test_rvc_proto.py"
Cohesion: 0.15
Nodes (30): encode_request(), pack_audio(), pack_end(), pack_error(), Wire protocol between the server-side RvcEngine client and the isolated .venv-rv, Read one request (JSON header line + length-prefixed PCM block) from a     binar, Yield converted PCM payloads frame by frame until the end marker.     Raises Rvc, _read_exact() (+22 more)

### Community 8 - "repl.py"
Cohesion: 0.11
Nodes (28): _ensure_server(), main(), _push_lexicon(), Clear any failed state and (re)start the managed systemd unit., Fallback: run our own uvicorn when no systemd unit manages the server., Push the lexicon to the droplet so the always-on glados picks up a :pron     cha, Make a healthy server reachable, deferring to systemd when it owns the     port, Resolve a voice's (backend, cb_params, speed) from the registry; fall back     o (+20 more)

### Community 9 - "config.py"
Cohesion: 0.11
Nodes (20): native_to_pct(), pct_to_native(), Inverse of pct_to_native, for showing values on the 0-100 scale., The source voice id for an RVC voice's source engine.      For a Kokoro source (, Map a 0-100 knob value onto its native range; 50 -> the default., resolve_source(), _check_gpu(), main() (+12 more)

### Community 10 - "VoiceLibrary"
Cohesion: 0.19
Nodes (11): bake(), main(), VoiceEntry, VoiceLibrary, Path, Piper-only composition root for the always-on droplet container.  Builds the hos, _make_wav(), test_add_then_get_and_list() (+3 more)

### Community 11 - "test_audio.py"
Cohesion: 0.16
Nodes (12): _find_ffplay(), _gain_align(), Apply gain to a byte chunk on 4-byte (float32) boundaries.      Network chunks m, Locate ffplay: prefer it on PATH, else the winget install location., _roundtrip(), test_byte_writes_realign_across_chunk_boundaries(), test_find_ffplay_absent(), test_find_ffplay_on_path() (+4 more)

### Community 12 - "PacatPlayer"
Cohesion: 0.13
Nodes (6): PacatPlayer, Play one complete utterance (uniform interface with WinSoundPlayer)., Streams raw PCM to PulseAudio via pacat. Correct for native Linux audio., test_default_cmd_targets_pacat(), test_pacat_end_utterance_is_noop(), test_player_writes_bytes_to_subprocess()

### Community 13 - "EngineRegistry"
Cohesion: 0.23
Nodes (10): EngineRegistry, FakeEngine, test_chatterbox_clones_from_seed(), test_kokoro_streams_audio(), test_registry_routes_by_backend(), test_registry_unknown_backend_raises(), test_warmup_all_warms_both(), BoomEngine (+2 more)

### Community 14 - "ChatterboxEngine"
Cohesion: 0.19
Nodes (7): ChatterboxEngine, Original Chatterbox cloning + tuning, run as a NON-realtime QUALITY mode.      O, KokoroEngine, Kokoro-82M. Presets + speed. 24 kHz mono output., first_chunk_ms(), full_gen_ms(), main()

### Community 15 - "glados_fx.py"
Cohesion: 0.21
Nodes (14): bandpass(), comb(), formant_shift(), _load_mono(), main(), pitch_shift(), process(), Warp the spectral envelope along frequency to move formants while     keeping pi (+6 more)

### Community 16 - "FfplayPlayer"
Cohesion: 0.22
Nodes (6): FfplayPlayer, Streams float32 PCM into a persistent ffplay.exe on the Windows host.      ffpla, test_ffplay_broken_pipe_is_handled(), test_ffplay_end_utterance_flushes_short_utterance(), test_ffplay_holds_lead_then_streams(), test_ffplay_relaunches_after_broken_pipe()

### Community 17 - "WinSoundPlayer"
Cohesion: 0.18
Nodes (6): Plays full utterances on the Windows host, bypassing WSLg/PulseAudio.      WSLg', _win_temp_dir(), WinSoundPlayer, test_winsound_accumulates_then_plays_on_end(), test_winsound_prepends_lead_silence(), test_winsound_writes_gained_wav_and_invokes_player()

### Community 18 - "app.js"
Cohesion: 0.31
Nodes (8): ensureAudio(), loadVoices(), openSocket(), PARAM_IDS, params(), reflectBackend(), selectedBackend(), speak()

### Community 19 - "audio.py"
Cohesion: 0.22
Nodes (6): _gained_pcm16(), make_player(), on_wslg(), True under WSLg, whose RDP audio bridge can't play cleanly (see below)., Pick the playback path: ffplay (streaming) on WSLg when present, else the     bu, ndarray

### Community 20 - "Spy"
Cohesion: 0.36
Nodes (6): _reg(), Spy, test_get_rvc_raises_when_absent(), test_get_rvc_returns_engine_when_present(), test_warmup_all_skips_rvc_when_absent(), test_warmup_all_warms_rvc_when_present()

### Community 21 - "HangEngine"
Cohesion: 0.22
Nodes (5): Event, HangEngine, Engine whose stream() blocks without ever yielding -- simulates a wedged     GPU, test_wedged_generation_triggers_shutdown(), _wait()

### Community 23 - "GatedEngine"
Cohesion: 0.33
Nodes (4): GatedEngine, Engine whose stream() blocks until released, so a test can pin the GPU     slot, test_concurrent_request_queues_instead_of_503(), test_queue_cap_returns_503_when_full()

### Community 25 - "smoke_server.sh"
Cohesion: 0.67
Nodes (3): smoke_server.sh script, play(), PYTHONPATH

## Knowledge Gaps
- **10 isolated node(s):** `PARAM_IDS`, `hosaka`, `fetch_glados_model.sh script`, `setup_bake_venv.sh script`, `setup_piper_venv.sh script` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VoiceLibrary` connect `VoiceLibrary` to `app.py`, `test_server.py`, `config.py`, `EngineRegistry`, `ChatterboxEngine`, `HangEngine`, `GatedEngine`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `EngineRegistry` connect `EngineRegistry` to `test_piper_proto.py`, `app.py`, `test_server.py`, `config.py`, `VoiceLibrary`, `Spy`, `HangEngine`, `Engine`, `GatedEngine`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `_input_lines()` connect `test_replcmd.py` to `repl.py`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `VoiceLibrary` (e.g. with `ChatterboxEngine` and `_GpuQueue`) actually correct?**
  _`VoiceLibrary` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `EngineRegistry` (e.g. with `_GpuQueue` and `Spy`) actually correct?**
  _`EngineRegistry` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `True under WSLg, whose RDP audio bridge can't play cleanly (see below).`, `Apply gain to a byte chunk on 4-byte (float32) boundaries.      Network chunks m`, `Locate ffplay: prefer it on PATH, else the winget install location.` to the rest of the system?**
  _94 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `RvcEngine` be split into smaller, more focused modules?**
  _Cohesion score 0.07093253968253968 - nodes in this community are weakly interconnected._