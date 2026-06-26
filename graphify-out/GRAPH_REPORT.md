# Graph Report - /home/wai/src/hosaka  (2026-06-26)

## Corpus Check
- Corpus is ~44,939 words - fits in a single context window. You may not need a graph.

## Summary
- 615 nodes · 1269 edges · 44 communities (33 shown, 11 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8f109fa4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_test piper proto.py|test piper proto.py]]
- [[_COMMUNITY_test lexicon.py|test lexicon.py]]
- [[_COMMUNITY_test rvc proto.py|test rvc proto.py]]
- [[_COMMUNITY_split fragments()|split fragments()]]
- [[_COMMUNITY_test replcmd.py|test replcmd.py]]
- [[_COMMUNITY_app.py|app.py]]
- [[_COMMUNITY_VoiceLibrary|VoiceLibrary]]
- [[_COMMUNITY_RvcEngine|RvcEngine]]
- [[_COMMUNITY_SourceCache|SourceCache]]
- [[_COMMUNITY_config.py|config.py]]
- [[_COMMUNITY_repl.py|repl.py]]
- [[_COMMUNITY_test server.py|test server.py]]
- [[_COMMUNITY_EngineRegistry|EngineRegistry]]
- [[_COMMUNITY_PacatPlayer|PacatPlayer]]
- [[_COMMUNITY_client with rvc()|client with rvc()]]
- [[_COMMUNITY_test audio.py|test audio.py]]
- [[_COMMUNITY_glados fx.py|glados fx.py]]
- [[_COMMUNITY_FfplayPlayer|FfplayPlayer]]
- [[_COMMUNITY_$()|$()]]
- [[_COMMUNITY_WinSoundPlayer|WinSoundPlayer]]
- [[_COMMUNITY_Spy|Spy]]
- [[_COMMUNITY_audio.py|audio.py]]
- [[_COMMUNITY_HangEngine|HangEngine]]
- [[_COMMUNITY_GatedEngine|GatedEngine]]
- [[_COMMUNITY_BoomEngine|BoomEngine]]
- [[_COMMUNITY_smoke server.sh|smoke server.sh]]
- [[_COMMUNITY_FakeProc|FakeProc]]
- [[_COMMUNITY_CudaBoomEngine|CudaBoomEngine]]
- [[_COMMUNITY_PCMPlayer|PCMPlayer]]
- [[_COMMUNITY_fetch rvc model.sh script|fetch rvc model.sh script]]
- [[_COMMUNITY_setup rvc venv.sh script|setup rvc venv.sh script]]
- [[_COMMUNITY_fetch glados model.sh script|fetch glados model.sh script]]
- [[_COMMUNITY_setup bake venv.sh script|setup bake venv.sh script]]
- [[_COMMUNITY_setup piper venv.sh script|setup piper venv.sh script]]
- [[_COMMUNITY_setup server venv.sh script|setup server venv.sh script]]
- [[_COMMUNITY_start server.sh script|start server.sh script]]
- [[_COMMUNITY_hosaka|hosaka]]

## God Nodes (most connected - your core abstractions)
1. `VoiceLibrary` - 47 edges
2. `EngineRegistry` - 33 edges
3. `split_fragments()` - 29 edges
4. `parse_line()` - 26 edges
5. `_client()` - 24 edges
6. `RvcEngine` - 22 edges
7. `SourceCache` - 20 edges
8. `create_app()` - 20 edges
9. `FfplayPlayer` - 18 edges
10. `PcmCache` - 18 edges

## Surprising Connections (you probably didn't know these)
- `_FakeProc` --uses--> `PacatPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `_FakeProc` --uses--> `WinSoundPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `_FakeProc` --uses--> `FfplayPlayer`  [INFERRED]
  tests/test_audio.py → hosaka/audio.py
- `FakeSource` --uses--> `SourceCache`  [INFERRED]
  tests/test_rvc_engine.py → hosaka/cache.py
- `FakeEngine` --uses--> `VoiceLibrary`  [INFERRED]
  tests/test_engines_gpu.py → hosaka/library.py

## Import Cycles
- None detected.

## Communities (44 total, 11 thin omitted)

### Community 0 - "test piper proto.py"
Cohesion: 0.09
Nodes (42): PiperEngine, Piper's length_scale is a duration multiplier (higher = slower). Map the     REP, Client for the out-of-process Piper sidecar (.venv-piper).      The server venv, speed_to_length_scale(), encode_request(), pack_audio(), pack_end(), pack_error() (+34 more)

### Community 1 - "test lexicon.py"
Cohesion: 0.10
Nodes (36): add_entry(), _anchored(), apply_lexicon(), _compile(), Lexicon, load_map(), Custom-pronunciation lexicon: respell words so the TTS lands them right.  A flat, Mtime-cached view of the lexicon file for the server's hot path.      Reloads (a (+28 more)

### Community 2 - "test rvc proto.py"
Cohesion: 0.12
Nodes (35): encode_request(), pack_audio(), pack_end(), pack_error(), Wire protocol between the server-side RvcEngine client and the isolated .venv-rv, Read one request (JSON header line + length-prefixed PCM block) from a     binar, The sidecar reported a synthesis failure (error frame)., The frame stream was malformed or ended before the end marker. (+27 more)

### Community 3 - "split fragments()"
Cohesion: 0.10
Nodes (37): _boundary(), _dash_ms(), _pause_marker(), pause_ms(), _ramp_wrap(), Split one dash-free segment into fragments. `start` is the number of     spoken, Split s into pieces no longer than limit, breaking on word boundaries., Emit fragments whose size cap grows geometrically from first_max_chars     to ma (+29 more)

### Community 4 - "test replcmd.py"
Cohesion: 0.11
Nodes (31): _input_lines(), Yield logical input lines using readline (importing it above wires the     editi, parse_line(), ReplAction, _fake_input(), Drive _input_lines: feed strings to return, exception instances to raise., test_backend_accepts_all_three(), test_backend_unknown_is_error() (+23 more)

### Community 5 - "app.py"
Cohesion: 0.11
Nodes (23): BaseException, BaseModel, FastAPI, _clamp(), clamp_params(), SpeechParams, SpeechRequest, VoiceInfo (+15 more)

### Community 6 - "VoiceLibrary"
Cohesion: 0.13
Nodes (16): bake(), main(), ChatterboxEngine, Original Chatterbox cloning + tuning, run as a NON-realtime QUALITY mode.      O, _win_temp_dir(), VoiceEntry, VoiceLibrary, Path (+8 more)

### Community 7 - "RvcEngine"
Cohesion: 0.17
Nodes (22): Client for the out-of-process RVC sidecar (.venv-rvc).      RVC converts timbre,, RvcEngine, Popen, _engine(), FakeSource, _ram_cache(), RAM-only source cache (no disk) for cache-behavior tests., Stand-in Kokoro: yields two deterministic chunks so a test can check the     eng (+14 more)

### Community 8 - "SourceCache"
Cohesion: 0.11
Nodes (15): PcmCache, RAM-hot, disk-durable PCM cache (the RvcEngine source cache).      A small in-RA, Bounded, thread-safe LRU cache of PCM byte blobs, keyed by a hashable key., SourceCache, test_disabled_cache_never_stores(), test_get_miss_then_hit(), test_lru_evicts_oldest_by_bytes(), test_put_same_key_replaces_and_adjusts_bytes() (+7 more)

### Community 9 - "config.py"
Cohesion: 0.11
Nodes (22): native_to_pct(), pct_to_native(), Inverse of pct_to_native, for showing values on the 0-100 scale., The source voice id for an RVC voice's source engine.      For a Kokoro source (, Map a 0-100 knob value onto its native range; 50 -> the default., resolve_source(), _check_gpu(), main() (+14 more)

### Community 10 - "repl.py"
Cohesion: 0.11
Nodes (26): _ensure_server(), main(), Clear any failed state and (re)start the managed systemd unit., Fallback: run our own uvicorn when no systemd unit manages the server., Make a healthy server reachable, deferring to systemd when it owns the     port, Resolve a voice's (backend, cb_params, speed) from the registry; fall back     o, Decide how the REPL should obtain a server. Pure policy, no I/O.      Returns on, (LoadState, ActiveState) of the managed unit, or (None, None) when     systemd o (+18 more)

### Community 11 - "test server.py"
Cohesion: 0.16
Nodes (22): _client(), _drain_ws(), Read one utterance off the socket: a start marker, PCM binary frames,     then a, test_app_static_page_served(), test_health_ok(), test_lock_released_after_request(), test_shutdown_route_exists(), test_speech_chatterbox_empty_voice_ok() (+14 more)

### Community 12 - "EngineRegistry"
Cohesion: 0.15
Nodes (11): Engine, EngineRegistry, KokoroEngine, Kokoro-82M. Presets + speed. 24 kHz mono output., Protocol, FakeEngine, test_chatterbox_clones_from_seed(), test_kokoro_streams_audio() (+3 more)

### Community 13 - "PacatPlayer"
Cohesion: 0.13
Nodes (6): PacatPlayer, Play one complete utterance (uniform interface with WinSoundPlayer)., Streams raw PCM to PulseAudio via pacat. Correct for native Linux audio., test_default_cmd_targets_pacat(), test_pacat_end_utterance_is_noop(), test_player_writes_bytes_to_subprocess()

### Community 14 - "client with rvc()"
Cohesion: 0.14
Nodes (14): _client_with_piper(), _client_with_rvc(), FakeEngine, PiperFakeEngine, RvcFakeEngine, test_speech_piper_voice_streams(), test_speech_rvc_voice_streams(), test_speech_unknown_piper_voice_is_400() (+6 more)

### Community 15 - "test audio.py"
Cohesion: 0.19
Nodes (9): _find_ffplay(), Locate ffplay: prefer it on PATH, else the winget install location., _roundtrip(), test_byte_writes_realign_across_chunk_boundaries(), test_find_ffplay_absent(), test_find_ffplay_on_path(), test_find_ffplay_via_winget_glob(), test_gain_clips_to_unit_range() (+1 more)

### Community 16 - "glados fx.py"
Cohesion: 0.21
Nodes (14): bandpass(), comb(), formant_shift(), _load_mono(), main(), pitch_shift(), process(), Warp the spectral envelope along frequency to move formants while     keeping pi (+6 more)

### Community 17 - "FfplayPlayer"
Cohesion: 0.22
Nodes (6): FfplayPlayer, Streams float32 PCM into a persistent ffplay.exe on the Windows host.      ffpla, test_ffplay_broken_pipe_is_handled(), test_ffplay_end_utterance_flushes_short_utterance(), test_ffplay_holds_lead_then_streams(), test_ffplay_relaunches_after_broken_pipe()

### Community 18 - "$()"
Cohesion: 0.31
Nodes (9): $(), ensureAudio(), loadVoices(), openSocket(), PARAM_IDS, params(), reflectBackend(), selectedBackend() (+1 more)

### Community 19 - "WinSoundPlayer"
Cohesion: 0.22
Nodes (5): Plays full utterances on the Windows host, bypassing WSLg/PulseAudio.      WSLg', WinSoundPlayer, test_winsound_accumulates_then_plays_on_end(), test_winsound_prepends_lead_silence(), test_winsound_writes_gained_wav_and_invokes_player()

### Community 20 - "Spy"
Cohesion: 0.36
Nodes (6): _reg(), Spy, test_get_rvc_raises_when_absent(), test_get_rvc_returns_engine_when_present(), test_warmup_all_skips_rvc_when_absent(), test_warmup_all_warms_rvc_when_present()

### Community 21 - "audio.py"
Cohesion: 0.22
Nodes (5): _gain_align(), _gained_pcm16(), Apply gain to a byte chunk on 4-byte (float32) boundaries.      Network chunks m, ndarray, test_gain_align_scales_and_holds_partial_float()

### Community 22 - "HangEngine"
Cohesion: 0.22
Nodes (5): Event, HangEngine, Engine whose stream() blocks without ever yielding -- simulates a wedged     GPU, test_wedged_generation_triggers_shutdown(), _wait()

### Community 23 - "GatedEngine"
Cohesion: 0.33
Nodes (4): GatedEngine, Engine whose stream() blocks until released, so a test can pin the GPU     slot, test_concurrent_request_queues_instead_of_503(), test_queue_cap_returns_503_when_full()

### Community 24 - "BoomEngine"
Cohesion: 0.40
Nodes (3): BoomEngine, test_ordinary_engine_error_does_not_shutdown(), test_speech_engine_error_is_not_silent()

### Community 25 - "smoke server.sh"
Cohesion: 0.67
Nodes (3): smoke_server.sh script, play(), PYTHONPATH

## Knowledge Gaps
- **8 isolated node(s):** `PARAM_IDS`, `hosaka`, `fetch_glados_model.sh script`, `setup_bake_venv.sh script`, `setup_piper_venv.sh script` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VoiceLibrary` connect `VoiceLibrary` to `app.py`, `config.py`, `repl.py`, `test server.py`, `EngineRegistry`, `client with rvc()`, `HangEngine`, `GatedEngine`, `BoomEngine`, `CudaBoomEngine`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Why does `EngineRegistry` connect `EngineRegistry` to `app.py`, `config.py`, `test server.py`, `client with rvc()`, `Spy`, `HangEngine`, `GatedEngine`, `BoomEngine`, `CudaBoomEngine`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `RvcEngine` connect `RvcEngine` to `SourceCache`, `config.py`, `test rvc proto.py`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `VoiceLibrary` (e.g. with `ChatterboxEngine` and `_GpuQueue`) actually correct?**
  _`VoiceLibrary` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `EngineRegistry` (e.g. with `_GpuQueue` and `Spy`) actually correct?**
  _`EngineRegistry` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `True under WSLg, whose RDP audio bridge can't play cleanly (see below).`, `Apply gain to a byte chunk on 4-byte (float32) boundaries.      Network chunks m`, `Locate ffplay: prefer it on PATH, else the winget install location.` to the rest of the system?**
  _84 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test piper proto.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09361393323657474 - nodes in this community are weakly interconnected._