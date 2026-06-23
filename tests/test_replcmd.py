from hosaka.cli.replcmd import parse_line


def test_plain_text_is_speak():
    a = parse_line("Hello world")
    assert (a.kind, a.value) == ("speak", "Hello world")


def test_voice_command():
    assert parse_line(":voice af_bella").value == ("af_bella", "")
    assert parse_line(":voice af_bella").kind == "voice"


def test_voice_command_with_inline_text():
    a = parse_line(":voice calm_brit hello there")
    assert a.kind == "voice"
    assert a.value == ("calm_brit", "hello there")


def test_voice_command_no_name_is_error():
    assert parse_line(":voice").kind == "error"


def test_set_param_parses_float():
    a = parse_line(":exag 0.7")
    assert a.kind == "set_param"
    assert a.value == ("exaggeration", 0.7)


def test_param_aliases():
    assert parse_line(":cfg 0.4").value == ("cfg_weight", 0.4)
    assert parse_line(":temp 0.8").value == ("temperature", 0.8)
    assert parse_line(":speed 1.2").value == ("speed", 1.2)


def test_bad_param_value_is_error():
    assert parse_line(":exag fast").kind == "error"


def test_vol_sets_volume():
    a = parse_line(":vol 1.8")
    assert a.kind == "volume"
    assert a.value == 1.8


def test_vol_bad_value_is_error():
    assert parse_line(":vol loud").kind == "error"


def test_status_command():
    assert parse_line(":status").kind == "status"
    assert parse_line(":info").kind == "status"


def test_quit_variants():
    assert parse_line(":quit").kind == "quit"
    assert parse_line(":quit --stop").kind == "quit_stop"


def test_unknown_command_is_error():
    assert parse_line(":frobnicate").kind == "error"


def test_pron_bare_and_list_are_list():
    assert parse_line(":pron").value == ("list", None)
    assert parse_line(":pron list").value == ("list", None)
    assert parse_line(":pron").kind == "pron"


def test_pron_add_single_word_respelling():
    a = parse_line(":pron add Wai Way")
    assert a.kind == "pron"
    assert a.value == ("add", ("Wai", "Way"))


def test_pron_add_multiword_respelling():
    assert parse_line(":pron add hosaka ho sock uh").value == (
        "add",
        ("hosaka", "ho sock uh"),
    )


def test_pron_add_missing_respelling_is_error():
    assert parse_line(":pron add Wai").kind == "error"


def test_pron_rm():
    assert parse_line(":pron rm Wai").value == ("rm", "Wai")
    assert parse_line(":pron del Wai").value == ("rm", "Wai")


def test_pron_rm_needs_one_word():
    assert parse_line(":pron rm").kind == "error"


def test_pron_unknown_subcommand_is_error():
    assert parse_line(":pron frob x").kind == "error"


def _fake_input(monkeypatch, items):
    """Drive _input_lines: feed strings to return, exception instances to raise."""
    seq = iter(items)

    def fake(prompt=""):
        v = next(seq)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("builtins.input", fake)


def test_input_lines_pass_through(monkeypatch):
    from hosaka.cli.repl import _input_lines

    _fake_input(monkeypatch, ["hello", ":voice af_heart", EOFError()])
    assert list(_input_lines("")) == ["hello", ":voice af_heart"]


def test_input_lines_flattens_multiline_paste(monkeypatch):
    from hosaka.cli.repl import _input_lines

    _fake_input(monkeypatch, ["first line\nsecond line\nthird", EOFError()])
    assert list(_input_lines("")) == ["first line second line third"]


def test_input_lines_eof_ends_iteration(monkeypatch):
    from hosaka.cli.repl import _input_lines

    _fake_input(monkeypatch, [EOFError()])
    assert list(_input_lines("")) == []


def test_input_lines_ctrl_c_abandons_line_and_continues(monkeypatch, capsys):
    from hosaka.cli.repl import _input_lines

    _fake_input(monkeypatch, [KeyboardInterrupt(), "after", EOFError()])
    assert list(_input_lines("")) == ["after"]


def test_speak_streams_chunks_then_ends(monkeypatch):
    import hosaka.cli.repl as repl

    events = []

    class FakePlayer:
        def write(self, chunk):
            events.append(("write", bytes(chunk)))

        def end_utterance(self):
            events.append(("end", None))

    class FakeResp:
        status_code = 200

        def iter_bytes(self):
            yield b"\x00\x00\x00\x00"
            yield b"\x01\x01\x01\x01"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(repl.httpx, "stream", lambda *a, **k: FakeResp())
    repl._speak(FakePlayer(), "kokoro", "af_heart", {}, "hi")

    assert events == [
        ("write", b"\x00\x00\x00\x00"),
        ("write", b"\x01\x01\x01\x01"),
        ("end", None),
    ]
