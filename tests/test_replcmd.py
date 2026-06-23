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
