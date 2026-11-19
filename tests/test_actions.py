import pytest

from aerorsi.actions import decode, encode, mode, native_format


def test_decode_bins():
    action, stop, valid = decode("98 49 49")
    assert valid and not stop
    assert action == pytest.approx({"fwd": 5.0, "down": 0.0, "yaw": 0.0})


def test_zero_action_is_a_stop():
    action, stop, valid = decode("00 49 49")
    assert valid and stop and action["fwd"] == 0.0


def test_land_and_malformed():
    assert decode("00 49 00 LAND")[1]
    assert not decode("fly forward")[2]
    assert decode("LAND")[2]


@pytest.mark.parametrize("text", ["52 49 49", "10 30 70", "00 49 49 LAND", "98 00 98"])
def test_encode_round_trip(text):
    action, stop, _ = decode(text)
    assert encode(action, stop) == text


def test_native_format():
    assert native_format("52 49 49") and native_format("00 49 49 LAND")
    assert not native_format("52 49") and not native_format("5 49 49") and not native_format("52 49 49 GO")


def test_mode():
    assert mode("52 49 49") == "translate"
    assert mode("00 30 49") == "vertical"
    assert mode("52 49 98") == "turn"
    assert mode("00 49 49") == "LAND"
    assert mode("go") == "invalid"
