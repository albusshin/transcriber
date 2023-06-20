from helpers import millisec
import pytest


def test_millisec():
    assert millisec("00:00:05.000") == 5000
    assert millisec("00:01:00.500") == 60500
    assert millisec("01:01:01.111") == 3661111
    assert millisec("00:05:00.000") == 300000
    with pytest.raises(ValueError):
        millisec("")
    with pytest.raises(IndexError):
        assert millisec("00:01") == 60000
        assert millisec("02:30:") == 9000000
