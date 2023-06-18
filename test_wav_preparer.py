#!/usr/bin/env python3

import pytest
import os
from wav_preparer import InputWavFilePreparer


def test_init():
    input_wave_file_preparer = InputWavFilePreparer("/fake/path/to/input.wav")
    assert input_wave_file_preparer.file_path == "/fake/path/to/input.wav"
    assert input_wave_file_preparer.prepared == False
    assert (
        input_wave_file_preparer.get_front_padding_in_milliseconds()
        == input_wave_file_preparer.front_padding_milliseconds
    )


def test_private_get_prepared_file_name():
    input_wave_file_preparer = InputWavFilePreparer("/fake/path/to/input.wav")
    assert input_wave_file_preparer._get_prepared_file_name() == "prepared-input.wav"


def test_private_get_prepared_file_path():
    input_wave_file_preparer = InputWavFilePreparer("/fake/path/to/input.wav")
    assert input_wave_file_preparer._get_prepared_file_path() == os.path.join(
        os.getcwd(), "prepared-input.wav"
    )
