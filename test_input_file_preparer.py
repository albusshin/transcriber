import pytest
import os
from input_file_preparer import InputFilePreparer
import tempfile
import functools

ORIGINAL_WORKING_DIR = os.getcwd()
SAMPLE_10SEC_PATH = os.path.join(ORIGINAL_WORKING_DIR, "samples/sample-10sec.aac")
SAMPLE_DIARIZED_GROUP = [
    [
        "[ 00:00:02.653 -->  00:00:03.216] A SPEAKER_00",
        "[ 00:00:04.906 -->  00:00:06.817] B SPEAKER_00",
        "[ 00:00:08.336 -->  00:00:08.370] C SPEAKER_00",
        "[ 00:00:09.308 -->  00:00:53.796] D SPEAKER_00",
    ],
    ["[ 00:00:22.414 -->  00:00:22.551] F SPEAKER_01"],
    ["[ 00:00:27.790 -->  00:00:28.284] G SPEAKER_01"],
    ["[ 00:00:39.837 -->  00:00:39.872] H SPEAKER_01"],
    ["[ 00:00:54.598 -->  00:01:02.005] E SPEAKER_00"],
]


def with_chdir_temporary_directory(test_function):
    @functools.wraps(test_function)
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            return test_function(*args, **kwargs)

    return wrapper


@with_chdir_temporary_directory
def test_init_object():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)
    assert input_file_preparer.file_path == SAMPLE_10SEC_PATH
    assert (
        input_file_preparer.get_front_padding_in_milliseconds()
        == input_file_preparer.front_padding_milliseconds
    )
    assert input_file_preparer.copied_wav_format_file_name == "sample-10sec.wav"


@with_chdir_temporary_directory
def test_get_original_input_file_name():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)

    assert input_file_preparer.get_original_input_file_name() == "sample-10sec.aac"


@with_chdir_temporary_directory
def test_get_prepared_file_name():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)

    assert input_file_preparer.get_prepared_file_name() == "prepared-sample-10sec.wav"


@with_chdir_temporary_directory
def test_get_prepared_file_path():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)
    assert input_file_preparer.get_prepared_file_path() == os.path.join(
        os.getcwd(), "prepared-sample-10sec.wav"
    )


@with_chdir_temporary_directory
def test_wav_file_save_path_is_in_current_dir():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)
    assert not os.path.exists(
        os.path.join(ORIGINAL_WORKING_DIR, input_file_preparer.get_prepared_file_name())
    )
    assert os.path.exists(
        os.path.join(os.getcwd(), input_file_preparer.get_prepared_file_name())
    )

    assert not os.path.exists(
        os.path.join(
            ORIGINAL_WORKING_DIR, input_file_preparer.copied_wav_format_file_name
        )
    )
    assert os.path.exists(
        os.path.join(os.getcwd(), input_file_preparer.copied_wav_format_file_name)
    )


@with_chdir_temporary_directory
def test_export_mini_audio_files_from_groups():
    input_file_preparer = InputFilePreparer(SAMPLE_10SEC_PATH)
    input_file_preparer.export_mini_audio_files_from_groups(SAMPLE_DIARIZED_GROUP)
    for i in range(len(SAMPLE_DIARIZED_GROUP)):
        i_file_name = f"sample-10sec.segment.{i}.wav"
        assert os.path.exists(os.path.join(os.getcwd(), i_file_name))
