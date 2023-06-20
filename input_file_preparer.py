import os
from pydub import AudioSegment
import logging
import re
from helpers import millisec


class InputFilePreparer:
    def __init__(self, file_path):
        self.file_path = os.path.expanduser(file_path)
        self.front_padding_milliseconds = 2000
        self._export_wav_format_file()
        self._prepare_front_padded_file()

    def _export_wav_format_file(self):
        base_name = self.get_original_input_file_base_name()
        self.copied_wav_format_file_name = f"{base_name}.wav"

        logging.debug(
            f"Creating {self.copied_wav_format_file_name} from {self.file_path}"
        )
        input_aac_file = AudioSegment.from_file(self.file_path, "aac")
        input_aac_file.export(self.copied_wav_format_file_name, format="wav")
        logging.debug(f"Created {self.copied_wav_format_file_name}")

    def get_prepared_file_name(self):
        return f"prepared-{self.copied_wav_format_file_name}"

    def get_prepared_file_path(self):
        return os.path.join(os.getcwd(), self.get_prepared_file_name())

    def get_original_input_file_base_name(self):
        base_name, _ = os.path.splitext(os.path.basename(self.file_path))
        return base_name

    def get_original_input_file_name(self):
        return os.path.basename(self.file_path)

    def _prepare_front_padded_file(self):
        logging.debug(f"Preparing file: {self.file_path}")

        AudioSegment.silent(duration=self.front_padding_milliseconds).append(
            AudioSegment.from_wav(self.copied_wav_format_file_name), crossfade=0
        ).export(
            self.get_prepared_file_path(),
            format="wav",
        )
        logging.debug("File prepared to:", self.get_prepared_file_path())

    def export_mini_audio_files_from_groups(self, diarized_groups):
        logging.debug(f"Exporting mini audio files...")
        audio = AudioSegment.from_wav(self.get_prepared_file_path())
        for i, g in enumerate(diarized_groups):
            start = millisec(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0])
            end = millisec(re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1])
            mini_audio_file_name = (
                self.get_mini_audio_file_name_for_diarized_group_number(i)
            )
            audio[start:end].export(mini_audio_file_name, format="wav")

    def get_front_padding_in_milliseconds(self):
        return self.front_padding_milliseconds

    def get_mini_audio_file_name_for_diarized_group_number(self, i: int):
        return f"{self.get_original_input_file_base_name()}.segment.{i}.wav"

    def get_json_file_name_for_diarized_group_number(self, i: int):
        return f"{self.get_original_input_file_base_name()}.segment.{i}.json"
