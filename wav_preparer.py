import os
from pydub import AudioSegment
import logging


class InputWavFilePreparer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.front_padding_milliseconds = 2000
        self.prepared = False

    def _get_prepared_file_path(self):
        return os.path.join(os.getcwd(), f"prepared-{os.path.basename(self.file_path)}")

    def _get_prepared_file_name(self):
        return f"prepared-{os.path.basename(self.file_path)}"

    def _prep(self, output_file_path):
        logging.debug(f"Preparing file: {self.file_path}")
        audio.export(
            output_file_path,
            AudioSegment.silent(duration=front_padding_milliseconds).append(
                AudioSegment.from_wav(self.file_path), crossfade=0
            ),
            format="wav",
        )
        logging.debug("File prepared to:", output_file_path)

    ### Public methods
    def get_prepared_wav_file_path(self):
        if not self.prepared:
            self._prep(_get_prepared_file_path())

        return self._get_prepared_file_path()

    def get_prepared_wav_file_name(self):
        if not self.prepared:
            self._prep(_get_prepared_file_path())

        return self._get_prepared_file_name()

    def get_front_padding_in_milliseconds(self):
        return self.front_padding_milliseconds
