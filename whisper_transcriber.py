import logging
from profiling import timer_decorator
from input_file_preparer import InputFilePreparer
import json


class WhisperTranscriber:
    def __init__(self, model_name):
        import whisper
        import torch

        logging.debug(f"Loading whisper {model_name} model...")
        self.model = whisper.load_model(
            # TODO Make model name configurable
            model_name,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    @timer_decorator
    def transcribe_mini_audio_files_and_dump_json(
        self,
        diarized_groups,
        input_file_preparer: InputFilePreparer,
    ):
        total_mini_audio_files = len(diarized_groups)
        for i in range(total_mini_audio_files):
            logging.debug(
                f"Transcribing audio file\t {i} of {total_mini_audio_files}..."
            )
            result = self.model.transcribe(
                audio=input_file_preparer.get_mini_audio_file_name_for_diarized_group_number(
                    i
                ),
                language="en",
                word_timestamps=True,
            )  # , initial_prompt=result.get('text', ""))

            logging.debug(
                f"Audio file \t{input_file_preparer.get_mini_audio_file_name_for_diarized_group_number(i)} transcribed."
            )
            with open(
                input_file_preparer.get_json_file_name_for_diarized_group_number(i),
                "w",
            ) as json_out_file_i:
                json.dump(result, json_out_file_i, indent=4)
