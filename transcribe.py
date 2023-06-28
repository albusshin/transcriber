import argparse
import re
import json
import os
from datetime import timedelta
import tempfile
from diarization_pipeline import DiarizationPipeline
from input_file_preparer import InputFilePreparer
from whisper_transcriber import WhisperTranscriber
from caption_writer import CaptionWriter
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Pipelines:
    def __init__(self, hugging_face_access_token: str):
        self.diarization_pipeline = DiarizationPipeline(hugging_face_access_token)
        self.whisper_transcriber = WhisperTranscriber("large-v2")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-aac-file", required=True, help="Path to the input AAC file"
    )
    parser.add_argument(
        "--hugging-face-access-token",
        required=True,
        help="Access token from https://hf.co",
    )
    parser.add_argument(
        "--output-txt-dir",
        required=True,
        help="Directory to output the transcribed text file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.debug(f"Changing into temporary directory {temp_dir}")

        os.chdir(temp_dir)
        input_file_preparer = InputFilePreparer(file_path=args.input_aac_file)

        pipelines = Pipelines(args.hugging_face_access_token)

        diarized_groups = pipelines.diarization_pipeline.get_diarized_groups(
            input_file_preparer.get_prepared_file_path()
        )

        input_file_preparer.export_mini_audio_files_from_groups(diarized_groups)

        pipelines.whisper_transcriber.transcribe_mini_audio_files_and_dump_json(
            diarized_groups, input_file_preparer
        )
        caption_writer = CaptionWriter(diarized_groups, input_file_preparer)
        caption_writer.generate_caption_txt_and_write_to_file(args.output_txt_dir)
