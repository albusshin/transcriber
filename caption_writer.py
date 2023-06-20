from helpers import millisec
from input_file_preparer import InputFilePreparer

import json


class CaptionWriter:
    def __init__(self, diarized_groups, input_file_preparer: InputFilePreparer):
        self.diarized_groups = diarized_groups
        self.input_file_preparer = input_file_preparer

    def generate_caption_txt_and_write_to_file(self, output_txt_dir):
        def timeStr(t):
            return "{0:02d}:{1:02d}:{2:06.2f}".format(
                round(t // 3600), round(t % 3600 // 60), t % 60
            )

        txt = list("")
        for i, group in enumerate(self.diarized_groups):
            shift = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=group[0])[0]
            shift = (
                millisec(shift)
                - self.input_file_preparer.get_front_padding_in_milliseconds()
            )  # the start time in the original video
            shift = max(shift, 0)

            with open(
                self.input_file_preparer.get_json_file_name_for_diarized_group_number(i)
            ) as json_wav_file:
                captions = json.load(json_wav_file)["segments"]

                if captions:
                    speaker = group[0].split()[-1]

                    for c in captions:
                        start = shift + c["start"] * 1000.0
                        start = start / 1000.0  # time resolution ot youtube is Second.
                        end = (shift + c["end"] * 1000.0) / 1000.0
                        txt.append(
                            f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n'
                        )

                        for _, w in enumerate(c["words"]):
                            if w == "":
                                continue
                            start = (shift + w["start"] * 1000.0) / 1000.0

        output_file_path = os.path.join(
            os.path.expanduser(output_txt_dir),
            f"{self.input_file_preparer.get_original_input_file_name()}.transcribed.txt",
        )

        output_directory_path = os.path.dirname(output_file_path)
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("".join(txt))
            logging.info(f"Captions saved to: {output_file_path}")
