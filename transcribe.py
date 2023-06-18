#!/usr/bin/env python3
#
import argparse
import re
import json
import os
from datetime import timedelta
import tempfile
from timeit import default_timer as timer
from datetime import timedelta


def millisec(time_str):
    spl = time_str.split(":")
    return (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)


def perform(input_wav_file: str, hugging_face_access_token: str, output_txt_dir: str):
    print("Prepping file:", input_wav_file)

    spacermilli = 2000
    from pydub import AudioSegment

    spacer = AudioSegment.silent(duration=spacermilli)

    audio = AudioSegment.from_wav(input_wav_file)

    prepped_input_wav_file = "prepped-" + os.path.basename(input_wav_file)
    audio = spacer.append(audio, crossfade=0)

    audio.export(prepped_input_wav_file, format="wav")

    print("File Prepped:", os.path.join(os.getcwd(), prepped_input_wav_file))

    print("Loading Pipeline module")
    timer_beginning = timer()
    from pyannote.audio import Pipeline

    print("Pipeline module loaded in", timedelta(seconds=timer() - timer_beginning))

    print("Loading pretrained pipeline")
    timer_beginning = timer()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hugging_face_access_token
    )
    print("Pretrained pipeline loaded in", timedelta(seconds=timer() - timer_beginning))

    print("Diarizing...")
    timer_beginning = timer()
    dzs = str(pipeline({"uri": "blabla", "audio": prepped_input_wav_file})).splitlines()
    print("Diarized in", timedelta(seconds=timer() - timer_beginning))

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = millisec(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)

    print("Exporting mini audio files...")
    timer_beginning = timer()
    audio = AudioSegment.from_wav(prepped_input_wav_file)
    gidx = -1
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        audio[start:end].export(str(gidx) + ".wav", format="wav")

    del dzs, audio, pipeline
    print("Mini audio files exported in ", timedelta(seconds=timer() - timer_beginning))

    print("Starting transcribing using Whisper.")
    import whisper
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    timer_beginning = timer()
    model = whisper.load_model("large-v2", device=device)
    print("Model loaded in", timedelta(seconds=timer() - timer_beginning))

    len_groups = len(groups)
    print(f"Transcribing {len_groups} mini audio files...")
    timer_beginning = timer()
    for i in range(len_groups):
        audiof = str(i) + ".wav"
        timer_beginning_i = timer()
        print(f"Transcribing audio file\t {i} of {len_groups}...")
        result = model.transcribe(
            audio=audiof, language="en", word_timestamps=True
        )  # , initial_prompt=result.get('text', ""))

        print(
            f"Audio file \t{i} transcribed. Time:",
            timedelta(seconds=timer() - timer_beginning_i),
        )
        with open(str(i) + ".json", "w") as outfile:
            json.dump(result, outfile, indent=4)

    print("Total transcription time:", timedelta(seconds=timer() - timer_beginning))

    def timeStr(t):
        return "{0:02d}:{1:02d}:{2:06.2f}".format(
            round(t // 3600), round(t % 3600 // 60), t % 60
        )

    speakers = {}

    txt = list("")
    gidx = -1
    for g in groups:
        shift = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        shift = millisec(shift) - spacermilli  # the start time in the original video
        shift = max(shift, 0)

        gidx += 1

        captions = json.load(open(str(gidx) + ".json"))["segments"]

        if captions:
            speaker = g[0].split()[-1]
            if speaker in speakers:
                speaker = speakers[speaker]

            for c in captions:
                start = shift + c["start"] * 1000.0
                start = start / 1000.0  # time resolution ot youtube is Second.
                end = (shift + c["end"] * 1000.0) / 1000.0
                txt.append(
                    f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n'
                )

                for i, w in enumerate(c["words"]):
                    if w == "":
                        continue
                    start = (shift + w["start"] * 1000.0) / 1000.0

    output_file_path = os.path.join(
        os.path.expanduser(output_txt_dir),
        f"{os.path.basename(input_wav_file)}.transcribed.txt",
    )

    output_directory_path = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    with open(output_file_path, "w", encoding="utf-8") as file:
        s = "".join(txt)
        file.write(s)
        print(f"Captions saved to {output_file_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-wav-file", required=True, help="Path to the input WAV file"
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
    print("Input file: " + args.input_wav_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Changing into directory {temp_dir}")
        os.chdir(temp_dir)
        perform(
            args.input_wav_file, args.hugging_face_access_token, args.output_txt_dir
        )
