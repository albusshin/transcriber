#!/usr/bin/env python3
#
import argparse
import re
import json
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
import torch
import os
from datetime import timedelta


def perform(input_wav_file: str):
    print("Prepping file:", input_wav_file)

    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)

    audio = AudioSegment.from_wav(input_wav_file)

    prepped_input_wav_file = "prepped-" + os.path.basename(input_wav_file)
    audio = spacer.append(audio, crossfade=0)

    audio.export(prepped_input_wav_file, format="wav")

    print("File Prepped:", prepped_input_wav_file)

    hugging_face_access_token = "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"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hugging_face_access_token
    )

    dz_pipeline_file = {"uri": "blabla", "audio": prepped_input_wav_file}
    dz = pipeline(dz_pipeline_file)

    print(*list(dz.itertracks(yield_label=True))[:10], sep="\n")

    def millisec(timeStr):
        spl = timeStr.split(":")
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
        return s

    dzs = str(dz).splitlines()

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
    print(*groups, sep="\n")

    audio = AudioSegment.from_wav(prepped_input_wav_file)
    gidx = -1
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        audio[start:end].export(str(gidx) + ".wav", format="wav")
        print(f"group {gidx}: {start}--{end}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large-v2", device=device)

    for i in range(len(groups)):
        audiof = str(i) + ".wav"
        result = model.transcribe(
            audio=audiof, language="en", word_timestamps=True
        )  # , initial_prompt=result.get('text', ""))
        with open(str(i) + ".json", "w") as outfile:
            json.dump(result, outfile, indent=4)

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

    with open(f"capspeaker.txt", "w", encoding="utf-8") as file:
        s = "".join(txt)
        file.write(s)
        print("captions saved to capspeaker.txt:")
        print(s + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-wav-file", required=True, help="Path to the input WAV file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print("Input file: " + args.input_wav_file)
    perform(args.input_wav_file)
