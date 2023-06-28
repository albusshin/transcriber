import logging
from profiling import timer_decorator
import re
from helpers import millisec


class DiarizationPipeline:
    def __init__(self, hugging_face_access_token: str):
        self.hugging_face_access_token = hugging_face_access_token
        self.load_pretrained_diarization_model()

    @timer_decorator
    def load_pretrained_diarization_model(self):
        from pyannote.audio import Pipeline

        logging.info("Loading pretrained diarization module")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=self.hugging_face_access_token,
        )

        import torch

        if torch.has_cuda:
            self.pipeline.to(torch.device("cuda"))

    @timer_decorator
    def _diarize(self, file_path: str):
        logging.info(f"Diarizing file: {file_path}")
        return self.pipeline({"uri": "blabla", "audio": file_path})

    def get_diarized_groups(self, file_path: str):
        diarized = str(self._diarize(file_path)).splitlines()

        groups = []
        g = []
        lastend = 0

        for d in diarized:
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
        return groups
