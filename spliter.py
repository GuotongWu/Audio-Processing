import auditok
import os
import json

person = json.load(open('meta.json', 'r', encoding = 'utf-8'))['person']
# remember to renew meta.json for person name

def separate(filename:str, personid:int, sampleid:int):
    # split returns a generator of AudioRegion objects
    audio_regions = auditok.split(
        filename,
        min_dur=0.001,     # minimum duration of a valid audio event in seconds
        max_dur=2,       # maximum duration of an event
        max_silence=0.3, # maximum duration of tolerated continuous silence within an event
        energy_threshold=55 # threshold of detection
    )

    for i, r in enumerate(audio_regions):

        # Regions returned by `split` have 'start' and 'end' metadata fields
        print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))

        # play detection
        # r.play(progress_bar=True)
        r.save('%s/ %d-%d-%d.wav' % (os.path.dirname(filename), personid, sampleid, i))
        # region's metadata can also be used with the `save` method
        # (no need to explicitly specify region's object and `format` arguments)

name = 'wgt'
for i, item in enumerate(os.listdir(name)):
    separate('%s/%s' % (name, item), person[name], i)