import subprocess
import os


def midi_to_wav(midi_path, sf2_path, out_path):

    

   
    assert os.path.exists(midi_path), "MIDI file not found"
    assert os.path.exists(sf2_path), "SF2 file not found"

   
    cmd = [
        "fluidsynth",
        "-F", out_path,
        "-r", "44100",
        sf2_path,
        midi_path
    ]

    subprocess.run(cmd, check=True)

    print("✅ WAV file saved at:", out_path)
