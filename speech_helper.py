# pip install librosa
# apt install ffmpeg

import librosa, os


def get_silence_waveform(y, y_rms, sr, fr_len, fr_int):

    win_len = int(fr_len * sr)  # 80ms
    win_int = int(fr_int * sr)  # 20ms

    # Constants
    HINRGDIV = 50  # Dead silence frames ratio (100)
    LOX = 1.5  # Multiply low (min) energy estimate by this
    PERCENTTHRESH = 0.07  # thresh = % of max-min (Nominal = 5% = 0.05)
    NFRAMMAX = 0.2 / fr_int  # At least n silence frames needed (10=200ms)
    SFMAX = 3  # Max number of non-silence frames to skip
    END = False  # True=Mark end frames.  False=Mark internal sil.

    ### Compute rms threshold
    sortrms = list(y_rms)
    sortrms.sort()
    hinrg = sortrms[int(0.9 * len(sortrms))]

    dead = 0  # Find and discount any dead silence (< 90%_max/50) frames
    for ss in sortrms:
        if ss < hinrg / HINRGDIV:
            dead += 1  # Count # of dead silence frames
    print("dead =", dead)
    lonrg = sortrms[int(0.1 * len(sortrms)) + dead]  # Find 10th (+ dead) percentile

    speechthresh = (hinrg - lonrg) * PERCENTTHRESH + LOX * lonrg  # = % of max-min
    print("lonrg =", lonrg)
    print("hinrg =", hinrg)
    print("speechthresh =", speechthresh)
    ### End of computing threshold

    sil_vec = [0] * len(y_rms)
    sframes = 0  # Number of non-silence frames detected
    nframes = 0  # Number of silence frames detected
    ii = 0

    for jj in range(0, len(y_rms)):
        if y_rms[jj] < speechthresh:
            if nframes == 0:
                start_sec = ii  # Mark start of silence
                start_idx = jj
            end_sec = ii
            end_idx = jj
            nframes = nframes + 1
            sframes = 0
        else:
            if sframes <= 4:
                sframes = sframes + 1

        if (sframes == 4) or (ii + win_int >= len(y)):
            if (nframes >= NFRAMMAX) or (ii + win_int >= len(y)):
                sil_vec[start_idx : end_idx + 1] = [1] * (end_idx - start_idx)
                sillen = 1.0 * (end_sec - 1) / sr - 1.0 * start_sec / sr
            nframes = 0
        ii += win_int
    return sil_vec


# pitch, rmse, silence detector, mdp
def get_speech_features(audio_file, pitch_txt_name, energy_txt_name, silence_txt_name):

    y, sr = librosa.load(audio_file, sr=None)

    frame_length = int(0.08 * sr)  # 80ms
    hop_length = int(0.02 * sr)  # 20ms

    ## Pitch
    print("**************** COMPUTING PITCH VALUES **********************")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, sr=sr, frame_length=frame_length, hop_length=hop_length, fmin=60, fmax=2000
    )
    print("**************** PITCH VALUES COMPUTED **********************")
    ## RMS Energy
    print("**************** COMPUTING ENERGY VALUES **********************")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    print("**************** ENERGY VALUES COMPUTED **********************")
    ## Silence detector
    print("**************** COMPUTING BOOLEAN SILENCE VALUES **********************")
    sil = get_silence_waveform(y, rms[0], sr, 0.08, 0.02)
    print("**************** SILENCE VALUES COMPUTED **********************")

    print(
        "**************** LOGGING PITCH, ENERGY AND SILENCE VALUES IN TXT FILES **********************"
    )
    # print(len(f0), len(rms[0]), len(sil))
    f0 = [x if str(x) != "nan" else 0 for x in f0]
    f0_text = [str(x) + "\n" for x in f0]
    rms_text = [str(x) + "\n" for x in rms[0]]
    sil_text = [str(x) + "\n" for x in sil]
    with open(pitch_txt_name, "w") as f:
        f.writelines(f0_text)
    with open(energy_txt_name, "w") as f:
        f.writelines(rms_text)
    with open(silence_txt_name, "w") as f:
        f.writelines(sil_text)


def convert_vid_to_audio(input_vid_file, output_aud_file):
    print("**************** STARTING VIDEO TO AUDIO CONVERSION **********************")
    os.system("ffmpeg -i " + input_vid_file + " -ar 16000 -ac 1 " + output_aud_file)
    print("**************** VIDEO TO AUDIO CONVERSION DONE **********************")


if __name__ == "__main__":
    convert_vid_to_audio("good.mp4", "audio_good_1.wav")
    get_speech_features(
        "audio_good_1.wav", "pitch_good.txt", "energy_good.txt", "silence_good.txt"
    )
