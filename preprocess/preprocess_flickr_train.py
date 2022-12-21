import os
import pandas as pd
import soundfile as sf
import librosa
import cv2
import sys
from tqdm import tqdm

"""
Preprocess Flickr SoundNet training set for Hear The Flow. 

 - Replace the two directories below to your system.
 
 - NOTE: This script processes samples against the testing sets, i.e.
    preprocessed data will not contain testing samples.
"""

FLICKR_DATA = "/mnt/ssd/datasets/flickr-soundnet/data/"
FLICKR_TRAIN_SAVEDIR = "/mnt/ssd/datasets/flickr-soundnet-replicate/train/"


if __name__ == "__main__":

    test_files = pd.read_csv("metadata/flickr_test_trimmed.csv", header=None)[0].tolist()
    test_files_expanded = pd.read_csv("metadata/flickr_test_expanded.csv", header=None)[0].tolist()

    test_files = test_files + test_files_expanded

    print(test_files, len(test_files))

    # Training Folder
    for root, dirs, files in os.walk(FLICKR_DATA + "audio/"):
        file_clean = [int(x.split(".")[0]) for x in files]

        for f in tqdm(files):

            filename = f.split(".")[0]

            if filename not in test_files:
                if os.path.exists(f"{FLICKR_TRAIN_SAVEDIR}/frames/{filename}.jpg") and os.path.exists(f"{FLICKR_TRAIN_SAVEDIR}/audio/{filename}.wav"):
                    print(f"{filename} already saved")
                    pass
                else:
                    aud, sr = librosa.load(os.path.join(root, f), sr=22050)

                    aud_middle_idx = aud.shape[0]//2
                    cut_aud = aud[max(0, int(aud_middle_idx - 1.5*22050)):int(aud_middle_idx + 1.5*22050)]

                    vidobj = cv2.VideoCapture(os.path.join(root, f).replace("audio/", "video/").replace(".flac", ".mp4"))
                    
                    totalFrames = vidobj.get(cv2.CAP_PROP_FRAME_COUNT)
                    
                    fps = vidobj.get(cv2.CAP_PROP_FPS)
                    middleframe = totalFrames // 2
                    nextframe = middleframe + 1

                    # Main image
                    vidobj.set(cv2.CAP_PROP_POS_FRAMES, middleframe)
                    success, image = vidobj.read()

                    # Second image for flow
                    vidobj.set(cv2.CAP_PROP_POS_FRAMES, nextframe)
                    success, nextimage = vidobj.read()

                    # Grayscale images for flow
                    flow_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    flow_nextimage = cv2.cvtColor(nextimage,cv2.COLOR_BGR2GRAY)
                    flowimg = cv2.calcOpticalFlowFarneback(flow_image,flow_nextimage, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flowimg_x = flowimg[:,:,0]
                    flowimg_y = flowimg[:,:,1]
                    
                    cv2.imwrite(f"{FLICKR_TRAIN_SAVEDIR}frames/{filename}.jpg", image)
                    sf.write(f"{FLICKR_TRAIN_SAVEDIR}audio/{filename}.wav", cut_aud, sr)
                    cv2.imwrite(f"{FLICKR_TRAIN_SAVEDIR}flow/flow_x/{filename}.jpg", flowimg_x)
                    cv2.imwrite(f"{FLICKR_TRAIN_SAVEDIR}flow/flow_y/{filename}.jpg", flowimg_y)
            else:
                print("Found test sample; Skipping.")


   