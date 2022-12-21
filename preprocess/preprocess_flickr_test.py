import os
import pandas as pd
import soundfile as sf
import librosa
import cv2
import sys
from tqdm import tqdm

"""
Preprocess Flickr SoundNet testing set for Hear The Flow. 

 - Replace the two directories below to your system.
 
 - NOTE: This script works under the assumption that the Flickr SoundNet
    data has been downloaded into two separate folders -- train and test.
    This can be accomplished by splitting test_urls_public.txt into train
    and test portions and download each of them separately. 
"""

FLICKR_TEST_DATA = "/home/dennis/flickr-soundnet-dl/test_downloaded_v2/"
FLICKR_TEST_SAVEDIR = "/mnt/ssd/datasets/flickr-soundnet-replicate/test_flow_v4/"


if __name__ == "__main__":
    # Or use imagetoframe_expanded if building expanded flickr test set
    keyframes = pd.read_pickle(r'imagetoframe.pickle')

    onlythis = {
		5344317532: 74,
		10624261424: 514,
		12158276143: 375}

    for key, value in onlythis.items():
        keyframes[str(key)] = [value]
    
    # Testing Folder
    for root, dirs, files in os.walk(FLICKR_TEST_DATA + "audio/"):
        file_clean = [int(x.split(".")[0]) for x in files]

        for f in tqdm(files):

            filename = f.split(".")[0]

            if os.path.exists(f"{FLICKR_TEST_SAVEDIR}/frames/{filename}.jpg") and os.path.exists(f"{FLICKR_TEST_SAVEDIR}/audio/{filename}.wav"):
                print(f"{filename} already saved")
                pass
            else:
                aud, sr = librosa.load(os.path.join(root, f), sr=22050)

                # load middle section in dataloader for test
                # aud_middle_idx = aud.shape[0]//2
                # cut_aud = aud[max(0, int(aud_middle_idx - 1.5*22050)):int(aud_middle_idx + 1.5*22050)]

                vidobj = cv2.VideoCapture(os.path.join(root, f).replace("audio/", "video/").replace(".flac", ".mp4"))
                
                totalFrames = vidobj.get(cv2.CAP_PROP_FRAME_COUNT)

                print(totalFrames, filename)
                
                # get the frame that matched the original flickr test set
                fps = vidobj.get(cv2.CAP_PROP_FPS)
                if int(totalFrames) == keyframes[filename][0]:
                    print("found end of video")
                    frametouse = keyframes[filename][0] - 2
                else:
                    frametouse = keyframes[filename][0] - 1
                
                nextframe = frametouse + 1

                # Main image
                vidobj.set(cv2.CAP_PROP_POS_FRAMES, frametouse)
                success, image = vidobj.read()

                print(frametouse, image, filename)

                # Second image for flow
                vidobj.set(cv2.CAP_PROP_POS_FRAMES, nextframe)
                success, nextimage = vidobj.read()

                # Grayscale images for flow
                flow_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                flow_nextimage = cv2.cvtColor(nextimage,cv2.COLOR_BGR2GRAY)
                flowimg = cv2.calcOpticalFlowFarneback(flow_image, flow_nextimage, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flowimg_x = flowimg[:,:,0]
                flowimg_y = flowimg[:,:,1]
                
                cv2.imwrite(f"{FLICKR_TEST_SAVEDIR}frames/{filename}.jpg", image)
                sf.write(f"{FLICKR_TEST_SAVEDIR}audio/{filename}.wav", aud, sr)
                cv2.imwrite(f"{FLICKR_TEST_SAVEDIR}flow/flow_x/{filename}.jpg", flowimg_x)
                cv2.imwrite(f"{FLICKR_TEST_SAVEDIR}flow/flow_y/{filename}.jpg", flowimg_y)
