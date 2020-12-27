#!/usr/bin/env python3  
"""
-----⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Welcome to ArtLine : Image to Sketch using AI ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀-----
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀Assembled and trained by u/vijish_madhavan⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀https://github.com/vijishmadhavan                  ⠀⠀⠀⠀⠀⠀⠀⠀  ⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Implementaion  by u/3dsf⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀u_3dsf.reddit.com⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
Default behaviour is process images in the input folder to output folder⠀⠀⠀⠀⠀  
By using the -i flag, you can select a movie file or another folder⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
GPU (cuda) inference by default, with CPU fallback   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀See⠀⠀⠀⠀run.py -h⠀⠀⠀⠀for options   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
---------------------------------------------------------------------------⠀
"""

# adapted ArtLine for video input  https://github.com/vijishmadhavan/ArtLine
#     and https://github.com/kkroening/ffmpeg-python/blob/master/examples/tensorflow_stream.py

# sorry about the monolithic file (trying to keep it to one file for ease)

import os
import io
import subprocess
import glob
import argparse
import time
import logging as logger

from torchvision import transforms as T
from fastai.utils.mem import nn
from fastai.vision import open_image, load_learner, image, torch, image2np

import ffmpeg
import numpy as np
import cv2

#There is scaling warning that might come up, and this block supresses user warnings
#Comment out this block if your don't mind seeing the warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

#default model
MODEL_URL = "https://www.dropbox.com/s/p9lynpwygjmeed2/ArtLine_500.pkl"  # orig model url
MODEL_FILE_NAME = "ArtLine_1024.pkl"  #orig model

#------   this class is from the original project and is the main functional part

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

#------   these function are done by me   **NOTICE** they contain the default model info

def setupDirs(dir_list):
    for sDir in dir_list :
        os.makedirs(sDir, exist_ok=True)

def checkModelExists(MODEL_FILE_NAME):
    print()
    if os.path.isfile(
            os.path.join(os.path.dirname(
            os.path.abspath(__file__)), MODEL_FILE_NAME)):
        print ("MODEL : " + MODEL_FILE_NAME)
    else:
        print ("MODEL NOT FOUND : " + MODEL_FILE_NAME )
        print ("DOWNLOADING DEFAULT MODEL :\n" + MODEL_URL + "\n")
        os.chdir(os.path.dirname(subprocess.call("wget " + MODEL_URL, shell=True)))
    return MODEL_FILE_NAME

def modelDeviceLoadSelect():  # DETERMINE IF CUDA AVAILABLE and LOAD MODEL
    path_script = os.path.dirname(os.path.abspath(__file__))
    global COMPUTEdEVICE
    class Spam(int): pass
    try: 
        COMPUTEdEVICE = Spam(COMPUTEdEVICE)
    except:
        print()
    print (COMPUTEdEVICE, isinstance(COMPUTEdEVICE, int))
    if torch.cuda.is_available() and isinstance(COMPUTEdEVICE, int):
        def load_model():
            global USEgPU
            learn = load_learner(path_script, MODEL_FILE_NAME, device=COMPUTEdEVICE )
            USEgPU = True
            print("INFERENCE DEVICE : cuda")
            return learn
    else:
        def load_model():
            learn = load_learner(path_script, MODEL_FILE_NAME, device='cpu')
            print("INFERENCE DEVICE : cpu")
            return learn
    learn=load_model()
    return learn

def processFolder(input_folder, output_folder) :
    input_imgs = glob.glob(os.path.join(input_folder, "*"))
    count_imgs = len(input_imgs)
    if count_imgs == 0 :
        print("  No Images Found  in \"", input_folder, "\" folder")
        quit()
    for i, img in enumerate(input_imgs):
        timeMark = time.process_time()
        # Load image in fastai's framework
        p,img_hr,b = learn.predict(open_image(img))
        # Convert output tensor into np array
        im = image2np(img_hr)
        # alpha and beta control line output darkness 
        norm_image = cv2.normalize(im, None, alpha=-60, beta=260, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Save file
        cv2.imwrite('output/' + os.path.basename(img), norm_image)
        print("{} ({}/{})".format(img, i+1 , count_imgs) + " time : " + str(time.process_time()-timeMark))
    return 

#--------- this block of functions is mostly python-ffmpeg

# This is the function for processing VIDEO frames
def processFrame(frame) :
    global INCR, WIDTHoUT, HEIGHToUT
    ### Frame comes in as np array
    # jenky np array to buffer, because of poor cv2 support in fastai=1, or my poor skills
    is_success, buffer = cv2.imencode(".bmp", frame)  
    io_buf = io.BytesIO(buffer)
    # Load image in fastai's framework
    p,img_hr,b = learn.predict(open_image(io_buf))
    # Convert output tensor into np array
    im = image2np(img_hr)
    # alpha and beta control line output darkness 
    norm_image = cv2.normalize(im, None, alpha=-60, beta=260, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) #16S)
    INCR += 1
    # enabling this line will also output images when processing videos
    #cv2.imwrite('output/' + str(INCR) + ".png", norm_image)  # INCR is just a frame counter
    return norm_image 

def getVideoMetaData(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    print(video_info)  #see what available
    width = int(video_info['width'])
    height = int(video_info['height'])
    fps = video_info['r_frame_rate']
    try:
        bits_in = video_info['bits_per_raw_sample']  # h264_cuvid complains with 10 bit
    except:
        bits_in = 10
    print (bits_in)
    time.sleep(5)
    #total_frames = int(video_info['nb_frames'])
    return width, height, fps, bits_in

def checkForAudio(in_filename):
    streams = ffmpeg.probe(in_filename)["streams"]
    for stream in streams:
        if stream["codec_type"] == "audio":
            return True
    return False

def readFrameAsNp(ffmpegDecode, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = ffmpegDecode.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def writeFrameAsByte(ffmpegEncode, frame):
    logger.debug('Writing frame')
    ffmpegEncode.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def aFrame():
    global WIDTHoUT
    global HEIGHToUT
    width, height, z, zz = getVideoMetaData(input_path)
    img = np.zeros([height,width,3],dtype=np.uint8)
    img.fill(255)
    is_success, buffer = cv2.imencode(".bmp", img)
    io_buf = io.BytesIO(buffer)
    # Load image in fastai's framework
    p,img_hr,b = learn.predict(open_image(io_buf))
    # Convert output tensor into np array
    im = image2np(img_hr)
    x = im.shape
    WIDTHoUT = x[1]
    HEIGHToUT = x[0]
    print(WIDTHoUT, HEIGHToUT)

def vid2np(in_filename, bits_in):
    logger.info('vid2np() -- Decoding to pipe')
    #print(int(bits_in), USEgPU)
    if int(bits_in) == 8 and USEgPU :
        args = (
            ffmpeg
            .input(in_filename,
                hwaccel_device=COMPUTEdEVICE,
                hwaccel='cuvid', **{'c:v': 'h264_cuvid'}, 
                hwaccel_output_format='rawvideo')
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args("-hide_banner")
            .compile()
        )
    else:
        args = (
            ffmpeg
            .input(in_filename,
                **{'c:v': 'h264'})
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args("-hide_banner")
            .compile()
        )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def np2vid(out_filename, fps_out, in_file):
    logger.info('np2vid() encoding from pipe')
    codec = 'h264_nvenc' if USEgPU else 'h264'
    if checkForAudio(in_file) :
        pipeline2 = ffmpeg.input(in_file)
        audio = pipeline2.audio
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                s='{}x{}'.format(WIDTHoUT, HEIGHToUT),
                framerate=fps_out )
            .output(audio, out_filename , pix_fmt='yuv420p', **{'c:v': codec}, 
                shortest=None, acodec='copy')
            .global_args("-hide_banner")
            .overwrite_output()
            .compile()
        )
    else:
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', 
                s='{}x{}'.format(WIDTHoUT, HEIGHToUT), 
                framerate=fps_out )
            .output(audio, out_filename , pix_fmt='yuv420p', **{'c:v': codec})
            .global_args("-hide_banner")
            .overwrite_output()
            .compile()
        )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def run(in_file, out_file, process_frame, width_out=0, height_out=0):
    width, height, fps_out, bits_in = getVideoMetaData(in_file)
    ffmpegDecode = vid2np(in_file, bits_in)
    ffmpegEncode = np2vid(out_file, fps_out, in_file)
    while True:
        in_frame = readFrameAsNp(ffmpegDecode, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        logger.debug('Processing frame')
        out_frame = processFrame(in_frame)
        writeFrameAsByte(ffmpegEncode, out_frame)

    logger.info('Waiting for ffmpegDecode')
    ffmpegDecode.wait()

    logger.info('Waiting for ffmpegEncode')
    ffmpegEncode.stdin.close()
    ffmpegEncode.wait()

    logger.info('Done')


### -----------   

def input_parser() :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', action="store", dest="source_input", default="input",
            help="This can be either a movie file or a folder -- default= input (folder)")
    parser.add_argument('-o', action="store", dest="source_output", default="output",
            help="Should be of same type as input: movie->movie or img->img -- default= output (folder)")
    # selects GPU by ID (int), or CPU by any non-int
    parser.add_argument('-c', action="store", dest="compute_device", default=0,
            help="Input your prefered GPU by ID (int) or type \'cpu\' for cpu inference")
    results = parser.parse_args()
    return results 

if __name__ == '__main__':
    INCR = 0 
    WIDTHoUT = 1024
    HEIGHToUT = 820
    USEgPU = False  # Can turn on in modelDeviceLoadSelect()

    setupDirs(["input","output"])

    iParser = input_parser()
    checkModelExists(MODEL_FILE_NAME)
    COMPUTEdEVICE = iParser.compute_device
    learn = modelDeviceLoadSelect()

    input_path = iParser.source_input
    output_path = iParser.source_output
    if os.path.isdir(input_path):
        print("INPUT DIRECTORY : ", input_path, "\n")
        processFolder(input_path, output_path)
    elif os.path.isfile(input_path):
        #run 1 frame
        aFrame()
        print("INPUT FILE : ", input_path)
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "out.ArtLine." 
                    + time.strftime("%YY%m%d-%H.%M.%S") + ".mp4")
        run(input_path, output_path, processFrame)
    else:
        print("DIRECTORY / FILE NOT FOUND : ", input_path)
    print()


