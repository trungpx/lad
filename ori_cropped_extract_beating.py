import argparse
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import ipdb

from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import annotator, change_channel, gray_normalizer

def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model

def rescale(image):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image

def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of video
    :param preds:
    :param shapes:
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w

# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:
        # load best model
        model = load_model(sess, m_type, m_name, logger)
        # check input source is a file or camera
        if video_path == None:
            video_path = 0
        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []
        # initialize for up/down beating
        counter1 = 2; counter2 = 2; down = 0; up = 0;
        B = 0; C = 0;
        v_down = 0; v_up = 0; SPV = 0;
        count_up = 0; count_down = 0;
        cum_SPV = []
        # initialize for left/right beating
        counter1_hor = 2; counter2_hor = 2; right = 0; left = 0;
        B_hor = 0; C_hor = 0;
        v_left = 0; v_right = 0; SPV_hor = 0;
        count_left = 0; count_right = 0;
        cum_SPV_hor = []
        v_thres = 0.95
        frame_thres = 6
        r_thres = 15
        
        while ret:
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here
                frames.append(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_shape = frame.shape
                if frame.shape[0] != 192:
                    frame = rescale(frame)
                image = gray_normalizer(frame)
                image = change_channel(image, config["input_channel"])
                [p] = model.predict(sess, [image])
                x1, y1, w = upscale_preds(p, f_shape)
                preds.append([x1, y1, w])
                counter += 1
                i = counter-1; r = w
                labeled_img = annotator((0, 250, 0), frames[counter-1], *preds[counter-1])
                if i==0:
                    A = y1 # init A to the first value of gaze x
                    A_hor = x1 # init A to the first value of gaze y
                    video_size = frames[0].shape[0:2]
                    #video = cv2.VideoWriter("predicted_video_test.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (video_size[1], video_size[0]))
                if (preds[i][2]<r_thres):
                    labeled_img = cv2.putText(labeled_img,'Eye closed, frame[{}]'.format(i+1),(3,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                else:
                    labeled_img = cv2.putText(labeled_img,'Eye opened, frame[{}]'.format(i+1),(3,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                # start detecting the nystagmus down-beating, up-beating, left-beating, right-beating
                if i>=2:
                    if preds[i-2][2]>r_thres and preds[i-1][2]>r_thres and preds[i][2]>r_thres:
                        print('Frame[{}/{}], eye opened!'.format(i+1,total_frame))
                        # down-beating, up-beating
                        flag1 = np.sign(preds[i-1][1] - preds[i-2][1])
                        flag2 = np.sign(preds[i][1] - preds[i-1][1])
                        if flag1 > 0 and flag2 > 0:
                            counter1 += 1
                        elif flag1 < 0 and flag2 < 0:
                            counter2 += 1
                        elif flag1 > 0 and flag2 < 0:
                            B = preds[i-1][1]
                            counter2 = 2
                            v_up = (B-A)/counter1
                            if counter1 < frame_thres and v_up > v_thres: # show text in frames
                                count_down += 1
                                print('Down-beating, v_up = {:0.2f}, v_down = {:0.2f}'.format(v_up,v_down))
                        elif flag1 < 0 and flag2 > 0:
                            A = preds[i-1][1]
                            counter1 = 2
                            v_down = (B-A)/counter2
                            #if count_down > 0:
                                #SPV = np.maximum(v_up, v_down)
                                #cum_SPV.append(SPV)
                            if counter2 < frame_thres and v_down > v_thres: # show text in frames
                                count_up += 1
                                print('Up-beating, v_up = {:0.2f}, v_down = {:0.2f}'.format(v_up,v_down))
                        # left-beating, right-beating
                        flag1_hor = np.sign(preds[i-1][0] - preds[i-2][0])
                        flag2_hor = np.sign(preds[i][0] - preds[i-1][0])
                        if flag1_hor > 0 and flag2_hor > 0:
                            counter1_hor += 1
                        elif flag1_hor < 0 and flag2_hor < 0:
                            counter2_hor += 1
                        elif flag1_hor > 0 and flag2_hor < 0:
                            B_hor = preds[i-1][0]
                            counter2_hor = 2
                            v_right = (B_hor-A_hor)/counter1_hor
                            if counter1_hor < frame_thres and v_right > v_thres: # show text in frames
                                count_left += 1
                                print('Left-beating, v_right = {:0.2f}, v_left = {:0.2f}'.format(v_right,v_left))
                        elif flag1_hor < 0 and flag2_hor > 0:
                            A_hor = preds[i-1][0]
                            counter1_hor = 2
                            v_left = (B_hor-A_hor)/counter2_hor
                            #if count_left > 0:
                                #SPV_hor = np.maximum(v_left, v_right)
                                #cum_SPV_hor.append(SPV_hor)
                            if counter2_hor < frame_thres and v_left > v_thres: # show text in frames
                                count_right += 1
                                print('Right-beating, v_right = {:0.2f}, v_left = {:0.2f}'.format(v_right,v_left))
                    else:
                        print('3 sucessive frames have eye closed')
                        counter1 = 1; counter2 = 1; counter1_hor = 1; counter2_hor = 1;
                #visualize
                labeled_img = cv2.putText(labeled_img,'Down-beating: {}'.format(count_down),(3,25),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Up-beating: {}'.format(count_up),(3,40),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Right-beating: {}'.format(count_right),(3,55),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Left-beating: {}'.format(count_left),(3,70),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                cv2.imshow('labeled_img',labeled_img)
                cv2.waitKey(20)
                #cv2.destroyAllWindows()
                #video.write(labeled_img)
        #video.release()
        cap.release()
        toc = time.time()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))
        print('Down-beating: {}, Up-beating: {}, Left-beating: {}, Right-beating: {}'.format(count_down,count_up,count_left,count_right))
    ipdb.set_trace()
    
if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=class_)
    parser.add_argument('--model_type', help="INC, YOLO, simple", default="INC")
    parser.add_argument('--model_name', help="name of saved model (3A4Bh-Ref25)", default="3A4Bh-Ref25")
    parser.add_argument('video_path', help="path to video file, empty for camera")

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type
    video_path = args.video_path

    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    main(model_type, model_name, logger, video_path)
