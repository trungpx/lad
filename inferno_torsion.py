import argparse
import os
import time
import cv2
import numpy as np
import tensorflow as tf
# import ipdb

from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import annotator
import pickle

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
def main(write_output=True):
    with open('objs.pkl', 'rb') as f:
        frames, preds = pickle.load(f)
    # get the video size
    video_size = frames[0].shape[0:2]
        
    if write_output:
        # prepare a video write to show the result
        video = cv2.VideoWriter("predicted_video_torsion.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (video_size[1], video_size[0]))
        #video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (video_size[1], video_size[0]))

        # n = len(preds)
        x = np.array(preds)[:,1]
        y = np.array(preds)[:,0]
        r = np.array(preds)[:,2]
        
        #init A to the first value of gaze x
        counter1 = 2; counter2 = 2;
        A = x[0]; B = 0; C = 0
        v_down = 0; v_up = 0; SPV = 0
        count_up = 0; count_down = 0;
        cum_SPV = [];
        degree = [];
        x_temp_new = []; y_temp_new = [];
        flag = False;
        
        # change coordinate Oxy to O'xy (be careful with the way image calculate the position of O(0,0))
        x_pupil_new = np.array(preds)[:,0]
        y_pupil_new = np.array(preds)[:,1] - 120
        
        # init the region of template-matching
        img=frames[3690]; test=img; x1=int(y[3690]); y1=int(x[3690]); 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); template = gray_img[x1-int(r[3690])-5:x1-int(r[3690])+25, y1+int(r[3690])+11:y1+int(r[3690])+39]
        result = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED);
        loc = np.where(result == np.amax(result))
        # initial angle
        alpha_0 = np.arctan((loc[0]-120-y_pupil_new[3690])/(loc[1]-x_pupil_new[3690]))
        
        #ipdb.set_trace()
        
        # start read frame video
        for i, img in enumerate(frames):
            labeled_img = annotator((0, 250, 0), img, *preds[i])
            
            #ipdb.set_trace()
            # if the eyes are closed or opened
            if (preds[i][2]<14):
                labeled_img = cv2.putText(labeled_img,'Eye closed. Frame[{}]'.format(i+1),(3,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                x_temp_new.append(0)
                y_temp_new.append(0)
                degree.append(0)
            else:
                labeled_img = cv2.putText(labeled_img,'Eye opened. Frame[{}]'.format(i+1),(3,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                # draw marker/template
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
                result = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED);
                loc = np.where(result == np.amax(result))
                
                # change coordinate to O'xy
                x1 = loc[1]; y1 = loc[0]-120;
                # computer distance or norm of vector PI
                vector = np.array([y1-y_pupil_new[i], x1-x_pupil_new[i]])
                norm = np.linalg.norm(vector)
                if norm > 30: # check distance is suddenly increased
                    if flag==True and i > 1:
                        x_temp_new.append(x_temp_new[i-1])
                        y_temp_new.append(y_temp_new[i-1])
                    else:
                        x_temp_new.append(0)
                        y_temp_new.append(0)
                else:
                    x_temp_new.append(x1)
                    y_temp_new.append(y1)
                    flag = True
                if flag==True:
                    # visualize the rectangular of template
                    cv2.rectangle(labeled_img, (x_temp_new[i],y_temp_new[i]+120), (x_temp_new[i]+28, y_temp_new[i]+120+30), (255,0,0), 1)
                    # calculate torsional angle (radian)
                    alpha = np.arctan((y_temp_new[i] - y_pupil_new[i])/(x_temp_new[i] - x_pupil_new[i])) - alpha_0
                else:
                    alpha = 0
                degree.append(alpha)
            
            # start detecting the nystagmus down-beating, up-beating
            
            if i>=2:
                if r[i-2]>14 and r[i-1]>14 and r[i]>14:
                    print('Frame[{}], eye opened!'.format(i+1))
                    flag1 = np.sign(x[i-1] - x[i-2])
                    flag2 = np.sign(x[i] - x[i-1])
                    if flag1 > 0 and flag2 > 0:
                        counter1 += 1
                    elif flag1 < 0 and flag2 < 0:
                        counter2 += 1
                    elif flag1 > 0 and flag2 < 0:
                        B = x[i-1]
                        counter2 = 2
                        v_down = (B-A)/counter1
                        if counter1 < 4 and v_down > 0.8: # show text in frames
                            count_up += 1
                            print('Down-beating, v_down = {:0.2f}, SPV = {:0.2f}'.format(v_down,SPV))

                    elif flag1 < 0 and flag2 > 0:
                        A = x[i-1]
                        counter1 = 2
                        v_up = (B-A)/counter2
                        if count_up > 0:
                            SPV = np.maximum(v_down, v_up)
                            cum_SPV.append(SPV)
                        if counter2 < 4 and v_up> 0.8: # show text in frames
                            count_down += 1
                            print('Up-beating, v_up = {:0.2f}, SPV = {:0.2f}'.format(v_up,SPV))

                else:
                    print('3 sucessive frames have eye closed')

            labeled_img = cv2.putText(labeled_img,'Down-beating: {}'.format(count_up),(3,25),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
            labeled_img = cv2.putText(labeled_img,'Up-beating: {}'.format(count_down),(3,40),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)

            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
        np.savetxt('degree.txt',degree,fmt='%f')
        np.savetxt('degree_cartesian.txt',np.array(degree)*180/np.pi,fmt='%f')
    print("Done...")
    print('Down-beating: {} times, Up-beating: {} times'.format(count_down,count_up))


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=class_)

    parser.add_argument('--model_type', help="INC, YOLO, simple", default="INC")

    parser.add_argument('--model_name', help="name of saved model (3A4Bh-Ref25)",
                        default="3A4Bh-Ref25")

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
