import argparse
import os, glob, pickle
import time
import cv2
import numpy as np
import tensorflow as tf
import ipdb
import math
from tqdm import tqdm

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

def detect_beat(model,sess,video_path=None,coordinates=None):
    # load the video or camera
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret = True
    counter = 0
    tic = time.time()
    frames = []
    preds = []
    # initialize for up/down beating
    counter1 = 1; counter2 = 1; down = 0; up = 0
    B = 0; C = 0
    v_down = 0; v_up = 0; SPV = 0
    count_up = 0; count_down = 0
    cum_SPV = []
    # initialize for left/right beating
    counter1_hor = 1; counter2_hor = 1; right = 0; left = 0
    B_hor = 0; C_hor = 0
    v_left = 0; v_right = 0; SPV_hor = 0
    count_left = 0; count_right = 0
    cum_SPV_hor = []
    v_thres = 0.80
    frame_thres = 5
    r_thres = 18 #15
    abs_r_thres = 4.0 # check difference of 2 frames
    v_max = 10.0
    norm_thres = 4.0
    all_down,all_up,all_right,all_left = [],[],[],[]
    beat_frame = []
    frame_diff = 8
    
    while ret:
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            frame = frame[coordinates[0]:coordinates[1],coordinates[2]:coordinates[3],:] #coordinates=2:120,0:160
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
            #labeled_img = annotator((0, 250, 0), frames[counter-1], *preds[counter-1])
            if i==0:
                A = y1 # init A to the first value of gaze x
                A_hor = x1 # init A to the first value of gaze y
            # start detecting the nystagmus down-beating, up-beating, left-beating, right-beating
            if i>=2:
                if preds[i-2][2]>r_thres and preds[i-1][2]>r_thres and preds[i][2]>r_thres:
                    print('Frame[{}/{}], eye opened!'.format(i+1,total_frame))
                    beat_tmp = [i,0,0,0,0] #[frame_id,down,up,left,right] velocity
                    """
                    norm_move = math.sqrt((preds[i-1][0]-x1)**2+(preds[i-1][1]-y1)**2)
                    print('Norm: {}'.format(norm_move))
                    if norm_move>norm_thres:
                        continue
                    """
                    # down-beating, up-beating
                    flag1 = np.sign(preds[i-1][1] - preds[i-2][1])
                    flag2 = np.sign(preds[i][1] - preds[i-1][1])
                    if flag1 > 0 and flag2 > 0:
                        counter1 += 1
                    elif flag1 < 0 and flag2 < 0:
                        counter2 += 1
                    elif flag1 > 0 and flag2 < 0:
                        B = preds[i-1][1]
                        counter2 = 1
                        v_up = (B-A)/counter1
                        if counter1<frame_thres and v_up>v_thres and v_up<v_max: # show text in frames
                            diff = np.abs(preds[i-1][2]-preds[i][2])
                            diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                            if(diff < abs_r_thres/4 and v_up>v_down and diff1 < abs_r_thres/4):
                                all_down.append(v_up)
                                count_down += 1
                                beat_tmp[1] = v_up
                                print('Down-beating, v_up = {:0.2f}, v_down = {:0.2f}, r_diff = {:0.2f}'.format(v_up,v_down,diff))
                            counter1 = 1
                    elif flag1 < 0 and flag2 > 0:
                        A = preds[i-1][1]
                        counter1 = 1
                        v_down = (B-A)/counter2
                        if counter2<frame_thres and v_down>v_thres and v_down<v_max: # show text in frames
                            #diff = np.abs(preds[i-counter2+1][2]-preds[i][2])
                            diff = np.abs(preds[i-1][2]-preds[i][2])
                            diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                            if(diff < abs_r_thres/4 and v_down>v_up and diff1 < abs_r_thres/4):
                                all_up.append(v_down)
                                count_up += 1
                                beat_tmp[2] = v_down
                                print('Up-beating, v_up = {:0.2f}, v_down = {:0.2f}, r_diff = {:0.2f}'.format(v_up,v_down,diff))
                            counter2 = 1
                    # left-beating, right-beating
                    flag1_hor = np.sign(preds[i-1][0] - preds[i-2][0])
                    flag2_hor = np.sign(preds[i][0] - preds[i-1][0])
                    if flag1_hor > 0 and flag2_hor > 0:
                        counter1_hor += 1
                    elif flag1_hor < 0 and flag2_hor < 0:
                        counter2_hor += 1
                    elif flag1_hor > 0 and flag2_hor < 0:
                        B_hor = preds[i-1][0]
                        counter2_hor = 1
                        v_right = (B_hor-A_hor)/counter1_hor
                        if counter1_hor<frame_thres and v_right>v_thres and v_right<v_max: # show text in frames
                            #diff = np.abs(preds[i-counter1_hor+1][2]-preds[i][2])
                            diff = np.abs(preds[i-1][2]-preds[i][2])
                            diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                            if(diff < abs_r_thres and v_right>v_left and diff1 < abs_r_thres):
                                all_left.append(v_right)
                                count_left += 1
                                beat_tmp[3] = v_right
                                print('Left-beating, v_right = {:0.2f}, v_left = {:0.2f}, r_diff = {:0.2f}'.format(v_right,v_left,diff))
                            counter1_hor = 1
                    elif flag1_hor < 0 and flag2_hor > 0:
                        A_hor = preds[i-1][0]
                        counter1_hor = 1
                        v_left = (B_hor-A_hor)/counter2_hor
                        if counter2_hor<frame_thres and v_left>v_thres and v_left<v_max: # show text in frames
                            diff = np.abs(preds[i-1][2]-preds[i][2])
                            diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                            if(diff < abs_r_thres and v_left>v_right and diff1 < abs_r_thres):
                                all_right.append(v_left)
                                count_right += 1
                                beat_tmp[4] = v_left
                                print('Right-beating, v_right = {:0.2f}, v_left = {:0.2f}, r_diff = {:0.2f}'.format(v_right,v_left,diff))
                            counter2_hor = 1
                    if np.sum(beat_tmp[1:])>0: # only save the frame that has at least 1 beating detected
                        beat_frame.append(beat_tmp)
                else:
                    print('3 sucessive frames have eye closed')
                    counter1 = 1; counter2 = 1; counter1_hor = 1; counter2_hor = 1;
    toc = time.time()
    cap.release()
    print("{0:0.2f} FPS".format(counter / (toc - tic)))
    all_down = 0 if all_down==[] else all_down; all_up = 0 if all_up==[] else all_up; all_right = 0 if all_right==[] else all_right; all_left = 0 if all_left==[] else all_left
    print('Down-beating: {}, Up-beating: {}, Left-beating: {}, Right-beating: {}'.format(count_down,count_up,count_left,count_right))
    print('avg_down: {}, avg_up: {}, avg_right: {}, avg_left: {}'.format(np.average(all_down),np.average(all_up),np.average(all_right),np.average(all_left)))
    print('max_down: {}, max_up: {}, max_right: {}, max_left: {}'.format(np.max(all_down),np.max(all_up),np.max(all_right),np.max(all_left)))

    return beat_frame

# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main(m_type, m_name, logger, video_path=None, video_range=None):
    
    with tf.Session() as sess:
        #ipdb.set_trace()
        # load best model
        left_eyes = [2,120,160,320]
        right_eyes = [2,120,0,160]
        
        model = load_model(sess, m_type, m_name, logger)
        if video_path != None:
            beat_left = detect_beat(model,sess,video_path,left_eyes)
            beat_right = detect_beat(model,sess,video_path,right_eyes)
        else:
            data_path = '../revised-data/corrected-label-2' #../revised-data/train (body+yolo)
            beat_path = 'beat_feat'
            avi_list = sorted(glob.glob(os.path.join(data_path, '*.avi')))
            for file_id in tqdm(range(video_range[0], video_range[1])):
                t0 = time.time()
                video_path = avi_list[file_id]
                base_name = os.path.basename(video_path).split('.')[0]
                filename = os.path.join(beat_path,base_name+'.pickle')
                print("CURRENT: %d, name=%s" % (file_id, video_path))
                #ipdb.set_trace()
                if os.path.exists(filename):
                    continue
                print('Detect left_eye...')
                beat_left = detect_beat(model,sess,video_path,left_eyes)
                print('Detect right_eye...')
                beat_right = detect_beat(model,sess,video_path,right_eyes)
                f = open(filename, 'wb'); pickle.dump([beat_left,beat_right], f); f.close()
                print("Elapsed: %.2fs"%(time.time()-t0))
                #ipdb.set_trace()
        ipdb.set_trace()
    
    '''
    with tf.Session() as sess:
        # load best model
        model = load_model(sess, m_type, m_name, logger)
        
        # check input source is a file or camera
        if video_path == None:
            video_path = 0
        
        #if video_path==None:
            #video_folder = '../revised-data/train (body+yolo)'
        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []
        # initialize for up/down beating
        counter1 = 1; counter2 = 1; down = 0; up = 0
        B = 0; C = 0;
        v_down = 0; v_up = 0; SPV = 0
        count_up = 0; count_down = 0
        cum_SPV = []
        # initialize for left/right beating
        counter1_hor = 1; counter2_hor = 1; right = 0; left = 0
        B_hor = 0; C_hor = 0;
        v_left = 0; v_right = 0; SPV_hor = 0;
        count_left = 0; count_right = 0;
        cum_SPV_hor = []
        v_thres = 0.80
        frame_thres = 5
        r_thres = 18
        abs_r_thres = 4.0 # check difference of 2 frames
        v_max = 10.0 #8
        norm_thres = 4.0
        all_down,all_up,all_right,all_left = [],[],[],[]
        beat_frame = []
        frame_diff = 8
        
        while ret:
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here
                frame = frame[2:120,0:160,:]
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
                if i==2: #2
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
                        print('Frame[{}/{}], eye opened!, r={}, x={}'.format(i+1,total_frame,r,preds[i][0]))
                        beat_tmp = [i,0,0,0,0] #[frame_id,down,up,left,right] velocity
                        """
                        norm_move = math.sqrt((preds[i-1][0]-x1)**2+(preds[i-1][1]-y1)**2)
                        print('Norm: {}'.format(norm_move))
                        if norm_move>norm_thres:
                            continue
                        """
                        # down-beating, up-beating
                        flag1 = np.sign(preds[i-1][1] - preds[i-2][1])
                        flag2 = np.sign(preds[i][1] - preds[i-1][1])
                        if flag1 > 0 and flag2 > 0:
                            counter1 += 1
                        elif flag1 < 0 and flag2 < 0:
                            counter2 += 1
                        elif flag1 > 0 and flag2 < 0:
                            B = preds[i-1][1]
                            counter2 = 1
                            v_up = (B-A)/counter1
                            if counter1<frame_thres and v_up>v_thres and v_up<v_max: # show text in frames
                                #diff = np.abs(preds[i-counter1+1][2]-preds[i][2])
                                diff = np.abs(preds[i-1][2]-preds[i][2])
                                diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                                if(diff < abs_r_thres/4 and v_up>v_down and diff1 < abs_r_thres/4):
                                    all_down.append(v_up)
                                    count_down += 1
                                    beat_tmp[1] = v_up
                                    print('Down-beating, v_up = {:0.2f}, v_down = {:0.2f}, r_diff = {:0.2f}'.format(v_up,v_down,diff))
                                counter1 = 1
                        elif flag1 < 0 and flag2 > 0:
                            A = preds[i-1][1]
                            counter1 = 1
                            v_down = (B-A)/counter2
                            if counter2<frame_thres and v_down>v_thres and v_down<v_max: # show text in frames
                                #diff = np.abs(preds[i-counter2+1][2]-preds[i][2])
                                diff = np.abs(preds[i-1][2]-preds[i][2])
                                diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                                if(diff < abs_r_thres/4 and v_down>v_up and diff1 < abs_r_thres/4):
                                    all_up.append(v_down)
                                    count_up += 1
                                    beat_tmp[2] = v_down
                                    print('Up-beating, v_up = {:0.2f}, v_down = {:0.2f}, r_diff = {:0.2f}'.format(v_up,v_down,diff))
                                counter2 = 1
                        # left-beating, right-beating
                        flag1_hor = np.sign(preds[i-1][0] - preds[i-2][0])
                        flag2_hor = np.sign(preds[i][0] - preds[i-1][0])
                        if flag1_hor > 0 and flag2_hor > 0:
                            counter1_hor += 1
                        elif flag1_hor < 0 and flag2_hor < 0:
                            counter2_hor += 1
                        elif flag1_hor > 0 and flag2_hor < 0:
                            #print('left_detected, {},{},{}, c1={}'.format(preds[i-2][0],preds[i-1][0],preds[i][0],counter1_hor))
                            B_hor = preds[i-1][0]
                            counter2_hor = 1
                            v_right = (B_hor-A_hor)/counter1_hor
                            #print('v_right:{}, v_thres:{}, v_max:{}'.format(v_right,v_thres,v_max))
                            if counter1_hor<frame_thres and v_right>v_thres and v_right<v_max: # show text in frames
                                #diff = np.abs(preds[i-counter1_hor+1][2]-preds[i][2])
                                diff = np.abs(preds[i-1][2]-preds[i][2])
                                diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                                #print(diff,diff1)
                                if(diff < abs_r_thres and v_right>v_left and diff1 < abs_r_thres):
                                #if(diff < abs_r_thres and diff1 < abs_r_thres):
                                    all_left.append(v_right)
                                    count_left += 1
                                    beat_tmp[3] = v_right
                                    print('Left-beating, v_right = {:0.2f}, v_left = {:0.2f}, r_diff = {:0.2f}'.format(v_right,v_left,diff))
                                counter1_hor = 1
                        elif flag1_hor < 0 and flag2_hor > 0:
                            #print('right_detected, {},{},{}, c2={}'.format(preds[i-2][0],preds[i-1][0],preds[i][0],counter2_hor))
                            A_hor = preds[i-1][0]
                            counter1_hor = 1
                            v_left = (B_hor-A_hor)/counter2_hor
                            #print('v_left:{}, v_thres:{}, v_max:{}, c2:{}'.format(v_left,v_thres,v_max,counter2_hor))
                            if counter2_hor<frame_thres and v_left>v_thres and v_left<v_max: # show text in frames
                                #diff = np.abs(preds[i-counter2_hor+1][2]-preds[i][2])
                                diff = np.abs(preds[i-1][2]-preds[i][2])
                                diff1 = np.abs(preds[i-2][2]-preds[i-1][2])
                                #print(diff,diff1)
                                if(diff < abs_r_thres and v_left>v_right and diff1 < abs_r_thres):
                                #if(diff < abs_r_thres and diff1 < abs_r_thres):
                                    all_right.append(v_left)
                                    count_right += 1
                                    beat_tmp[4] = v_left
                                    print('Right-beating, v_right = {:0.2f}, v_left = {:0.2f}, r_diff = {:0.2f}'.format(v_right,v_left,diff))
                                counter2_hor = 1
                        if np.sum(beat_tmp[1:])>0: # only save the frame that has at least 1 beating detected
                            beat_frame.append(beat_tmp)
                    else:
                        print('3 sucessive frames have eye closed,r={}'.format(r))
                        counter1 = 1; counter2 = 1; counter1_hor = 1; counter2_hor = 1
                #visualize
                
                labeled_img = cv2.putText(labeled_img,'Down-beating: {}'.format(count_down),(3,25),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Up-beating: {}'.format(count_up),(3,40),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Right-beating: {}'.format(count_right),(3,55),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                labeled_img = cv2.putText(labeled_img,'Left-beating: {}'.format(count_left),(3,70),cv2.FONT_HERSHEY_SIMPLEX,.3,(0, 250, 0),1)
                cv2.imshow('labeled_img',labeled_img)
                cv2.waitKey(10)
                #ipdb.set_trace()
                #cv2.destroyAllWindows()
                #video.write(labeled_img)
        #video.release()
        toc = time.time()
        cap.release()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))
        all_down = 0 if all_down==[] else all_down; all_up = 0 if all_up==[] else all_up; all_right = 0 if all_right==[] else all_right; all_left = 0 if all_left==[] else all_left
        print('Down-beating: {}, Up-beating: {}, Left-beating: {}, Right-beating: {}'.format(count_down,count_up,count_left,count_right))
        print('avg_down: {}, avg_up: {}, avg_right: {}, avg_left: {}'.format(np.average(all_down),np.average(all_up),np.average(all_right),np.average(all_left)))
        print('max_down: {}, max_up: {}, max_right: {}, max_left: {}'.format(np.max(all_down),np.max(all_up),np.max(all_right),np.max(all_left)))
    
    ipdb.set_trace()
    '''
    
if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=class_)
    parser.add_argument('--model_type', help="INC, YOLO, simple", default="INC")
    parser.add_argument('--model_name', help="name of saved model (3A4Bh-Ref25)", default="3A4Bh-Ref25")
    parser.add_argument('--video_path', help="path to video file, empty for camera", default=None)
    parser.add_argument('--range', type=int, default=[0,1000], nargs='+')

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type
    video_path = args.video_path
    video_range = args.range
    
    #ipdb.set_trace()
    
    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    # run for 1 video
    # CUDA_VISIBLE_DEVICES=0 python extract_beating.py --video_path='Class2_000086.avi'
    # run for extracting multiple videos --> predefine folder
    # CUDA_VISIBLE_DEVICES=0 python extract_beating.py
    main(model_type, model_name, logger, video_path, video_range)
