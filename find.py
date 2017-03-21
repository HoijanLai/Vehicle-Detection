import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
import argparse
from itertools import product
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from classifier import *
from multiprocessing import Pool
from glob import glob


####################################################
#   Some pre-designed windows                      #
####################################################
infected_scales = [1]

def box_generator(yoi, xoi, yd_ratio = 0.17, scales = [1.2,1.8,1.6,1.8,2.0,3.0], xover = 0.9, window = 64):
    """
    a windows generator that gives a the searching strategy
    """
    width = xoi[1] - xoi[0]

    sizes = (np.array(scales)*window).astype(int)
    xsteps = (sizes*(1 - 1/np.array(scales)*xover)).astype(int)
    ydists = np.cumsum((yd_ratio*sizes).astype(int)) + yoi[0]
    begins = (width - sizes) % xsteps // 2
    nbs = (width - sizes) // xsteps + 1

    xleft = [xsteps[i]*np.arange(nbs[i]) + begins[i] for i in range(len(scales))]
    xright = [xleft[i] + sizes[i] for i in range(len(scales))]
    ytop = [np.ones(nbs[i], dtype = int)*ydists[i] for i in range(len(scales))]
    ybtm = [ytop[i] + sizes[i] for i in range(len(scales))]

    wins = [np.vstack([xleft[i], ytop[i], xright[i],ybtm[i]]).T.reshape([-1, 2, 2]) for i in range(len(scales))]

    wins_scaled = []
    for i in range(len(scales)):
        a = wins[i][:,0]/scales[i]
        wins_scaled.append(np.hstack([a, a + window]).reshape((-1, 2, 2)).astype(int))

    return scales, wins, wins_scaled

def infected_box(box, bias = 20, scales = infected_scales):
    """
    generate 8 neiggboring windows for a given window
    """
    side = box[1][1] - box[0][1]

    bs = [bias, -bias, 0]
    scheme = [np.repeat(list(product(bs, bs))[:-1],2,0).reshape(-1,2,2) for scl in scales]
    ss = [np.repeat([int(scl*side)], 8, 0) for scl in scales]
    diffs = [(s-side)//2 for s in ss]

    x_as = [window[0][0] - diff for diff in diffs]
    y_as = [window[0][1] - diff for diff in diffs]
    x_bs = [x_as[i] + s for i, s in enumerate(ss)]
    y_bs = [y_as[i] + s for i, s in enumerate(ss)]

    boxes = [np.vstack([x_as[i], y_as[i], x_bs[i], y_bs[i]]).T.reshape((-1,2,2)) + scheme[i]
            for i in range(len(scales))]
    boxes_scaled = [(wins[i]/scales[i]).astype(int) for i in range(len(scales))]
    return scales, boxes, boxes_scaled


def inscribed_square(box):
    """
    calculate the inscribed_square for a given box
    """
    x_a, y_a = box[0]
    x_b, y_b = box[1]
    w = x_b - x_a
    h = y_b - y_a
    if w == h:
        return box
    elif w > h:
        diff = (w - h)//2
        return ((x_a + diff, y_a), (x_b + diff, y_b))
    else:
        diff = (h - w)//2
        return ((x_a, y_a + diff), (x_b, y_b + diff))






class car_detector:
    def __init__(self, clf, X_scaler, feat_maker, cells_per_step = 1):
        """
        params:
            pos: a list of tuple, in format: (ystart, ystop, scale)
            clf: the trained classifier
            cells_per_step: sliding window step, measured in blocks actually
        """


        self.clf = clf
        self.scaler = X_scaler
        self.feat_maker = feat_maker
        self.cells_per_step = cells_per_step
        self.pool = Pool(4)
        self.pos = None
        self.box_info = None
        self.lazy_info = None
        self.cached_bboxes_1 = []
        self.cached_bboxes_2 = []

        self.cached_lazy = []

    # Define a single function that can extract features using hog sub-sampling and make predictions

    ## 1 ##
    # calculate the hog features for the region of interest defined by self.pos

    def set_pos(self, pos):
        self.pos = pos

    def _block_search(self, img):
        bboxes = []
        for p in self.pos:
            ystart, ystop, scale = p
            roi = img[ystart:ystop,:,:]
            roi = cv2.cvtColor(roi, CVT[cspace])

            # further cars are smaller
            if scale != 1:
                im_shape = roi.shape
                roi = cv2.resize(roi, (np.int(im_shape[1]/scale), np.int(im_shape[0]/scale)))

            ch_ids = hog_ch if hasattr(hog_ch, "__len__") else [hog_ch]
            rois = [roi[:,:,ch] for ch in ch_ids]
            hog_ims = self.pool.map(hog_feat, rois)

            # Define blocks and steps as above
            blocks_per_window = (window // ppc) - 1
            nyblocks, nxblocks = hog_ims[0].shape[:2]
            nxsteps = (nxblocks - blocks_per_window) // self.cells_per_step
            nysteps = (nyblocks - blocks_per_window) // self.cells_per_step

            #bboxes = self.cached_bboxes
            bboxes = []
            ypos = 0
            xpos = 0
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*self.cells_per_step
                    xpos = xb*self.cells_per_step
                    # sub-area for hog
                    y_end = ypos + blocks_per_window
                    x_end = xpos + blocks_per_window

                    xleft = xpos*ppc
                    ytop = ypos*ppc
                    hog_ravel = np.hstack([hg[ypos:y_end, xpos:x_end].ravel() for hg in hog_ims])
                    subimg = roi[ytop:ytop+window, xleft:xleft+window]
                    features = self.feat_maker.get_feat(subimg, hog_ravel, cvt = False)
                    test_features = self.scaler.transform(features.reshape((1, -1)))
                    test_pred= self.clf.predict(test_features)

                    if test_pred == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

        return bboxes



    ## 2 ##
    # implement a pre-designed window search based on the self.box_info


    def set_windows(self, boxes_info):
        self.box_info = boxes_info

    def _window_search(self, img, lazy = False):
        img = cv2.cvtColor(img, CVT[cspace])
        ch_ids = hog_ch if hasattr(hog_ch, "__len__") else [hog_ch]

        bboxes = []
        h, w = img.shape[0], img.shape[1]
        scls, windows, windows_scaled = self.box_info if not lazy else self.lazy_info
        for i, scl in enumerate(scls):
            resized_img = cv2.resize(img, (int(w/scl), int(h/scl)))
            for j, win_scaled in enumerate(windows_scaled[i]):
                y_a, x_a = win_scaled[0] # from drawing index to query index
                y_b, x_b = win_scaled[1]
                subimg = resized_img[x_a:x_b, y_a:y_b]
                feat = self.feat_maker.get_feat(subimg, cvt = False).reshape(1, -1)
                test_feat = self.scaler.transform(feat)
                test_pred = self.clf.predict(test_feat)


                if test_pred == 1:
                    bboxes.append(tuple(map(tuple, windows[i][j])))
        result_boxes = bboxes #+ self.cached_bboxes_1 + self.cached_bboxes_2
        #self.cached_bboxes_2 = self.cached_bboxes_1
        #self.cached_bboxes_1 = bboxes
        return result_boxes



    ## 3 ##
    # lazy version of window search, based on the infected_box boxes

    def set_lazy(self, boxes_info):
        self.lazy_info = boxes_info

    def _lazy_search(self, image):
        img = np.copy(image)
        if not self.cached_lazy:
            self.cached_lazy = self.window_search(img)
            return self.cached_lazy
        else:
            h, w = img.shape[0], img.shape[1]
            img = cv2.cvtColor(img,CVT[cspace])
            lazy_bboxes = self.window_search(img, lazy = True)
            lazy_bboxes = []
            for win in self.cached_lazy:
                scales, windows, windows_scaled = infected_box(win)
                for i, scl in enumerate(scales):
                    for j, win_scaled in enumerate(windows_scaled[i]):
                        y_a, x_a = win_scaled[0] # from drawing index to query index
                        y_b, x_b = win_scaled[1]
                        if y_a >= 0 and x_a >= 0 and y_b <= h - 1 and x_b <= w - 1:
                            subimg = cv2.resize(img[x_a:x_b, y_a:y_b], (window, window))
                            feat = self.feat_maker.get_feat(subimg, cvt = False).reshape(1, -1)
                            test_feat = self.scaler.transform(feat)
                            test_pred = self.clf.predict(test_feat)

                            if test_pred == 1:
                                lazy_bboxes.append(tuple(map(tuple, windows[i][j])))
            result_boxes = lazy_bboxes # + self.cached_bboxes_1 + self.cached_bboxes_2
            #self.cached_bboxes_2 = self.cached_bboxes_1
            #self.cached_bboxes_1 = lazy_bboxes
            return result_boxes


    ## 4 ##
    # the heatmap utility for calculating continuous boxes

    def _heatmap(self, shape, bboxes, thres):
        heat = np.zeros(shape)
        for bbox in bboxes:
            heat[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        return (heat > thres).astype(int)

    def _draw_car_box(self, img, heatmap):
        labels = label(heatmap)
        for car_num in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_num).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            #self.cached_bboxes_1.append(bbox)
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)
        return img


    ## integrated ##
    # the detecting functionalities integrating the functions above

    def _find(self, img, find_style):
        im = cv2.blur(img, (5,5))
        car_boxes = find_style(im)
        heat = self._heatmap(img.shape, car_boxes, 1)
        return self._draw_car_box(img, heat)

    def window_find(self, img):
        return self._find(img, self._window_search)

    def lazy_find(self, img):
        return self._find(img, self._lazy_search)

    def block_find(self, img):
        return self._find(img, self._block_search)


    def draw(self, img):
        """
        draw the detecting boxes from window_search
        """
        car_boxes = self.window_search(im)
        for car_box in car_boxes:
            cv2.rectangle(img, car_box[0], car_box[1], color = [0, 255, 0], thickness = 2)
        return img


positions = [(380, 700, 1.5), (380, 700, 1.5), (400, 700, 2.0)]
normal_boexs = box_generator((360, 700), (0, 1280))
lazy_boxes = box_generator((400, 700), (0, 1280), yd_ratio = 0.1, scales = [1.5,1.5,1.5,2.0])
out_path = './output_images'


def get_video(retrain, nb_per_class = 1000):
    if retrain:
        train_SVC(nb_per_class)
    with open('clf.pickle', 'rb') as f:
        clf, scaler, feat_maker = pickle.load(f)


    detector = car_detector(clf, scaler, feat_maker)
    detector.set_windows(normal_boexs)
    detector.set_lazy(lazy_boxes)
    detector.set_pos(positions)

    video = VideoFileClip('./project_video.mp4')
    video = video.fl_image(detector.window_find)
    video.write_videofile('test.mp4', audio = False)

def get_image_result(retrain, input_path, nb_per_class = 1000):

    if retrain:
        train_SVC(nb_per_class)
    with open('clf.pickle', 'rb') as f:
        clf, scaler, feat_maker = pickle.load(f)
    imgf_names = os.listdir(input_path)
    imgfs = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    imgs = [cv2.cvtColor(cv2.imread(imgf), cv2.COLOR_BGR2RGB) for imgf in imgfs]
    detector = car_detector(clf, scaler, feat_maker)
    detector.set_windows(normal_boexs)
    detector.set_lazy(lazy_boxes)
    detector.set_pos(positions)

    out_imgs = [detector.window_find(img) for img in imgs]
    for i in range(len(imgfs)):
        cv2.imwrite(os.path.join(out_path, imgf_names[i]), cv2.cvtColor(out_imgs[i], cv2.COLOR_RGB2BGR))




parser = argparse.ArgumentParser()
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--nbtrain', nargs = 1)
parser.add_argument('--video')
args = parser.parse_args()
try:
    nb_train = int(args.nbtrain[0])
except:
    nb_train = None
get_video(args.retrain, nb_train)
