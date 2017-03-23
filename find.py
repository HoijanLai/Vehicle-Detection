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
#   Some pre-designed bboxes utils                 #
####################################################

heat_thres = 2 # the detector believe an area where at least two boxes overlap.
box_scales = [1.2,3.2,1.5,1.8,2.5,4.2]
yd_ratio = 0.2
xover = 0.9

lazy_scales = [1.2,  2.5]
lazy_xover = 0.3
lazy_yd = 0.6

def box_generator(yoi, xoi, yd_ratio = yd_ratio, scales = box_scales, xover = xover):

    """
    a windows generator that gives a the searching strategy

    params:
        yoi, xoi: both are tuple, determine the region of interest
        yd_ratio: float: how the boxes bias vertically
        scales: list of float, different sizes of windowss
        xover: float, overlap rate horizontally

    return:
        scales: the scales
        boxes: the boxes in real sizes
        boxes_scaled: window*window scaled version of boxes
    """
    width = xoi[1] - xoi[0]

    sizes = (np.array(scales)*window).astype(int)
    xsteps = (sizes*(1 - 1/np.array(scales)*xover)).astype(int)
    ydists = (yd_ratio*sizes).astype(int) + yoi[0]
    begins = (width - sizes) % xsteps // 2
    nbs = (width - sizes) // xsteps + 1

    xleft = [xsteps[i]*np.arange(nbs[i]) + begins[i] for i in range(len(scales))]
    xright = [xleft[i] + sizes[i] for i in range(len(scales))]
    ytop = [np.ones(nbs[i], dtype = int)*ydists[i] for i in range(len(scales))]
    ybtm = [ytop[i] + sizes[i] for i in range(len(scales))]

    boxes = [np.vstack([xleft[i], ytop[i], xright[i],ybtm[i]]).T.reshape([-1, 2, 2]) for i in range(len(scales))]

    boxes_scaled = []
    for i in range(len(scales)):
        a = boxes[i][:,0]/scales[i]
        boxes_scaled.append(np.hstack([a, a + window]).reshape((-1, 2, 2)).astype(int))

    return scales, boxes, boxes_scaled





def infected_box(box, bias_x = 0.5, bias_y = 0.15, nb = 3):
    """
    generate 8 neiggboring windows for a given box

    params:
        box: opencv standard box
        bias: int
        scales: the neighbors can vary in size

    return:
        a ndarray, boxes
    """
    side_x = box[1][0] - box[0][0]
    side_y = box[1][1] - box[0][1]
    bx = int(bias_x*side_x)
    by = int(bias_y*side_y)

    scheme_x = np.random.randint(-bx, bx, (nb, 2, 1))
    scheme_y = np.random.randint(-by, by, (nb, 2, 1))
    return box + np.dstack((scheme_x, scheme_y))




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



    ## 1. BLOCK SEARCH ##

    # calculate the hog features
    # for the region of interest
    # defined by self.pos

    def set_pos(self, pos):
        self.pos = pos

    def _block_search(self, img):
        """
        calculate the hog blocks and implemet exhausted search

        # TODO: make it possible for pre design windows search
        """
        bboxes = []
        for p in self.pos: # iterate through all the positions
            ystart, ystop, scale = p
            roi = img[ystart:ystop,:,:]

            # further cars are smaller
            if scale != 1:
                im_shape = roi.shape
                roi = cv2.resize(roi, (np.int(im_shape[1]/scale), np.int(im_shape[0]/scale)))

            # we only calculate the hog map once
            hog_im = hog_feat(roi)

            # Define blocks and steps as above
            blocks_per_window = (window // ppc) - 1
            nyblocks, nxblocks = hog_im.shape[:2]
            nxsteps = (nxblocks - blocks_per_window) // self.cells_per_step
            nysteps = (nyblocks - blocks_per_window) // self.cells_per_step


            bboxes = []
            ypos = 0
            xpos = 0
            roi = cv2.cvtColor(roi, CVT[cspace])
            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*self.cells_per_step
                    xpos = xb*self.cells_per_step

                    # sub-area for hog map
                    y_end = ypos + blocks_per_window
                    x_end = xpos + blocks_per_window

                    xleft = xpos*ppc
                    ytop = ypos*ppc
                    hog_ravel = hog_im[ypos:y_end, xpos:x_end].ravel()

                    # the actual subimg
                    subimg = roi[ytop:ytop+window, xleft:xleft+window]
                    features = self.feat_maker.get_feat(subimg, hog_ravel, cvt = False)
                    test_features = self.scaler.transform(features.reshape((1, -1)))
                    test_pred= self.clf.predict(test_features)


                    if test_pred == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

        result_boxes = bboxes + self.cached_bboxes_1 + self.cached_bboxes_2
        self.cached_bboxes_2 = self.cached_bboxes_1
        self.cached_bboxes_1 = bboxes
        return result_boxes








    ## 2. WINDOW SEARCH ##

    # implement a pre-designed window
    # search based on the self.box_info


    def set_windows(self, boxes_info):
        self.box_info = boxes_info

    def _window_search(self, img, lazy = False):
        bboxes = []
        h, w = img.shape[0], img.shape[1]
        scls, windows, windows_scaled = self.box_info if not lazy else self.lazy_info
        for i, scl in enumerate(scls):
            resized_img = cv2.resize(img, (int(w/scl), int(h/scl)))
            for j, win_scaled in enumerate(windows_scaled[i]):
                y_a, x_a = win_scaled[0] # from drawing index to query index
                y_b, x_b = win_scaled[1]
                subimg = resized_img[x_a:x_b, y_a:y_b]
                feat = self.feat_maker.get_feat(subimg).reshape(1, -1)
                test_feat = self.scaler.transform(feat)
                test_pred = self.clf.predict(test_feat)


                if test_pred == 1:
                    bboxes.append(tuple(map(tuple, windows[i][j])))
        result_boxes = bboxes + self.cached_bboxes_1# + self.cached_bboxes_2
        self.cached_bboxes_2 = self.cached_bboxes_1
        self.cached_bboxes_1 = bboxes
        return result_boxes








    ## 3. LAZY SEARCH  ##
    # lazy version of window search,
    # based on the infected_box boxes

    def set_lazy(self, boxes_info):
        self.lazy_info = boxes_info

    def _lazy_search(self, img):
        h, w = img.shape[0], img.shape[1]
        if len(self.cached_lazy) < 1:
            bboxes = self._window_search(img)
            self.cached_lazy = self._heat_box(img.shape, bboxes)
            return self.cached_lazy
        else:
            # exhaustedly search neighboring boxes
            lazy_bboxes = self._window_search(img, lazy = True)
            for box in self.cached_lazy:
                if box[0][1] - box[1][1] > 20:
                    neigbors = infected_box(box)
                    for neigbor in neigbors:
                        x_a, y_a = max(neigbor[0][0], 0), max(neigbor[0][1], 0)
                        x_b, y_b = min(neigbor[1][0], w - 1), min(neigbor[1][1], h - 1)
                        subimg = cv2.resize(img[y_a:y_b, x_a:x_b], (window, window))
                        feat = self.feat_maker.get_feat(subimg).reshape(1, -1)
                        test_feat = self.scaler.transform(feat)
                        test_pred = self.clf.predict(test_feat)

                        if test_pred == 1:
                            lazy_bboxes.append(tuple(map(tuple, neigbor)))

            self.cached_lazy = self._heat_box(img.shape, lazy_bboxes)
            return self.cached_lazy


    ## HEAT ##
    # the heatmap utility for calculating continuous boxes
    def _heat_box(self, shape, bboxes, thres = heat_thres):
        """
        get the overlapped area as boxes
        """
        heat = np.zeros(shape)
        for bbox in bboxes:
            heat[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        heatmap = (heat > thres).astype(int)

        result = []
        labels = label(heatmap)
        for car_num in range(1, labels[1] + 1):
            nonzero = (labels[0] == car_num).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            result.append(bbox)
        return result


    ## INTEGRATED ##
    # the detecting functionalities
    # integrating the functions above

    ## visualzie the boxes
    def draw(self, img, bboxes):
        """
        draw the detecting boxes from window_search
        """
        for box in bboxes:
            cv2.rectangle(img, box[0], box[1], color = [0, 255, 0], thickness = 2)
        return img


    def _find(self, img, find_style):
        im = cv2.blur(img, (5,5))
        car_boxes = find_style(im)
        bboxes = self._heat_box(img.shape, car_boxes, heat_thres)
        return self.draw(img, bboxes)

    def window_find(self, img):
        return self._find(img, self._window_search)

    def block_find(self, img):
        return self._find(img, self._block_search)

    def lazy_find(self, img):
        bboxes = self._lazy_search(img)
        return self.draw(img, bboxes)







## Main ##

positions = [(380, 600, 1.5), (380, 450, 1.5), (500, 700, 2.0)]
normal_boexs = box_generator((390, 600), (0, 1280))
lazy_boxes = box_generator((370, 660), (0, 1280), yd_ratio = lazy_yd, scales = lazy_scales, xover = lazy_xover)
out_path = './output_images'


def get_video(vf, nb_per_class = None, style = 'designed'):
    if nb_per_class:
        train_SVC(nb_per_class)

    with open('clf.pickle', 'rb') as f:
        clf, scaler, feat_maker = pickle.load(f)
    detector = car_detector(clf, scaler, feat_maker)
    detector.set_windows(normal_boexs)
    detector.set_lazy(lazy_boxes)
    detector.set_pos(positions)


    style_map = {'designed': detector.window_find,
                 'block': detector.block_find,
                 'lazy': detector.lazy_find}

    video = VideoFileClip(vf)
    video = video.fl_image(style_map[style])
    video.write_videofile('%s_output.mp4'%(vf[:-4]), audio = False)
    print("video is successfully processed! check out %s_output.mp4"%(vf[:-4]))

def get_image_result(input_path, nb_per_class = None):

    if nb_per_class:
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

    out_imgs = [detector.block_find(img) for img in imgs]
    for i in range(len(imgfs)):
        cv2.imwrite(os.path.join(out_path, imgf_names[i]), cv2.cvtColor(out_imgs[i], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required = True, type = str)
    parser.add_argument('style')
    parser.add_argument('--retrain', nargs = 1,  type = int)

    args = parser.parse_args()
    f = args.filename
    try:
        nb_train = int(args.retrain[0])
    except:
        nb_train = None
    # get_image_result(args.retrain, './test_images',nb_train)

    get_video(f, nb_train, style = args.style)
