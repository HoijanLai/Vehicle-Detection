import numpy as np
import cv2
import time
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler as Xscaler
import pickle


window = 64 # sampling window (side)
###############################
# params for the hog features #
###############################
ppc = 16 # pixels per cell
cpb = 2  # cells per block
ort = 6  # orientations

#################################
# params for the color features #
#################################
cspace = 'YCrCb'    # the color space
spatial_size = (8, 8)
hist_bins = 16

#########################################################
# I hate the way cv2 convert color, so I created a dict #
#########################################################


CVT = {'HSV': cv2.COLOR_RGB2HSV,
       'LUV': cv2.COLOR_RGB2LUV,
       'HLS': cv2.COLOR_RGB2HLS,
       'YUV': cv2.COLOR_RGB2YUV,
     'YCrCb': cv2.COLOR_RGB2YCrCb,
      'GRAY': cv2.COLOR_RGB2GRAY,
       'RGB': cv2.COLOR_BGR2RGB}

###########################################
# the paths, non for non-car, car for car #
###########################################
non = ['./data/non-vehicles/non-vehicles/Extras',
       './data/non-vehicles/non-vehicles/GTI']

car = ['./data/vehicles/vehicles/GTI_Far',
       './data/vehicles/vehicles/GTI_Left',
       './data/vehicles/vehicles/GTI_MiddleClose',
       './data/vehicles/vehicles/GTI_Right',
       './data/vehicles/vehicles/KITTI_extracted']



########### Feature Functions ##############


def hog_feat(img, feat_vec=False):
    """
    get the hog feature according to the global
    parameters

    params:
        feat_vec: bool, whether to ravel the hog feature
    """
    gray = CVT['GRAY']
    im = cv2.cvtColor(img, gray)
    return hog(im, orientations=ort,
                pixels_per_cell=(ppc, ppc),
                cells_per_block=(cpb, cpb),
                 transform_sqrt=False, feature_vector=feat_vec)

def color_bin_feat(img):
    """
    get the color bins according to the
    global parameters

    """
    colors = [cv2.resize(img[:,:,i], spatial_size).ravel()
              for i in range(img.shape[2])]
    return np.hstack(colors)

def color_hist_feat(img):
    """
    get the histogram of colors according to the
    global parameters
    """
    ch_hist = [np.histogram(img[:,:,i], bins = hist_bins)[0]
               for i in range(img.shape[2])]
    return np.hstack(ch_hist)



class feature_maker:
    """
    a wrap up feature maker according to the global parameters

    the instance is pickled so that the feature is consistent in both
    the classifier and the detector.


    """
    def __init__(self, hog = True, color_bins = False, color_hists = False):
        """
        you can select which kind of infomation to involve
        in the feature by setting bools
        """
        self.hog = hog
        self.color_bins = color_bins
        self.color_hists = color_hists

    def get_feat(self, img = None, hog_ravel = None, cvt = True):
        """
        if hog_ravel is set, skip the hog_feat process
        """

        feat = []
        if self.hog and hog_ravel is None:
            feat.append(hog_feat(img, feat_vec = True))
        else:
            feat.append(hog_ravel)

        im = np.copy(img)
        if (self.color_bins or self.color_hists) and cvt:
            im = cv2.cvtColor(img, CVT[cspace])
        if self.color_bins:
            feat.append(color_bin_feat(im))
        if self.color_hists:
            feat.append(color_hist_feat(im))

        return np.concatenate(feat)




############## File system stuff #####################

def from_paths(yes_paths, no_paths, nb_per_class,
               hog = True, color_bins = False, color_hists = False, verbose = True):
    """
    given true folders and false folders, read, get features and fit

    params:
        yes_paths: str or array-like of str, the paths for true samples
        no_paths: same as the yes paths, for false samples
        hog_params: a dictionary for hog operations

    return: a dataset and fitted scaler on the feature_vector
    """
    start = time.time()
    check = start

    if verbose:
        print("reading samples list...\n")

    plural = lambda p : p if hasattr(p, "__len__") and (not isinstance(p, str)) else [p]
    t_paths = plural(yes_paths)
    f_paths = plural(no_paths)

    ##       = 1 =
    ##  read files list to make a sketch of the data

    def get_files(ps):
        """
        return nb_per_class random files
        """
        fs = []
        for p in ps:
            for ext in ['.png', '.jpg', '.jpeg']:
                fs.extend(glob('%s/**%s'%(p, ext)))
        return np.random.choice(np.array(fs), nb_per_class, replace = False)


    files = np.concatenate((get_files(t_paths), get_files(f_paths)))
    y = np.concatenate((np.ones(nb_per_class), np.zeros(nb_per_class)))
    files, y = shuffle(files, y)


    if verbose:
        print("reading sample file list takes %.3fs\n"%(time.time() - check))
    check = time.time()

    ##   = 2 =
    ## actually read files and make the features
    X = []
    if verbose:
        print("getting hog features...")

    # initialize the feature maker to get feature from images

    feat_maker = feature_maker(hog, color_bins, color_hists)

    for f in tqdm(files):
        # read the images
        im = cv2.cvtColor(cv2.imread(f), CVT['RGB'])
        feat = feat_maker.get_feat(im)
        X.append(feat)

    X = np.vstack(X)

    # the scaler should be returned to make sure constency
    scaler = Xscaler()
    X = scaler.fit_transform(X)

    if verbose:
        print("getting features takes %.3fs"%(time.time() - check))
    check = time.time()
    return X, y, scaler, feat_maker



############# The training #############

def train_SVC(nb_per_class):
    """
    a training pipeline for the svc
    """

    # == 1 ==
    # use the from_paths function to get the data, scaler and feature maker,
    # which we need in the detector script

    X, y, scaler, feat_maker =from_paths(car, non, nb_per_class, color_bins = True, color_hists = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
    print(X.shape, y.shape)
    print(X[0].min(), X[0].max(), X[0].mean())

    # == 2 ==
    # train the LinearSVC classifier

    clf = SVC(random_state = 30)

    print("\ntraining the %s..."%type(clf).__name__)
    start = time.time()

    clf.fit(X_train, y_train)

    print("training takes %.2fs"%(time.time() - start))

    # == 3 ==
    # do a quick prediction
    start = time.time()
    y_pred = clf.predict(X_test)
    print("prediction takes %.2fs"%(time.time() - start))
    print(np.sum(y_pred == y_test)/y_test.shape[0])

    # == 4 ==
    # Serialize the model because the training may takes some time
    with open('clf.pickle', 'wb') as f:
        pickle.dump((clf, scaler, feat_maker), f)
    print("the model is saved!")
