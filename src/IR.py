import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import shutil

def splitDataset(ratio):
    caltech = '../caltrain'
    _, directories, _ = next(os.walk(caltech))

    nb_data_class = np.zeros(len(directories))
    for d in range(len(directories)):
        path = os.path.join(caltech, directories[d])
        paths = np.array(os.listdir(path))
        nb_data_class[d] = len(paths)
    nb_val = np.round(nb_data_class * ratio)

    for d in range(len(directories)):
        origin = os.path.join(caltech, directories[d])
        tosave = os.path.join('../caltest', directories[d])

        if not os.path.exists(tosave):
            os.makedirs(tosave)
        paths = np.array(os.listdir(origin))
        for t in range(int(nb_val[d])):
            dest = os.path.join(tosave, paths[t])
            source = os.path.join(origin, paths[t])
            shutil.move(source, dest)


def getImages(path):
    imgs = []
    for f in os.listdir(path):
        for img in os.listdir(os.path.join(path, f)):
            imgs.append(os.path.join(path, os.path.join(f, img)))
    return imgs

def kmClustering(all_features, nb_clus, epsilon, max_iter):
    n = all_features.shape[0]
    dim = all_features.shape[1]

    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    centers = np.random.randn(nb_clus,dim)*std + mean

    centers_old = np.zeros(centers.shape)
    centers_new = centers

    error = np.linalg.norm(centers_new - centers_old)

    distances = np.zeros((n,nb_clus))
    clusters = np.zeros(n)

    iteration = 0
    while error >= epsilon and iteration < max_iter :
        for i in range(nb_clus):
            distances[:, i] = np.linalg.norm(all_features - centers_new[i], axis = 1)

        clusters = np.argmin(distances, axis = 1)
        centers_old = centers_new
        for i in range(nb_clus):
            centers_new[i] = np.mean(all_features[clusters == i],axis= 0)

        error = np.linalg.norm(centers_new - centers_old)
        iteration = iteration + 1

    nans = []
    for i in range(nb_clus):
        if np.isnan(centers_new[i,0]):
            nans.append(i)
    centers_new = np.delete(centers_new, nans, axis=0)

    return centers_new

def getCodebook(imgs_paths, nb_clus):
    all_features = []
    for path in imgs_paths:
        img = cv2.imread(path)[:, :, ::-1]
        features = feature_extraction(img)
        all_features.append(features)
    all_features = np.concatenate(all_features, 0)

    centers = kmClustering(all_features, nb_clus, 1e-4, 1000)

    return centers

def feature_extraction(img, kpts=False):
    sift = cv2.xfeatures2d.SIFT_create()
    descripters = []
    height = img.shape[0]
    width = img.shape[1]
    split1 = np.array_split(img, width/20, axis=1)
    for split in split1:
        split2 = np.array_split(split, height/20, axis=0)
        for ig in split2:
            _, descripter = sift.detectAndCompute(ig, None)
            if descripter is not None:
                descripters.append(descripter)
    if len(descripters) > 0:
        descripters = np.vstack(descripters)
    else:
        return None

    if kpts:
        keypoints, _ = sift.detectAndCompute(img, None)
        return keypoints

    return descripters

def getFeaturesPCA(codebook, feat_num):
    avr = np.mean(codebook, axis = 0)
    codebook_norm = codebook - avr

    covMat = np.cov(codebook_norm, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))

    i = np.argsort(eigVals)
    feat_num_indice = i[-1:-(feat_num+1):-1]
    feat_num_eigVect = eigVects[:,feat_num_indice]
    reduction_codebook = codebook_norm * feat_num_eigVect

    return reduction_codebook

def pdist(a, b):
    a_square = np.einsum('ij,ij->i', a, a)
    a_square = np.tile(np.reshape(a_square, [a.shape[0], 1]), [1, b.shape[0]])

    b_square = np.einsum('ij,ij->i', b, b)
    b_square = np.tile(np.reshape(b_square, [1, b.shape[0]]), [a.shape[0], 1])

    ab = np.dot(a, b.T)

    dist = a_square + b_square - 2 * ab
    dist = dist.clip(min=0)

    return np.sqrt(dist)

def getBOW(codebook, imgs_paths):
    codebook_size = codebook.shape[0]
    bag_of_words = np.zeros((len(imgs_paths), codebook_size))
    n = 0
    for path in imgs_paths:
        img = cv2.imread(path)[:, :, ::-1]
        features = feature_extraction(img)
        distances = []
        for i in range(features.shape[0]):
            distances = pdist(np.mat(features[i]), codebook)
            indice = np.argsort(distances)[0, 0]
            bag_of_words[n, indice] = bag_of_words[n, indice] + 1
        n = n + 1
    bag_of_words = bag_of_words / bag_of_words.max(axis=0)
    return bag_of_words

def getSPF(codebook, imgs_paths, lvl):
    codebook_size = codebook.shape[0]

    imgs_n = int((1 / 3) * (4 ** (lvl + 1) - 1))
    sp = np.zeros((len(imgs_paths), codebook_size*imgs_n)).astype(np.float)
    n = 0

    for path in imgs_paths:
        img_origin = cv2.imread(path)
        height = img_origin.shape[0]
        width = img_origin.shape[1]
        idx = 0
        for l in range(0, lvl+1):
            item_width = int(width / (2**l))
            item_height = int(height / (2**l))
            for i in range(0, 2**l):
                for j in range(0, 2**l):
                    subimg = img_origin[j*item_height:(j+1)*item_height, i*item_width:(i+1)*item_width, :]
                    features = feature_extraction(subimg)
                    distances = []
                    if features is None:
                        print('NONE')
                    else:
                        for k in range(features.shape[0]):
                            distances = pdist(np.mat(features[k]), codebook)
                            indice = np.argsort(distances)[0,0]
                            sp[n,idx * codebook_size + indice] = sp[n,idx * codebook_size + indice] + 1
                    idx = idx + 1
        n += 1
    return sp

def svmClassifier(train_lab, train_image_feats, test_image_feats):
    categories = np.unique(train_lab)
    prediction = []
    lin_clf = svm.LinearSVC()
    lin_clf.fit(train_image_feats, train_lab)
    df = lin_clf.decision_function(test_image_feats)
    for i in range(df.shape[0]):
        max_index = list(df[i]).index(max(df[i]))
        prediction.append(categories[max_index])

    prediction = np.array(prediction)
    return prediction

def getAllPathsAndLabels():
    train_image_paths = getImages('../caltrain')
    test_image_paths = getImages('../caltest')
    train_labels, test_labels = [], []

    for img_path in train_image_paths:
        if("ant" in img_path):
            class_index = 0
        elif("beaver" in img_path):
            class_index = 1
        elif("brontosaurus" in img_path):
            class_index = 2
        elif("cannon" in img_path):
            class_index = 3
        elif("chair" in img_path):
            class_index = 4
        elif("crab" in img_path):
          class_index = 5
        elif("cup" in img_path):
            class_index = 6
        elif("dragonfly" in img_path):
            class_index = 7
        elif("euphonium" in img_path):
            class_index = 8
        elif("ferry" in img_path):
            class_index = 9
        elif("gerenuk" in img_path):
          class_index = 10
        elif("headphone" in img_path):
            class_index = 11
        elif("inline_skate" in img_path):
            class_index = 12
        elif("lamp" in img_path):
            class_index = 13
        elif("lobster" in img_path):
            class_index = 14
        elif("menorah" in img_path):
          class_index = 15
        elif("nautilus" in img_path):
            class_index = 16
        elif("panda" in img_path):
            class_index = 17
        elif("pyramid" in img_path):
            class_index = 18
        else:
          class_index = 19
        train_labels = np.append(train_labels, class_index)

    for img_path in test_image_paths:
        if("ant" in img_path):
            class_index = 0
        elif("beaver" in img_path):
            class_index = 1
        elif("brontosaurus" in img_path):
            class_index = 2
        elif("cannon" in img_path):
            class_index = 3
        elif("chair" in img_path):
            class_index = 4
        elif("crab" in img_path):
          class_index = 5
        elif("cup" in img_path):
            class_index = 6
        elif("dragonfly" in img_path):
            class_index = 7
        elif("euphonium" in img_path):
            class_index = 8
        elif("ferry" in img_path):
            class_index = 9
        elif("gerenuk" in img_path):
          class_index = 10
        elif("headphone" in img_path):
            class_index = 11
        elif("inline_skate" in img_path):
            class_index = 12
        elif("lamp" in img_path):
            class_index = 13
        elif("lobster" in img_path):
            class_index = 14
        elif("menorah" in img_path):
          class_index = 15
        elif("nautilus" in img_path):
            class_index = 16
        elif("panda" in img_path):
            class_index = 17
        elif("pyramid" in img_path):
            class_index = 18
        else:
          class_index = 19

        test_labels = np.append(test_labels, class_index)

    return train_image_paths, test_image_paths, train_labels, test_labels

def confusionMatrix(true, predictions):
    np.set_printoptions(precision=1)
    classes = ["ant", "beaver", "brontosaurus", "cannon", "chair", "crab", "cup", "dragonfly", "euphonium", "ferry", "gerenuk", "headphone", "inline_skate", "lamp", "lobster", "menorah", "nautilus", "panda", "pyramid", "saxophone"]

    cmap=plt.cm.Blues
    title = 'Normalized confusion matrix'
    confM = confusion_matrix(true, predictions)
    confM = confM.astype('float') / confM.sum(axis=1)[:, np.newaxis]
    print(title)

    f, axis = plt.subplots()
    im = axis.imshow(confM, interpolation='nearest', cmap=cmap)
    axis.figure.colorbar(im, ax=axis)

    axis.set(xticks=np.arange(confM.shape[1]),
           yticks=np.arange(confM.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(axis.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.1f'
    th = confM.max() / 3.
    for i in range(confM.shape[0]):
        for j in range(confM.shape[1]):
            axis.text(j, i, format(confM[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confM[i, j] < th else "black")
    f.tight_layout()

    plt.title('Confusion matrix')
    plt.show()

def accuracyScore(true, pred):
    print('Accuracy : %0.3f' % accuracy_score(true, pred))

def visualizeVisualWords(path):
    img = mpimg.imread(path)
    print('Image read')
    kp = feature_extraction(img, kpts=True)
    kpImg = img.copy()

    for curKey in kp:
        x = np.int(curKey.pt[0])
        y = np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(kpImg, (x, y), size, (0, 255, 0), thickness=5, lineType=8, shift=0)

    plt.imshow(kpImg)
    plt.title('Visualizing the important features of the picture of an ant')
    print('kp Img')
    plt.show()


def main():
    #Use this to split the dataset in the training and testing folders if it is not done yet
    splitDataset(0.2)

    train_image_paths, test_image_paths, train_labels, test_labels = getAllPathsAndLabels()
    print('All paths and labels saved')

    nb_clus = 100
    codebook = getCodebook(train_image_paths, nb_clus)
    print('Codebook learned')

    #codebook = getFeaturesPCA(codebook, 128)
    #print('Dimensions reduced')

    representation = 'Bag of Visual Words'
    representation = None
    if representation == 'Bag of Visual Words':
        train_image_feats = getBOW(codebook, train_image_paths)
        test_image_feats = getBOW(codebook, test_image_paths)
    else:
        train_image_feats = getSPF(codebook, train_image_paths, 1)
        test_image_feats = getSPF(codebook, test_image_paths, 1)
    print(representation, 'done')

    pred = svmClassifier(train_labels, train_image_feats, test_image_feats)
    print('Predictions finished')
    print(classification_report(test_labels, pred))

    confusionMatrix(test_labels, pred)
    print('Confusion matrix')

    accuracyScore(test_labels, pred)
    print('Accuracy score')

    chosenImg = '../caltest/ant/image_0001.jpg'
    visualizeVisualWords(chosenImg)
    print('Visual Words visualized')

    print('Finished')


if __name__ == '__main__':
    main()
