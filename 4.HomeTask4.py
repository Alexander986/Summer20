# You have to implement your own image retrieval solution. There is a dataset with images of a few classes.
# Your code should take image filename as an input parameter, search for most similar images over the whole dataset
# and visualize input image + 5 top matches. Feel free to use any classic features/ descriptors (histograms, Gabor,
# HOG etc.) except for neural networks stuff. Select matches by the minimal distance.

import cv2
import numpy as np
import glob
import os
import time
import matplotlib.pyplot as plt


def getAllfiles_os():
    data_folder = 'dataset'

    list_files = []
    list_images = []
    list_images_gray = []
    list_fft = []
    for folder in sorted(os.listdir(data_folder)):
        if folder[0] == 'n':
            for file in sorted(os.listdir(data_folder+'/'+folder)):
                list_files.append(data_folder+'/'+folder+'/'+file)
                img = cv2.imread(data_folder+'/'+folder+'/'+file)
                list_images.append(img)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                list_images_gray.append(img_gray)

                # create fft
                f = np.fft.fft2(img_gray)
                fshifted= np.fft.fftshift(f)
                magnitude = 20 * np.log(np.abs(fshifted) + 10**-10)
                magnitude = np.asarray(magnitude, dtype = np.uint8)
                list_fft.append(magnitude)




    list_images = np.asarray(list_images, dtype = np.uint8) #np.float32
    list_images_gray = np.asarray(list_images_gray, dtype=np.uint8)  # np.float32
    list_fft = np.asarray(list_fft, dtype = np.uint8 )

    return list_files, list_images, list_images_gray, list_fft

# def createOneDatafolder():
#     os.mkdir('datasetAllTogether')


def getAllfiles_glob():
    total_list_files=[]
    list_folders = glob.iglob('dataset/*')
    for folder in list_folders:
        list_files = glob.iglob(folder+'/*')
        for file in list_files:
            total_list_files.append(file)
    return total_list_files

# def getAllfiles():
#     os.get

def runMe():
    list_files = getAllfiles_glob()
    for file in list_files:
        img = cv2.imread(file)
        cv2.imshow('img', img)
        cv2.waitKey(0)

def runMe2():
    list_files, list_images, list_images_gray = getAllfiles_os()
    cv2.imshow('img', list_images[0])

    mydict=dict()
    for i in range(len(list_images)):
        mydict[i] = cv2.norm(list_images[0], list_images[i])
        # print("i=",i," : ", cv2.norm(list_images[0], list_images[i]))
    mydict2 = sorted(mydict, key = lambda x: mydict[x])
    print(mydict2)
    # for i in dict:
    #     print(i, " : ", dict[i])
    cv2.waitKey(3000)

def runMe3():
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()

    # mydict = {}
    # for i in range(len(list_fft)):
    #     mydict[i] = cv2.norm(list_fft[0], list_fft[i])
    # mydict2 = sorted(mydict, key = lambda x: mydict[x])
    # print(type(mydict2))
    # print(mydict2)

    mydict = {}
    for i in range(len(list_fft)):
        # mydict[i] = cv2.matchTemplate(list_fft[0], list_fft[i], cv2.TM_CCOEFF_NORMED)
        mydict[i] = cv2.matchTemplate(list_fft[700], list_fft[i], cv2.TM_CCOEFF)
    mydict2 = sorted(mydict, key=lambda x: -mydict[x])
    print(mydict2)

    cv2.imshow('img0', list_fft[mydict2[0]])
    cv2.imshow('img1', list_fft[mydict2[1]])
    cv2.imshow('img2', list_fft[mydict2[2]])

    cv2.imshow('img0', list_images[mydict2[0]])
    cv2.imshow('img1', list_images[mydict2[1]])
    cv2.imshow('img2', list_images[mydict2[2]])

    cv2.waitKey(0)

    print(type(mydict2))
    print(mydict2)

def runMe4():
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()

    i=711
    j=982

    orb = cv2.ORB_create(nfeatures = 2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    kp0, descr0 = orb.detectAndCompute(list_images_gray[i], None)
    kp1, descr1 = orb.detectAndCompute(list_images_gray[j], None)
    matches = bf.match(descr0, descr1)
    matches = sorted(matches, key = lambda x: x.distance)
    print(len(matches))
    matching_result = cv2.drawMatches(list_images_gray[i], kp0, list_images_gray[j], kp1, matches[:10], None)
    for m in matches:
        print(m.distance)
    cv2.imshow('matching', matching_result)
    cv2.waitKey(0)


def runMe5():
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()

    feature_params = dict(maxCorners = 1000,
                          qualityLevel = 0.02,
                          minDistance = 7,
                          blockSize = 7)
    i=0
    j=1
    p0 = cv2.goodFeaturesToTrack(list_images_gray[i], mask =None, **feature_params)
    p1 = cv2.goodFeaturesToTrack(list_images_gray[j], mask=None, **feature_params)

    print(p0)
    return
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    orb = cv2.ORB_create(nfeatures = 2000)
    res = orb.compute(list_images_gray[i], None)
    print(res)

def dist_eval(hist1, hist2, option = 1):
    if option == 1 or option == 'chi2':
        return np.sum((hist1-hist2)**2 / (hist1+hist2+10**-10) )
    if option == 2 or option == 'cos':
        return -np.abs(np.sum(hist1*hist2) / ( (np.sum(hist1**2))**(0.5) * (np.sum(hist2**2))**(0.5) ))
    # return(cv2.norm(hist1, hist2))
    # return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def runMe6():
    bins = [8, 12, 3]
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()


    i=1233

    img1_hsv = cv2.cvtColor(list_images[i], cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    distances={} # create a dictionary to save the distances
    distances1 = {}
    distances2 = {}

    f1 = np.fft.fft2(list_images_gray[i])
    fshift1 = np.fft.fftshift(f1)
    magnitude1 = 20 * np.log(np.abs(fshift1) + 10**-10)

    for j in range(len(list_images)):
        # histogram matching in hsv space
        img2_hsv = cv2.cvtColor(list_images[j], cv2.COLOR_BGR2HSV)
        hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        distances1[j] = dist_eval(hist1, hist2, option = 1)

        # fft comparison
        f2 = np.fft.fft2(list_images_gray[j])
        fshift2 = np.fft.fftshift(f2)
        magnitude2 = 20 * np.log(np.abs(fshift2) + 10**-10)
        distances2[j] = dist_eval(magnitude1, magnitude2, option = 2)

        distances[j] = distances1[j]

    sorted_distances = sorted(distances, key = lambda x: distances[x])

    n=6
    h, w, ch = list_images[i].shape
    print(h, w, ch)
    res = np.zeros((h, w*n, ch), dtype = np.uint8)
    res_fft = np.zeros((h, w*n), dtype = np.uint8)
    for j in range(n):
        res[:, j*w: (j+1)*w, :] = list_images[sorted_distances[j]]
        cv2.imshow('res'+str(j), list_images[sorted_distances[j]])
        print(sorted_distances[j], distances[sorted_distances[j]])

        f = np.fft.fft2(list_images_gray[sorted_distances[j]])
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 10**-10)
        res_fft[:, j*w:(j+1)*w] = magnitude

    res = cv2.resize(res, None,  fx = 3, fy = 3)
    res_fft = cv2.resize(res_fft, None, fx = 3, fy = 3)
    cv2.imshow('res', res)
    cv2.imshow('res_fft', res_fft)
    cv2.waitKey(0)

    # a1 = [231, 528, 255, 119, 1512, 4]
    # a2 = [231, 528, 225, 322, 360, 531]
    # for i in a1:
    #     print(i," : ", distances[i])
    # for i in a2:
    #     print(i, " : ", distances[i])

    return
    # for j in range(len(list_fft)):

    hist1 = cv2.calcHist([list_images[i]], [0], mask = None, histSize = [256], ranges = [0, 256])
    hist2 = cv2.calcHist([list_images[j]], [0], mask=None, histSize=[256], ranges=[0, 256])

    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print(res)

    plt.subplot(121), plt.plot(hist1)
    plt.subplot(122), plt.plot(hist2)
    plt.show()

def runMe6_2():
    bins = [60, 4, 2]
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()


    i=41

    # creating masks
    w, h = list_images[i].shape[:2]
    cX, cY = int(w/2), int(h/2)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
    axesX, axesY = int(w * 0.375), int(h * 0.375)
    ellipMask = np.zeros(list_images[i].shape[:2], dtype = np.uint8)
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
    # creating other masks
    masks = [ellipMask]
    for (startX, endX, startY, endY) in segments:
        cornerMask = np.zeros(list_images[i].shape[:2], dtype = np.uint8)
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)
        masks.append(cornerMask)

    img1_hsv = cv2.cvtColor(list_images[i], cv2.COLOR_BGR2HSV)
    features1=[]
    for mask in masks:
        hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        features1.extend(hist1)
    features1 = np.asarray(features1, dtype=np.float32)
    distances={} # create a dictionary to save the distances
    distances1 = {}
    distances2 = {}

    f1 = np.fft.fft2(list_images_gray[i])
    fshift1 = np.fft.fftshift(f1)
    magnitude1 = 20 * np.log(np.abs(fshift1) + 10**-10)

    for j in range(len(list_images)):
        # histogram matching in hsv space
        img2_hsv = cv2.cvtColor(list_images[j], cv2.COLOR_BGR2HSV)
        features2=[]
        for mask in masks:
            hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()
            features2.extend(hist2)
        features2 = np.asarray(features2, dtype = np.float32)
        distances1[j] = dist_eval(features1, features2, option = 1)

        # fft comparison
        f2 = np.fft.fft2(list_images_gray[j])
        fshift2 = np.fft.fftshift(f2)
        magnitude2 = 20 * np.log(np.abs(fshift2) + 10**-10)
        distances2[j] = dist_eval(magnitude1, magnitude2, option = 2)

        distances[j] = distances1[j]

    sorted_distances = sorted(distances, key = lambda x: distances[x])

    n=6
    h, w, ch = list_images[i].shape
    print(h, w, ch)
    res = np.zeros((h, w*n, ch), dtype = np.uint8)
    res_fft = np.zeros((h, w*n), dtype = np.uint8)
    for j in range(n):
        res[:, j*w: (j+1)*w, :] = list_images[sorted_distances[j]]
        cv2.imshow('res'+str(j), list_images[sorted_distances[j]])
        print(sorted_distances[j], distances[sorted_distances[j]])

        f = np.fft.fft2(list_images_gray[sorted_distances[j]])
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 10**-10)
        res_fft[:, j*w:(j+1)*w] = magnitude

    res = cv2.resize(res, None,  fx = 3, fy = 3)
    res_fft = cv2.resize(res_fft, None, fx = 3, fy = 3)
    cv2.imshow('res', res)
    cv2.imshow('res_fft', res_fft)
    cv2.waitKey(0)

    # a1 = [231, 528, 255, 119, 1512, 4]
    # a2 = [231, 528, 225, 322, 360, 531]
    # for i in a1:
    #     print(i," : ", distances[i])
    # for i in a2:
    #     print(i, " : ", distances[i])

    return
    # for j in range(len(list_fft)):

    hist1 = cv2.calcHist([list_images[i]], [0], mask = None, histSize = [256], ranges = [0, 256])
    hist2 = cv2.calcHist([list_images[j]], [0], mask=None, histSize=[256], ranges=[0, 256])

    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print(res)

    plt.subplot(121), plt.plot(hist1)
    plt.subplot(122), plt.plot(hist2)
    plt.show()

def runMe8(input_file = "dataset/n01855672/n0185567200000003.jpg"):
    bins = [60, 4, 2]
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()

    img1 = cv2.imread(input_file)

    # creating masks
    w, h = img1.shape[:2]
    cX, cY = int(w/2), int(h/2)
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
    axesX, axesY = int(w * 0.375), int(h * 0.375)
    ellipMask = np.zeros(img1.shape[:2], dtype = np.uint8)
    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
    # creating other masks
    masks = [ellipMask]
    for (startX, endX, startY, endY) in segments:
        cornerMask = np.zeros(img1.shape[:2], dtype = np.uint8)
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        cornerMask = cv2.subtract(cornerMask, ellipMask)
        masks.append(cornerMask)

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    features1=[]
    for mask in masks:
        hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        features1.extend(hist1)
    features1 = np.asarray(features1, dtype=np.float32)
    distances = {} # create a dictionary to save the distances
    distances1 = {}
    # distances2 = {}

    f1 = np.fft.fft2(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    fshift1 = np.fft.fftshift(f1)
    magnitude1 = 20 * np.log(np.abs(fshift1) + 10**-10)

    for j in range(len(list_images)):
        # histogram matching in hsv space
        img2_hsv = cv2.cvtColor(list_images[j], cv2.COLOR_BGR2HSV)
        features2=[]
        for mask in masks:
            hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()
            features2.extend(hist2)
        features2 = np.asarray(features2, dtype = np.float32)
        distances1[j] = dist_eval(features1, features2, option = 1)

        # # fft comparison
        # f2 = np.fft.fft2(list_images_gray[j])
        # fshift2 = np.fft.fftshift(f2)
        # magnitude2 = 20 * np.log(np.abs(fshift2) + 10**-10)
        # distances2[j] = dist_eval(magnitude1, magnitude2, option = 2)

        distances[j] = distances1[j]

    sorted_distances = sorted(distances, key = lambda x: distances[x])

    # if the first image is in the dataset, start the suggestions from the second closest
    N=5
    if distances[sorted_distances[0]] == 0:
        n=N+1
    else:
        n=N
    h, w, ch = img1.shape
    res = np.zeros((h, w*N, ch), dtype = np.uint8)
    res_fft = np.zeros((h, w*N), dtype = np.uint8)
    for j in range(n-N, n):
        res[:, (j+N-n)*w: (j+N-n+1)*w, :] = list_images[sorted_distances[j]]
        # cv2.imshow('res'+str(j), list_images[sorted_distances[j]])
        # print(sorted_distances[j], distances[sorted_distances[j]])

        # f = np.fft.fft2(list_images_gray[sorted_distances[j]])
        # fshift = np.fft.fftshift(f)
        # magnitude = 20 * np.log(np.abs(fshift) + 10**-10)
        # res_fft[:, j*w:(j+1)*w] = magnitude



    img1 = cv2.resize(img1, None, fx = 3, fy = 3)
    cv2.imshow('original', img1)
    res = cv2.resize(res, None,  fx = 3, fy = 3)
    # res_fft = cv2.resize(res_fft, None, fx = 3, fy = 3)
    cv2.imshow('similar', res)
    # cv2.imshow('res_fft', res_fft)
    cv2.waitKey(0)

    # a1 = [231, 528, 255, 119, 1512, 4]
    # a2 = [231, 528, 225, 322, 360, 531]
    # for i in a1:
    #     print(i," : ", distances[i])
    # for i in a2:
    #     print(i, " : ", distances[i])

    return
    # for j in range(len(list_fft)):

    hist1 = cv2.calcHist([list_images[i]], [0], mask = None, histSize = [256], ranges = [0, 256])
    hist2 = cv2.calcHist([list_images[j]], [0], mask=None, histSize=[256], ranges=[0, 256])

    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print(res)

    plt.subplot(121), plt.plot(hist1)
    plt.subplot(122), plt.plot(hist2)
    plt.show()

def runMe7():
    bins = [8, 12, 3]
    list_files, list_images, list_images_gray, list_fft = getAllfiles_os()

    for i in range(len(list_images)):

        img1_hsv = cv2.cvtColor(list_images[i], cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([img1_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        distances = {}  # create a dictionary to save the distances
        distances1 = {}
        distances2 = {}

        f1 = np.fft.fft2(list_images_gray[i])
        fshift1 = np.fft.fftshift(f1)
        magnitude1 = 20 * np.log(np.abs(fshift1) + 10 ** -10)

        for j in range(len(list_images)):
            # histogram matching in hsv space
            img2_hsv = cv2.cvtColor(list_images[j], cv2.COLOR_BGR2HSV)
            hist2 = cv2.calcHist([img2_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
            distances1[j] = dist_eval(hist1, hist2, option=1)

            # fft comparison
            f2 = np.fft.fft2(list_images_gray[j])
            fshift2 = np.fft.fftshift(f2)
            magnitude2 = 20 * np.log(np.abs(fshift2) + 10 ** -10)
            distances2[j] = dist_eval(magnitude1, magnitude2, option=2)

            distances[j] = distances1[j]

        sorted_distances = sorted(distances, key=lambda x: distances[x])

        n = 6
        h, w, ch = list_images[i].shape
        print(h, w, ch)
        res = np.zeros((h, w * n, ch), dtype=np.uint8)
        res_fft = np.zeros((h, w * n), dtype=np.uint8)
        for j in range(n):
            res[:, j * w: (j + 1) * w, :] = list_images[sorted_distances[j]]
            cv2.imshow('res' + str(j), list_images[sorted_distances[j]])
            print(sorted_distances[j], distances[sorted_distances[j]])

            f = np.fft.fft2(list_images_gray[sorted_distances[j]])
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 10 ** -10)
            res_fft[:, j * w:(j + 1) * w] = magnitude

        res = cv2.resize(res, None, fx=3, fy=3)
        res_fft = cv2.resize(res_fft, None, fx=3, fy=3)
        cv2.imshow('res', res)
        cv2.imshow('res_fft', res_fft)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # a1 = [231, 528, 255, 119, 1512, 4]
    # a2 = [231, 528, 225, 322, 360, 531]
    # for i in a1:
    #     print(i," : ", distances[i])
    # for i in a2:
    #     print(i, " : ", distances[i])

    return
    # for j in range(len(list_fft)):

    hist1 = cv2.calcHist([list_images[i]], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist2 = cv2.calcHist([list_images[j]], [0], mask=None, histSize=[256], ranges=[0, 256])

    res = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print(res)

    plt.subplot(121), plt.plot(hist1)
    plt.subplot(122), plt.plot(hist2)
    plt.show()

if __name__ == '__main__':
    start = time.time()

    runMe8("dataset/n02114548/n0211454800000058.jpg") Good
    # runMe8("dataset/n02114548/n0211454800000051.jpg") Bad
    # runMe8("dataset/n02114548/n0211454800000076.jpg") Bad
    # runMe8("dataset/n02114548/n0211454800000056.jpg")




    print(time.time()-start, "seconds")
