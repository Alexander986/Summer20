#Literature
# opencv tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# Histograms https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_table_of_contents_histograms/py_table_of_contents_histograms.html#table-of-content-histograms
# Histograms first appearance: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#histograms-getting-started
# Histogram equalization: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization

# smoothing: blurs https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html

import cv2
import numpy as np
import matplotlib.pyplot as plt # for the histograms


def playvideo():
    '''Play the original video'''
    input_file = "input_video.avi"


    cap = cv2.VideoCapture(input_file)
    print(cap.isOpened())

    ret, frm = cap.read()

    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]
        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5)
        cv2.imshow("Video", frm_fringed)

        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 30
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    cv2.waitKey(0)

def play_and_write_video():
    ''' Play the original video and write it into output file '''

    # open the file
    input_file = "input_video.avi"
    output_file= "output_video.avi"

    # read the first frame
    cap = cv2.VideoCapture(input_file)
    # print(cap.isOpened())

    ret, frm = cap.read()
    frm_fringed = frm[60:-60,:,:]


    # create the output file format
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
    frames_per_second = 20.0
    image_size = (frm_fringed.shape[1], frm_fringed.shape[0])
    writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size )


    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]
        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5)
        # cv2.imshow("Video", frm)

        writer.write(frm_fringed)


        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 1
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def creation_of_samples_of_frames():
    input_file = "input_video.avi"

    cap = cv2.VideoCapture(input_file)
    print(cap.isOpened())

    ret, frm = cap.read()

    frm_count = 0
    key = ord(' ')

    # first pick up some frames with different luminency
    # bright: 50 109 164 179 225 266        dark: 362 444
    N = 0 # number of the frame which we save

    while (ret):

        if not (N-1 < frm_count < N+1):
            frm_count += 1
            ret, frm = cap.read()
            continue

        print(frm_count)

        frm_fringed = frm[60:-60, :, :]
        cv2.imshow("frame"+str(N), frm_fringed)
        cv2.imwrite('frame'+str(N)+'.jpg', frm_fringed)

        ret, frm = cap.read()
        frm_count += 1
    cap.release()
    cv2.waitKey(0)

def displaying_selected_frames():
    '''Preprocesing of some frames from the video'''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg",\
                      "frame362.jpg", "frame444.jpg"]
    for el in list_of_frames:
        frm = cv2.imread(el)
        cv2.putText(frm, el, (10,100), 0, 2, [255,255,255],5)
        cv2.imshow("frame", frm)
        cv2.waitKey(0)

def analysis_of_some_frames():
    '''Preprocesing of some frames from the video'''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg",\
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    #rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1]//7*6, frm.shape[0]//7*6 ))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)
    frm_gray = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2GRAY)




    # cv2.imshow("frame", frm_resized)
    # cv2.imshow("HSV", frm_hsv)
    # cv2.imshow("Gray", frm_gray)

    frm_split = split_into_3_ch(frm_hsv)

    # # histg = cv2.calcHist([frm_hsv[:,:,0]], [0], None, [256], [0,256])
    # # print(type(histg))
    # plt.hist(frm_resized[:, :, 0].flat, bins=100, range=(0, 255))
    # plt.hist(frm_resized[:, :, 1].flat, bins=100, range=(0, 255))
    # plt.hist(frm_resized[:, :, 2].flat, bins=100, range=(0, 255))
    #
    # # plt.plot(histg)
    # plt.show()


    #Histograms of hsv
    hist_hsv_h = cv2.calcHist([frm_hsv], [0], mask=None, histSize=[256], ranges = [0, 256])
    hist_hsv_s = cv2.calcHist([frm_hsv], [1], mask=None, histSize=[256], ranges = [0, 256])
    hist_hsv_v = cv2.calcHist([frm_hsv], [2], mask=None, histSize=[256], ranges = [0, 256])

    # drawing hsv histograms
    plt.subplot(421), plt.imshow(frm_hsv[:,:,:],'gray')
    plt.subplot(422), plt.plot(hist_hsv_h)
    plt.subplot(423), plt.plot(hist_hsv_s)
    plt.subplot(424), plt.plot(hist_hsv_v)



    #Histograms of BGR
    hist_bgr_h = cv2.calcHist([frm_resized], [0], mask=None, histSize=[256], ranges =[0,256])
    hist_bgr_s = cv2.calcHist([frm_resized], [1], mask=None, histSize=[256], ranges=[0, 256])
    hist_bgr_v = cv2.calcHist([frm_resized], [2], mask=None, histSize=[256], ranges=[0, 256])

    # drawing BGR histograms
    plt.subplot(425), plt.imshow(frm_resized[:,:,:],'gray')
    plt.subplot(426), plt.plot(hist_bgr_h)
    plt.subplot(427), plt.plot(hist_bgr_s)
    plt.subplot(428), plt.plot(hist_bgr_v)

    plt.show()

    #Now apply equalized hist to bgr
    frm_equalized_b = cv2.equalizeHist(frm_resized[:,:,0])
    frm_equalized_g = cv2.equalizeHist(frm_resized[:,:,1])
    frm_equalized_r = cv2.equalizeHist(frm_resized[:,:,2])
    frm_equalized = np.zeros((frm_resized.shape), dtype=np.uint8)
    frm_equalized[:,:,0] = frm_equalized_b
    frm_equalized[:, :, 1] = frm_equalized_g
    frm_equalized[:, :, 2] = frm_equalized_r
    cv2.imshow("frame resized bgr", frm_resized)
    cv2.imshow("frame equalized bgr", frm_equalized)
    cv2.waitKey(0)

    #  Now plot histograms of equalized figure
    hist_bgr_equalized_b = cv2.calcHist([frm_equalized],[0],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist_bgr_equalized_b)
    plt.show()



    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hist_equalization_bgr():
    '''We will apply histogram equalization for chanells b,g,r, plot histograms afterwards,
        combine into a new frame, and show the frame
    '''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg",\
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    #rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1]//7*6, frm.shape[0]//7*6 ))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)
    frm_gray = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2GRAY)

    # Now equalize different channels in bgr
    frm_eq_b = cv2.equalizeHist(frm_resized[:, :, 0])
    frm_eq_g = cv2.equalizeHist(frm_resized[:, :, 1])
    frm_eq_r = cv2.equalizeHist(frm_resized[:, :, 2])
    # Now combine them into new figure
    frm_eq = np.zeros(frm_resized.shape, dtype=np.uint8)
    frm_eq[:, :, 0] = frm_eq_b
    frm_eq[:, :, 1] = frm_eq_g
    frm_eq[:, :, 2] = frm_eq_r
    #Now show the frame
    cv2.imshow("Equalized", frm_eq)
    cv2.imshow("frame", frm_resized)
    cv2.waitKey(0)

    #Now compute the histograms of equalized and original frames
    hist_eq_b = cv2.calcHist([frm_eq], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist_eq_g = cv2.calcHist([frm_eq], [1], mask=None, histSize=[256], ranges=[0, 256])
    hist_eq_r = cv2.calcHist([frm_eq], [2], mask=None, histSize=[256], ranges=[0, 256])
    #Plot the histograms
    plt.subplot(231), plt.plot(hist_eq_b)
    plt.subplot(232), plt.plot(hist_eq_g)
    plt.subplot(233), plt.plot(hist_eq_r)

    # Now compute the histograms of the original frame
    hist_b = cv2.calcHist([frm_resized], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist_g = cv2.calcHist([frm_resized], [1], mask=None, histSize=[256], ranges=[0, 256])
    hist_r = cv2.calcHist([frm_resized], [2], mask=None, histSize=[256], ranges=[0, 256])
    #Plot the histograms of original frame
    plt.subplot(234), plt.plot(hist_b)
    plt.subplot(235), plt.plot(hist_g)
    plt.subplot(236), plt.plot(hist_r)
    plt.show()

def hist_equalization_hsv():
    '''We transform frame into hsv, equalize it there, and transform back to bgr'''

    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg", \
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    # rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1] // 7 * 6, frm.shape[0] // 7 * 6))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)

    # Now equalize the hsv histogram
    frm_eq_h = cv2.equalizeHist(frm_hsv[:, :, 0])
    frm_eq_s = cv2.equalizeHist(frm_hsv[:, :, 1])
    frm_eq_v = cv2.equalizeHist(frm_hsv[:, :, 2])

    #Combine them together
    frm_hsv_eq = np.zeros(frm_hsv.shape, dtype=np.uint8)
    frm_hsv_eq[:, :, 0] = frm_eq_h
    frm_hsv_eq[:, :, 1] = frm_eq_s
    frm_hsv_eq[:, :, 2] = frm_eq_v

    #Transform into bgr channels
    frm_bgr_eq = cv2.cvtColor(frm_hsv_eq, cv2.COLOR_HSV2BGR)

    # show
    cv2.imshow("bgr equalized through hsv", frm_bgr_eq)
    cv2.waitKey(0)

    # Now equalize only the h channel
    frm_hsv_eq = frm_hsv[:,:,:]
    frm_hsv_eq[:,:,0] = frm_eq_h

    #transform to bgr
    frm_bgr_eq = cv2.cvtColor(frm_hsv_eq, cv2.COLOR_HSV2BGR)

    # show
    cv2.imshow("bgr equalized through h only of hsv", frm_bgr_eq)
    cv2.imshow("frame", frm_resized)
    cv2.waitKey(0)

def bgr2hsv2bgr():
    '''We transform bgr to hsv to bgr'''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg", \
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    # rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1] // 7 * 6, frm.shape[0] // 7 * 6))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)

    #Transform into bgr channels
    frm_bgr_2 = cv2.cvtColor(frm_hsv, cv2.COLOR_HSV2BGR)

    # show
    cv2.imshow("bgr from hsv", frm_bgr_2)
    cv2.imshow("bgr", frm_resized)
    cv2.waitKey(0)

def h_hist():
    '''We transform frame into hsv, equalize it there, and transform back to bgr'''

    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg", \
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    # rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1] // 7 * 6, frm.shape[0] // 7 * 6))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)


    # Now compute the h - histogram and plot it
    hist_h = cv2.calcHist([frm_hsv], [0], mask=None, histSize=[256], ranges=[0, 256])
    plt.subplot(211), plt.plot(hist_h)

    # Now equalize the h-channel
    frm_eq_h = cv2.equalizeHist(frm_hsv[:,:,0])
    print(frm_eq_h.shape)

    # now compute the new histogram
    hist_eq_h = cv2.calcHist([frm_eq_h], [0], None, [256], [0, 256])
    # now plot it
    plt.subplot(212), plt.plot(hist_eq_h)
    plt.show()

    # Now cook up a new frame from equlized h channel

    print(frm_hsv.shape)
    frm_hsv_eq = np.zeros(frm_hsv.shape, dtype=np.uint8)

    frm_hsv_eq[:, :, 0] = frm_eq_h
    frm_hsv_eq[:, :, 1] = frm_hsv[:, :, 1]
    frm_hsv_eq[:, :, 2] = frm_hsv[:, :, 2]

    frm_bgr = cv2.cvtColor(frm_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frm_bgr yzhe isporchennyj', frm_bgr)
    cv2.waitKey(0)



    #Show the difference between old h and equalized h channels
    cv2.imshow("old h channel", frm_hsv[:,:,0])
    cv2.imshow("new h channel", frm_eq_h)
    print(np.max(frm_hsv_eq-frm_hsv))
    print(np.max(frm_hsv - frm_hsv_eq))
    cv2.waitKey(0)



    #Transform into bgr channels
    frm_bgr_eq = cv2.cvtColor(frm_hsv_eq, cv2.COLOR_HSV2BGR)
    frm_bgr = cv2.cvtColor(frm_hsv, cv2.COLOR_HSV2BGR)

    # show
    cv2.imshow("bgr equalized through hsv", frm_bgr_eq)
    cv2.imshow("bgr non equalized through hsv", frm_bgr)
    cv2.imshow("frame", frm_resized)
    cv2.waitKey(0)

def prisvaivaniya_v_np_arrays():
    '''Check chto za fignya s prisvaivaniyami v slice'''
    a = np.zeros((2,2,3), np.uint8)
    print(a)

    b = a[:,:,:]
    print(b)
    b[:,:,0] = np.array([[1,2],[3,4]],np.uint8)
    print("b", b)
    print("a", a)

def prisvaivaniya_v_np_arrays_2():
    ''' prodolzhaem chekat' fignyu s prisvaivaniyami v np.arrays '''
    a = np.array([1,2,3],np.uint8)
    b = a
    c = a[:]
    d = a.copy()
    print("a: ", a)
    print("b: ", b)
    print("c: ", c)
    print("d: ", d)

    a[0]=10
    print("a: ", a)
    print("b: ", b)
    print("c: ", c)
    print("d: ", d)
    # Blyad', a kak zhe kopirovat znacheniya, a ne ssylki?

def prisvaivaniya_v_np_arrays_3():
    ''' prodolzhaem chekat fignyu s prisvaivaniyami v np.arrays '''
    a = np.array([ [[0,0,0], [0,0,0]], [[0,0,0], [0,1,0]]] ,np.uint8)
    print(a)
    print("a", a.shape)

    b = np.ones(a.shape, np.uint8)
    print("b", b)

    c = np.ones((2,2), np.uint8)
    print("c", c)

    b[:, :, 0] = c
    print("b", b)
    print("c", c)

    c = c*2

    print("b", b)
    print("c", c)

    b[:,:,0] = b[:,:,0] + 33* np.ones((2,2),np.uint8)

    print("b", b)
    print("c", c)

def prisvaivaniya_v_np_arrays_4():
    '''Prodolzhaem s prisvaivaniyami v np.arrays'''
    a=np.array([[1,2],[3,4]])
    print("a", a)
    print(a.shape)

    b = np.array([3,3])
    print("b", b)
    print(b.shape)

    c = np.array([[3], [3]])
    print("c", c)
    print(c.shape)

    a[:,0] = b

    print("a", a)
    print("b", b)

def hist_eq_s():
    '''We transform frame into hsv, equalize it there by s, and transform back to bgr
        Note: histogramit' nyzhno tol'ko po saturation, i ne po Hue i ne po value
    '''

    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg", \
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[6]
    frm = cv2.imread(file)
    # rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1] // 7 * 6, frm.shape[0] // 7 * 6))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)

    # Now equalize the hsv histogram
    # frm_eq_h = cv2.equalizeHist(frm_hsv[:, :, 0])
    frm_eq_s = cv2.equalizeHist(frm_hsv[:, :, 1])
    frm_eq_v = cv2.equalizeHist(frm_hsv[:, :, 2])

    #Combine them together
    frm_hsv_eq = np.zeros(frm_hsv.shape, dtype=np.uint8)
    frm_hsv_eq[:, :, 0] = frm_hsv[:, :, 0]
    frm_hsv_eq[:, :, 1] = frm_eq_s
    frm_hsv_eq[:, :, 2] = frm_hsv[:, :, 2]

    #Transform into bgr channels
    frm_bgr_eq = cv2.cvtColor(frm_hsv_eq, cv2.COLOR_HSV2BGR)
    frm_bgr = cv2.cvtColor(frm_hsv, cv2.COLOR_HSV2BGR)

    # show
    cv2.imshow("bgr equalized through saturation in hsv", frm_bgr_eq)
    cv2.waitKey(0)

    cv2.imshow("bgr not equalized", frm_bgr)
    cv2.imshow("frame", frm_resized)
    cv2.waitKey(0)

def call_eq_fun():
    '''We call the equalization function equalize_in_s
    '''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg",\
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    #rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1]//7*6, frm.shape[0]//7*6 ))
    # transform to HSV

    equalize_in_s(frm_resized,1)
    return

def video_equalized():
    ''' Play the original video, equalize it and write it into output file '''

    # open the file
    input_file = "input_video.avi"
    output_file= "output_equalized_video.avi"

    # read the first frame
    cap = cv2.VideoCapture(input_file)
    # print(cap.isOpened())

    ret, frm = cap.read()
    frm_fringed = frm[60:-60,:,:]


    # create the output file format
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
    frames_per_second = 20.0
    image_size = (frm_fringed.shape[1], frm_fringed.shape[0])
    writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size )


    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]

        cv2.imshow("Video", frm)
        frm_eq = equalize_in_s(frm_fringed)
        cv2.putText(frm_eq, "Count: " + str(frm_count), (10, 100), 0, 2, [255, 255, 255], 5)
        cv2.imshow("Video equalized", frm_eq)
        writer.write(frm_eq)

        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 30
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocessing_of_some_frames():
    '''Preprocesing of some frames from the video'''
    list_of_frames = ["frame50.jpg", "frame109.jpg", "frame164.jpg", "frame225.jpg", "frame266.jpg",\
                      "frame362.jpg", "frame444.jpg"]
    file = list_of_frames[0]
    frm = cv2.imread(file)
    #rescaling, otherwise imshow() will crop a part of the image in split_in_3_ch
    frm_resized = cv2.resize(frm, (frm.shape[1]//7*6, frm.shape[0]//7*6 ))
    # transform to HSV
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)
    frm_gray = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2GRAY)



    cv2.imshow("frame", frm_resized)
    cv2.imshow("HSV", frm_hsv)
    cv2.imshow("Gray", frm_gray)

    frm_split = split_into_3_ch(frm_resized)

    # cv2.imshow("SPlit again", frm_split)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_blur():
    ''' We apply filters on selected images
    '''

    # blurparam
    blurparam=5

    # Select the frame from the collection
    list_of_frames = Param.list_of_frames
    file = list_of_frames[0]
    frm = cv2.imread(file)

    #Blurr, then equalize in s (HSV)
    frm_blurred = cv2.blur(frm,(blurparam, blurparam))
    frm_bleq = equalize_in_s(frm_blurred)

    # Equalize in s (HSV), then blurr
    frm_eq = equalize_in_s(frm)
    frm_eqbl = cv2.blur(frm_eq, (blurparam, blurparam))


    cv2.imshow('frame', frm)
    cv2.imshow('frame blurred', frm_blurred)
    cv2.imshow('frame equalized', frm_eq)
    cv2.imshow('frame first blurred then equalized', frm_bleq)
    cv2.imshow('frame first equalized then blurred', frm_eqbl)
    cv2.waitKey(0)

def apply_Gausianblur():
    ''' We apply filters on selected images
    '''

    # blurparam
    blurparam=25

    # Select the frame from the collection
    list_of_frames = Param.list_of_frames
    file = list_of_frames[0]
    frm = cv2.imread(file)

    #Blurr, then equalize in s (HSV)
    frm_blurred = cv2.GaussianBlur(frm,(blurparam, blurparam), 0)
    frm_bleq = equalize_in_s(frm_blurred)

    # Equalize in s (HSV), then blurr
    frm_eq = equalize_in_s(frm)
    frm_eqbl = cv2.GaussianBlur(frm_eq, (blurparam, blurparam), 0)


    cv2.imshow('frame', frm)
    cv2.imshow('frame blurred', frm_blurred)
    cv2.imshow('frame equalized', frm_eq)
    cv2.imshow('frame first blurred then equalized', frm_bleq)
    cv2.imshow('frame first equalized then blurred', frm_eqbl)


    # binary = cv2.threshold(frm_eqbl, 140, 255, cv2.THRESH_BINARY)[1]
    frm_gray = cv2.cvtColor(frm_eq, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(frm_gray, 140, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("Binary ", binary)
    cv2.waitKey(0)

def apply_medianblur():
    ''' We apply filters on selected images
    '''

    # blurparam
    blurparam=11

    # Select the frame from the collection
    list_of_frames = Param.list_of_frames
    file = list_of_frames[0]
    frm = cv2.imread(file)

    #Blurr, then equalize in s (HSV)
    frm_blurred = cv2.medianBlur(frm, blurparam)
    frm_bleq = equalize_in_s(frm_blurred)

    # Equalize in s (HSV), then blurr
    frm_eq = equalize_in_s(frm)
    frm_eqbl = cv2.medianBlur(frm_eq, blurparam)


    cv2.imshow('frame', frm)
    cv2.imshow('frame blurred', frm_blurred)
    cv2.imshow('frame equalized', frm_eq)
    cv2.imshow('frame first blurred then equalized', frm_bleq)
    cv2.imshow('frame first equalized then blurred', frm_eqbl)


    # binary = cv2.threshold(frm_eqbl, 140, 255, cv2.THRESH_BINARY)[1]
    frm_gray = cv2.cvtColor(frm_eq, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(frm_gray, 140, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("Binary ", binary)
    cv2.waitKey(0)

def apply_bilateralFilter():
    ''' We apply filters on selected images
    '''

    # parameters of the filter
    param1 = 9
    param2 = 75

    # Select the frame from the collection
    list_of_frames = Param.list_of_frames
    file = list_of_frames[0]
    frm = cv2.imread(file)

    #Blurr, then equalize in s (HSV)
    frm_blurred = cv2.bilateralFilter(frm, param1, param2, param2 )
    frm_bleq = equalize_in_s(frm_blurred)

    # Equalize in s (HSV), then blurr
    frm_eq = equalize_in_s(frm)
    frm_eqbl = cv2.bilateralFilter(frm_eq, param1, param2, param2)


    cv2.imshow('frame', frm)
    cv2.imshow('frame blurred', frm_blurred)
    cv2.imshow('frame equalized', frm_eq)
    cv2.imshow('frame first blurred then equalized', frm_bleq)
    cv2.imshow('frame first equalized then blurred', frm_eqbl)


    # binary = cv2.threshold(frm_eqbl, 140, 255, cv2.THRESH_BINARY)[1]
    frm_gray = cv2.cvtColor(frm_eq, cv2.COLOR_BGR2GRAY)
    thresh, binary = cv2.threshold(frm_gray, 140, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("Binary ", binary)
    cv2.waitKey(0)

def trackbars_and_ranges_red():
    '''We pick up manually the ranges for masks. We use trackbars
    '''

    list_of_frames = Param.list_of_frames
    file = list_of_frames[6]
    frm = cv2.imread(file)

    # # Now preprocessing: median blur and equalization
    # frm_bl = cv2.medianBlur(frm, 5)
    # frm_bleq = equalize_in_s(frm_bl)

    frm_proc = preprocessing(frm)

    # Transform into hsv
    frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)

    # # display what we have now
    # cv2.imshow('frm', frm)
    # cv2.imshow('frm bleq', frm_bleq)
    # cv2.imshow('frm hsv', frm_hsv)
    # split_into_3_ch(frm_hsv, 1)
    # cv2.waitKey(0)

    # creation trackbars
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('LH', 'Trackbars', Param.red_lower_HSV[0], 179, nothing)
    cv2.createTrackbar('UH', 'Trackbars', Param.red_upper_HSV[0], 179, nothing)
    cv2.createTrackbar('LS', 'Trackbars', Param.red_lower_HSV[1], 255, nothing)
    cv2.createTrackbar('US', 'Trackbars', Param.red_upper_HSV[1], 255, nothing)
    cv2.createTrackbar('LV', 'Trackbars', Param.red_lower_HSV[2], 255, nothing)
    cv2.createTrackbar('UV', 'Trackbars', Param.red_upper_HSV[2], 255, nothing)

    # tool for choosing ranges
    key = ord(' ')

    while(True):
        # get trackbars positions
        lh = cv2.getTrackbarPos('LH', 'Trackbars')
        uh = cv2.getTrackbarPos('UH', 'Trackbars')
        ls = cv2.getTrackbarPos('LS', 'Trackbars')
        us = cv2.getTrackbarPos('US', 'Trackbars')
        lv = cv2.getTrackbarPos('LV', 'Trackbars')
        uv = cv2.getTrackbarPos('UV', 'Trackbars')

        # create a mask
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv2.inRange(frm_hsv, lower, upper)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frm)
        key = cv2.waitKey(30)
        if key == ord('1') or key == 27:
            break

    cv2.waitKey(0)

def video_red_mask(write_to_file = 0):
    '''
    Function plays a video with a red mask
    :param write_to_file: if 1, then we produce an output video
    :return:
    '''

    # Open the file
    input_file = "input_video.avi"
    if (write_to_file == 1):
        output_file = "output_video_red_mask.avi"

    # read the first frame
    cap = cv2.VideoCapture(input_file)
    # print(cap.isOpened())

    ret, frm = cap.read()
    frm_fringed = frm[60:-60,:,:]

    # create the output file format
    if (write_to_file == 1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
        frames_per_second = 20.0
        image_size = (frm_fringed.shape[1], frm_fringed.shape[0])
        writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size )

    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]

        # processing the frame, transform it to HSV, and apply the red mask to it
        frm_proc = preprocessing(frm_fringed,0)
        # BGR to HSV
        frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)
        # create the red mask
        lower = Param.red_lower_HSV
        upper = Param.red_upper_HSV
        mask = cv2.inRange(frm_hsv, lower, upper)
        mask_median = cv2.medianBlur(mask, 7)
        # applying the mask to original BGR
        result = cv2.bitwise_and(frm_fringed, frm_fringed, mask = mask_median)

        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5)
        cv2.imshow("Video", frm_fringed)
        cv2.imshow("Median Mask", mask_median)
        cv2.imshow("Result", result)



        if (write_to_file == 1):
            writer.write(result)

        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 30
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    if (write_to_file == 1):
        writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def trackbars_and_ranges_green():
    '''We pick up manually the ranges for masks. We use trackbars
    '''

    list_of_frames = Param.list_of_frames
    file = list_of_frames[Param.number]
    frm = cv2.imread(file)

    # # Now preprocessing: median blur and equalization
    # frm_bl = cv2.medianBlur(frm, 5)
    # frm_bleq = equalize_in_s(frm_bl)

    frm_proc = preprocessing(frm, Param.preproc)

    # Transform into hsv
    frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)

    # # display what we have now
    # cv2.imshow('frm', frm)
    # cv2.imshow('frm bleq', frm_bleq)
    # cv2.imshow('frm hsv', frm_hsv)
    # split_into_3_ch(frm_hsv, 1)
    # cv2.waitKey(0)

    # creation trackbars
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('LH', 'Trackbars', Param.green_lower_HSV[0], 179, nothing)
    cv2.createTrackbar('UH', 'Trackbars', Param.green_upper_HSV[0], 179, nothing)
    cv2.createTrackbar('LS', 'Trackbars', Param.green_lower_HSV[1], 255, nothing)
    cv2.createTrackbar('US', 'Trackbars', Param.green_upper_HSV[1], 255, nothing)
    cv2.createTrackbar('LV', 'Trackbars', Param.green_lower_HSV[2], 255, nothing)
    cv2.createTrackbar('UV', 'Trackbars', Param.green_upper_HSV[2], 255, nothing)

    # tool for choosing ranges
    key = ord(' ')

    while(True):
        # get trackbars positions
        lh = cv2.getTrackbarPos('LH', 'Trackbars')
        uh = cv2.getTrackbarPos('UH', 'Trackbars')
        ls = cv2.getTrackbarPos('LS', 'Trackbars')
        us = cv2.getTrackbarPos('US', 'Trackbars')
        lv = cv2.getTrackbarPos('LV', 'Trackbars')
        uv = cv2.getTrackbarPos('UV', 'Trackbars')

        # create a mask
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv2.inRange(frm_hsv, lower, upper)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frm)
        cv2.imshow("3 channels of HSV", split_into_3_ch(frm_hsv) )
        cv2.imshow("3 channels of BGR", split_into_3_ch(frm_proc))
        key = cv2.waitKey(30)
        if key == ord('1') or key == 27:
            break

    cv2.waitKey(0)

def trackbars_and_ranges_yellow():
    '''We pick up manually the ranges for masks. We use trackbars
    '''

    list_of_frames = Param.list_of_frames
    file = list_of_frames[Param.number]
    frm = cv2.imread(file)

    # # Now preprocessing: median blur and equalization
    # frm_bl = cv2.medianBlur(frm, 5)
    # frm_bleq = equalize_in_s(frm_bl)

    frm_proc = preprocessing(frm, Param.preproc)

    # Transform into hsv
    frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)

    # # display what we have now
    # cv2.imshow('frm', frm)
    # cv2.imshow('frm bleq', frm_bleq)
    # cv2.imshow('frm hsv', frm_hsv)
    # split_into_3_ch(frm_hsv, 1)
    # cv2.waitKey(0)

    # creation trackbars
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('LH', 'Trackbars', Param.yellow_lower_HSV[0], 179, nothing)
    cv2.createTrackbar('UH', 'Trackbars', Param.yellow_upper_HSV[0], 179, nothing)
    cv2.createTrackbar('LS', 'Trackbars', Param.yellow_lower_HSV[1], 255, nothing)
    cv2.createTrackbar('US', 'Trackbars', Param.yellow_upper_HSV[1], 255, nothing)
    cv2.createTrackbar('LV', 'Trackbars', Param.yellow_lower_HSV[2], 255, nothing)
    cv2.createTrackbar('UV', 'Trackbars', Param.yellow_upper_HSV[2], 255, nothing)

    # tool for choosing ranges
    key = ord(' ')

    while(True):
        # get trackbars positions
        lh = cv2.getTrackbarPos('LH', 'Trackbars')
        uh = cv2.getTrackbarPos('UH', 'Trackbars')
        ls = cv2.getTrackbarPos('LS', 'Trackbars')
        us = cv2.getTrackbarPos('US', 'Trackbars')
        lv = cv2.getTrackbarPos('LV', 'Trackbars')
        uv = cv2.getTrackbarPos('UV', 'Trackbars')

        # create a mask
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv2.inRange(frm_hsv, lower, upper)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frm)
        cv2.imshow("3 channels of HSV", split_into_3_ch(frm_hsv) )
        cv2.imshow("3 channels of BGR", split_into_3_ch(frm_proc))
        key = cv2.waitKey(30)
        if key == ord('1') or key == 27:
            break

    cv2.waitKey(0)

def trackbars_and_ranges_green2():
    '''We pick up manually the ranges for masks. We use trackbars
    '''

    list_of_frames = Param.list_of_frames
    file = list_of_frames[Param.number]
    frm = cv2.imread(file)

    # # Now preprocessing: median blur and equalization
    # frm_bl = cv2.medianBlur(frm, 5)
    # frm_bleq = equalize_in_s(frm_bl)

    frm_proc = preprocessing(frm, Param.preproc)

    # Transform into hsv
    frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)

    # # display what we have now
    # cv2.imshow('frm', frm)
    # cv2.imshow('frm bleq', frm_bleq)
    # cv2.imshow('frm hsv', frm_hsv)
    # split_into_3_ch(frm_hsv, 1)
    # cv2.waitKey(0)

    # creation trackbars
    def nothing(x):
        pass
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('LH', 'Trackbars', Param.green2_lower_HSV[0], 179, nothing)
    cv2.createTrackbar('UH', 'Trackbars', Param.green2_upper_HSV[0], 179, nothing)
    cv2.createTrackbar('LS', 'Trackbars', Param.green2_lower_HSV[1], 255, nothing)
    cv2.createTrackbar('US', 'Trackbars', Param.green2_upper_HSV[1], 255, nothing)
    cv2.createTrackbar('LV', 'Trackbars', Param.green2_lower_HSV[2], 255, nothing)
    cv2.createTrackbar('UV', 'Trackbars', Param.green2_upper_HSV[2], 255, nothing)

    # tool for choosing ranges
    key = ord(' ')

    while(True):
        # get trackbars positions
        lh = cv2.getTrackbarPos('LH', 'Trackbars')
        uh = cv2.getTrackbarPos('UH', 'Trackbars')
        ls = cv2.getTrackbarPos('LS', 'Trackbars')
        us = cv2.getTrackbarPos('US', 'Trackbars')
        lv = cv2.getTrackbarPos('LV', 'Trackbars')
        uv = cv2.getTrackbarPos('UV', 'Trackbars')

        # create a mask
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        mask = cv2.inRange(frm_hsv, lower, upper)

        cv2.imshow('mask', mask)
        cv2.imshow('frame', frm)
        cv2.imshow("3 channels of HSV", split_into_3_ch(frm_hsv) )
        cv2.imshow("3 channels of BGR", split_into_3_ch(frm_proc))
        key = cv2.waitKey(30)
        if key == ord('1') or key == 27:
            break

    cv2.waitKey(0)

def video_green_mask(write_to_file = 0):
    '''
    Function plays a video with a green mask
    :param write_to_file: if 1, then we produce an output video
    :return:
    '''

    # Open the file
    input_file = "input_video.avi"
    if (write_to_file == 1):
        output_file = "output_video_green_mask.avi"

    # read the first frame
    cap = cv2.VideoCapture(input_file)
    # print(cap.isOpened())

    ret, frm = cap.read()
    frm_fringed = frm[60:-60,:,:]

    # create the output file format
    if (write_to_file == 1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
        frames_per_second = 20.0
        image_size = (frm_fringed.shape[1], frm_fringed.shape[0])
        writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size )

    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]

        # processing the frame, transform it to HSV, and apply the red mask to it
        frm_proc = preprocessing(frm_fringed, 1)
        # BGR to HSV
        frm_hsv = cv2.cvtColor(frm_proc, cv2.COLOR_BGR2HSV)
        # create the red mask
        lower = Param.green_lower_HSV
        upper = Param.green_upper_HSV
        mask = cv2.inRange(frm_hsv, lower, upper)
        mask_median = cv2.medianBlur(mask, 7)
        # applying the mask to original BGR
        result = cv2.bitwise_and(frm_fringed, frm_fringed, mask = mask_median)

        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5)
        cv2.imshow("Video", frm_fringed)
        cv2.imshow("Median", mask)
        cv2.imshow("Result", result)



        if (write_to_file == 1):
            writer.write(result)

        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 30
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    if (write_to_file == 1):
        writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_green_and_red_masks(write_to_file = 0, write_mask = 0):
    '''
    Function plays a video with a green mask
    :param write_to_file: if 1, then we produce an output video, otherwise not
    write_mask:
        1: it will write the mask (white and black)
        else: it will write the resulting figures in color
    :return:
    '''

    # Open the file
    input_file = "input_video.avi"
    if (write_to_file == 1):
        output_file = "output_video_green_and_red_mask.avi"

    # read the first frame
    cap = cv2.VideoCapture(input_file)
    # print(cap.isOpened())

    ret, frm = cap.read()
    frm_fringed = frm[60:-60,:,:]

    # create the output file format
    if (write_to_file == 1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
        frames_per_second = 20.0
        image_size = (frm_fringed.shape[1], frm_fringed.shape[0])
        if (write_mask!=1):
            writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size )
        else:
            writer = cv2.VideoWriter(output_file, fourcc, frames_per_second, image_size, 0)

    frm_count = 0
    key=ord(' ')

    while(ret):
        frm_fringed = frm[60:-60,:,:]

        #Treating red mask
        # processing the frame, transform it to HSV, and apply the red mask to it
        frm_proc_red = preprocessing(frm_fringed,0)
        # BGR to HSV
        frm_hsv_red = cv2.cvtColor(frm_proc_red, cv2.COLOR_BGR2HSV)
        # create the red mask
        lower_red = Param.red_lower_HSV
        upper_red = Param.red_upper_HSV
        mask_red = cv2.inRange(frm_hsv_red, lower_red, upper_red)
        mask_red_median = cv2.medianBlur(mask_red, 7)
        # # applying the mask to original BGR
        # result_red = cv2.bitwise_and(frm_fringed, frm_fringed, mask = mask_red_median)

        #Treating green mask
        # processing the frame, transform it to HSV, and apply the red mask to it
        frm_proc_green = preprocessing(frm_fringed, 1)
        # BGR to HSV
        frm_hsv_green = cv2.cvtColor(frm_proc_green, cv2.COLOR_BGR2HSV)
        # create the red mask
        lower_green = Param.green_lower_HSV
        upper_green = Param.green_upper_HSV
        mask_green = cv2.inRange(frm_hsv_green, lower_green, upper_green)
        # mask_green_dilate = cv2.dilate(mask_green, np.ones((17, 17), np.uint8))
        # mask_green_filter = cv2.erode(mask_dilate, np.ones((5,5),np.uint8))
        #
        # mask_green_filter = cv2.filter2D(mask_green, (np.ones((5,5), np.float32) * 255//25) )
        mask_green_median = cv2.medianBlur(mask_green, 15)

        # Combining the read and green masks
        mask_green_red = cv2.bitwise_or(mask_red_median, mask_green_median)

        # Treating remaining parts. Yellow mask
        # preprocessing the same as for green mask
        lower_yellow = Param.yellow_lower_HSV
        upper_yellow = Param.yellow_upper_HSV
        mask_yellow = cv2.inRange(frm_hsv_green, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask_green_red, mask_yellow)

        # Treating remaining parts. Green2 mask
        # preprocessing the same as for green mask
        lower_green2 = Param.green2_lower_HSV
        upper_green2 = Param.green2_upper_HSV
        mask_green2 = cv2.inRange(frm_hsv_green, lower_green2, upper_green2)
        mask = cv2.bitwise_or(mask_green2, mask)
        mask = cv2.medianBlur(mask, 7)

        # apply the mask to the original BGR
        result = cv2.bitwise_and(frm_fringed, frm_fringed, mask=mask)

        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5)
        cv2.imshow("Video", frm_fringed)
        cv2.imshow("Median", mask)
        cv2.imshow("Result", result)



        if (write_to_file == 1):
            if (write_mask!=1):
                writer.write(result)
            else:
                writer.write(mask)

        if key == ord(' '):
            wait_period = 0
        elif ( key == 27 or key == ord(' ') ):
            break
        else:
            wait_period = 30 # 1
        key = cv2.waitKey(wait_period)
        ret, frm = cap.read()
        frm_count+=1
    cap.release()
    if (write_to_file == 1):
        writer.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Below are Class and functions that are called from other functions

class Param:
    list_of_frames = ["frame0.jpg", "frame3.jpg", "frame5.jpg", "frame50.jpg", "frame109.jpg", "frame164.jpg", "frame178.jpg", \
                      "frame179.jpg", "frame180.jpg", "frame225.jpg",
                      "frame266.jpg", "frame362.jpg", "frame444.jpg", "frame495.jpg", "frame504.jpg"]
    red_lower_HSV = np.array([157, 51, 149], np.uint8)
    red_upper_HSV = np.array([179, 255, 255], np.uint8)

    # # for processing: 0, but it does not work well
    # green_lower_HSV = np.array([0, 52, 140], np.uint8)
    # green_upper_HSV = np.array([80, 121, 255], np.uint8)

    # green mask, works for preprocessing: 1, frame 5
    # green_lower_HSV = np.array([0, 185, 110], np.uint8)
    # green_upper_HSV = np.array([80, 255, 194], np.uint8)
    green_lower_HSV = np.array([22, 121, 116], np.uint8)
    green_upper_HSV = np.array([76, 255, 255], np.uint8)

    yellow_lower_HSV = np.array([21, 0, 183], np.uint8)
    yellow_upper_HSV = np.array([56, 255, 255], np.uint8)

    green2_lower_HSV = np.array([20, 160, 180], np.uint8)
    green2_upper_HSV = np.array([74, 255, 255], np.uint8)

    # the number of frame
    number = 1
    preproc = 1 # the way preprocessing goes

    # green mask, works for preprocessing: 1, frame 6
    # green_lower_HSV = np.array([0, 86, 140], np.uint8)
    # green_upper_HSV = np.array([80, 255, 255], np.uint8)

    error = []

def call_preproc():
    '''We call preprocessing() '''
    file = Param.list_of_frames[2]
    frm = cv2.imread(file)
    cv2.imshow("frm", preprocessing(frm, 1) )
    cv2.waitKey(0)

def equalize_in_s(frm, display = 0):
    ''' Function receives frm in bgr, equalize its histogram in s in hsv, transforms it back to bgr and returns
    '''
    frm_hsv = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
    eq_s = cv2.equalizeHist(frm_hsv[:,:,1])

    # Create array for equalized histograms in s
    frm_eq = np.zeros(frm.shape, dtype = np.uint8)
    frm_eq[:, :, 0] = frm_hsv[:, :, 0]
    frm_eq[:, :, 2] = frm_hsv[:, :, 2]
    frm_eq[:, :, 1] = eq_s

    # Transform to BGR
    frm_bgr = cv2.cvtColor(frm_eq, cv2.COLOR_HSV2BGR)

    if display ==1:
        cv2.imshow("Equalized in s", frm_bgr)
        cv2.imshow("frame in s", frm)
        cv2.waitKey(0)

    return frm_bgr

def preprocessing(frm, option = 0, already_fringed = 1):
    '''
    This function preprocess a frame, and returns a frame
    :param frm: frame
    :param option:
        0 - no preproccesing,
        1 -- median blur then equalization in s, hsv
        2 -- equalization in s, hsv, then median blur
        3 -- equalization in s, hsv
        4 -- median blur
    :param already_fringed:
        1 if black fringes were already removed
        0 if they are still there; and we will remove them
    :return: a new frame
    '''
    if (already_fringed!=1):
        frm_fringed = frm[60:-60, :, :]
    else:
        frm_fringed = frm

    if (option == 0):
        return frm_fringed
    if (option == 1):
        frm_bl = cv2.medianBlur(frm_fringed, 5)
        frm_bleq = equalize_in_s(frm_bl)
        return frm_bleq
    elif (option == 2):
        frm_eq = equalize_in_s(frm_fringed)
        frm_eqbl = cv2.medianBlur(frm_eq, 5)
        return frm_eqbl
    elif (option ==3):
        frm_eq = equalize_in_s(frm_fringed)
        return frm_eq
    elif (option ==4):
        frm_bl = cv2.medianBlur(frm_fringed, 5)
        return frm_bl
    else:
        print('Shel by ty')
        Param.error.append('ERROR: Vyzov iz preprocessing')

def split_into_3_ch(frm, display = 0):
    '''decompose the image into chanells and display
       note that imshow cut a portion of an image, if the latter is too big
       display = 1 -- it will plot the image, otherwise not
    '''

    # create an output frame
    frm_res = np.zeros((frm.shape[0], frm.shape[1] * frm.shape[2]), dtype = np.uint8)
    for i in range(0,frm.shape[2]):
        frm_res[:, i*frm.shape[1]: frm.shape[1]*(i+1)  ] = frm[:,:,i]

    if (display == 1):
        cv2.imshow("3chanells", frm_res)

    return frm_res

if __name__ == '__main__':

    # playvideo()
    # play_and_write_video()

    # creation_of_samples_of_frames()

    # displaying_selected_frames()
    # analysis_of_some_frames()
    # hist_equalization_bgr()
    # hist_equalization_hsv()
    # h_hist()
    # prisvaivaniya_v_np_arrays()
    # prisvaivaniya_v_np_arrays_2()
    # prisvaivaniya_v_np_arrays_3()
    # prisvaivaniya_v_np_arrays_4()
    # bgr2hsv2bgr()
    # hist_eq_s()
    # call_eq_fun()
    # video_equalized()
    # apply_blur()
    # apply_Gausianblur()
    # apply_medianblur()
    # apply_bilateralFilter()
    # trackbars_and_ranges_red()
    # video_red_mask(1) # options: 0  for not creating a video file, 1 for creating a video file

    # trackbars_and_ranges_green()

    # trackbars_and_ranges_yellow()

    # trackbars_and_ranges_green2()

    # video_green_mask() # options: 0  for not creating a video file, 1 for creating a video file

    video_green_and_red_masks(1, 1) # first option: 1 for creating a video file, 0 for not creating file\
                                    # second option: 1 for writing mask (white and black), 0 for writing
                                    # colored figures

    # preprocessing_of_some_frames()
    # call_preproc()

