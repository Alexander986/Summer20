# Literature
# Histogram https://lmcaraig.com/understanding-image-histograms-with-opencv
#     see also https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
#     https://docs.opencv.org/4.3.0/d6/dc7/group__imgproc__hist.html
#     https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
#     https://drive.google.com/file/d/1e95vLF6fhoBmv4CuX6shptVLSBbsmz3e/view
#     https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/


# work with video
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# saving grayscale video
# https://stackoverflow.com/questions/50037063/save-grayscale-video-in-opencv

import cv2
import numpy as np
from matplotlib import pyplot as plt

def myfun():
    input_file = 'input_video.avi'
    output_file = 'output_video.mp4'

    cap = cv2.VideoCapture(input_file)
    ret,frm = cap.read()
    print(frm.shape)

    #fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4

    #writer = cv2.VideoWriter(output_file, fourcc, 20.0, (frm.shape[1],frm.shape[0]) )

    frm_count = 0
    key = None

    while(ret):
        # first pick up several frames and save them
        # bright: 50 109 164 225 266     dark: 362 444
        # N=50
        # if not (N-1 < frm_count < N+1):
        #     frm_count+=1
        #     ret, frm = cap.read()
        #     continue
        # print(frm_count)
        # cv2.imwrite('frame50.jpg', frm)


        # Remove the upper and lower black fringes
        frm_fringed  = frm[60:-60, :, :]

        cv2.putText(frm_fringed, "Count: "+ str(frm_count), (10,100), 0, 2, [255,255,255],5 )
        cv2.imshow("Video", frm_fringed)

        # hist = cv2.calcHist(frm_fringed, [4], None, [256], [0,256])
        # plt.plot(hist) 50 109 164 225 266        dark: 362 444

        #resize
        frm_resized = cv2.resize(frm_fringed, (frm_fringed.shape[1]//4*3, frm_fringed.shape[0]//4*3 ), cv2.INTER_LINEAR)

        #transform into HSV and equalize the histogram
        H, S, V = cv2.split(cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
        cv2.imshow("equ HSV", eq_image)

        #transform to gray picture
        frm_gray = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray figure" ,frm_gray)

        # hist = cv2.calcHist(frm_gray, [0], None, [256],[0,256])
        # plt.plot(hist)
        # plt.show()

        frm_gray_eq = cv2.equalizeHist(frm_gray)
        cv2.imshow("gray equalized hist", frm_gray_eq)




        frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)
        frm_lab = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2LAB)

        frm = cv2.Sobel(frm, cv2.CV_8U,0,1,3,3);
        cv2.putText(frm_resized, 'frame: ' + str(frm_count), (10,100),0,2,[255,255,255],5)
        cv2.imshow('Video hsv', frm_hsv)
        cv2.imshow('Video frame', frm_resized)
        # cv2.waitKey(0)

        # cv2.waitKey(0)

        #writer.write(frm)

        if key == ord(' '):
            wait_period = 0
        elif key == ord('1'):
            break
        else:
            wait_period = 1

        key = cv2.waitKey(wait_period)

        # decomp_3_ch(frm_hsv)
        #cv2.waitKey(0)
        ret,frm = cap.read()

        frm_count+=1
    cap.release()
    #writer.release()
    cv2.waitKey(0)
    return


def myfun2():
    '''will create a simple mask'''

    #read the image
    input_file = 'frame362.jpg'
    frm = cv2.imread(input_file)

    # remove fringes from the figure
    frm_fringed = frm[60:-60, :, :]

    # resize
    frm_resized = cv2.resize(frm_fringed, (frm_fringed.shape[1]//4*3, frm_fringed.shape[0]//4*3) )

    #plot the image
    cv2.imshow("Frame"+input_file, frm_resized)
    print(frm_resized.shape)
    cv2.waitKey(0)


    # transform into hsv
    frm_hsv = cv2.cvtColor(frm_resized, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV frame", frm_hsv)
    cv2.waitKey(0)

    # split into three chanells
    # frm_splited_ch = decomp_3_ch(frm_hsv)

    # cv2.imshow("Splited chanells", frm_splited_ch)
    # cv2.waitKey(0)

    green_mask_HSV = cv2.inRange(frm_hsv, Param.green_min_HSV, Param.green_max_HSV)
    result_HSV = cv2.bitwise_and( frm_resized, frm_resized, mask = green_mask_HSV )

    cv2.imshow("Mask green", green_mask_HSV)
    cv2.imshow("Result mask green hsv", result_HSV)
    cv2.waitKey(0)


    return

def myfun3():
    '''will create a video with a simple mask'''

    input_file = 'input_video.avi'
    output_file = 'output_video_mask_1.mp4'

    # read the first frame and set up the video writing parameters
    cap = cv2.VideoCapture(input_file)
    ret, frm = cap.read()
    # remove fringes
    frm_fringed = frm[60:-60, :, :]
    #print(frm.shape)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # for avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4

    # 3 channel output
    writer = cv2.VideoWriter(output_file, fourcc, 20.0, (frm_fringed.shape[1], frm_fringed.shape[0]) )

    frm_count = 0
    key = None

    while (ret):
        frm_fringed = frm[60:-60, :, :]

        # transform into hsv
        frm_hsv = cv2.cvtColor(frm_fringed, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV frame", frm_hsv)
        # cv2.waitKey(0)

        # split into three chanells
        # frm_splited_ch = decomp_3_ch(frm_hsv)

        # cv2.imshow("Splited chanells", frm_splited_ch)
        # cv2.waitKey(0)

        # create a general mask
        green_mask_HSV = cv2.inRange(frm_hsv, Param.green_min_HSV, Param.green_max_HSV)
        result_HSV = cv2.bitwise_and(frm_fringed, frm_fringed, mask=green_mask_HSV)

        cv2.imshow("Mask green", green_mask_HSV)
        cv2.imshow("Result mask green hsv", result_HSV)

        # in order to save the video, transform the 1 channel picture to 3 channels picture
        green_mask_HSV_3ch = cv2.cvtColor(green_mask_HSV, cv2.COLOR_GRAY2BGR)
        writer.write(green_mask_HSV_3ch)

        if key == ord(' '):
            wait_period = 0
        elif key == ord('1'):
            break
        else:
            wait_period = 1

        key = cv2.waitKey(wait_period)

        ret,frm = cap.read()
        frm_count+=1

    cap.release()
    writer.release()

    return







class Param():
    '''This class keeps some variables'''
    green_BGR = [250, 110, 60] #[100, 110, 200]
    threshold = 150 # 100
    green_HSV = cv2.cvtColor(np.uint8([[green_BGR]] ), cv2.COLOR_BGR2HSV )[0][0]
    green_min_HSV = np.array([green_HSV[0] - threshold, green_HSV[1] - threshold, green_HSV[2] - threshold] )
    green_max_HSV = np.array([green_HSV[0] + threshold, green_HSV[1] + threshold, green_HSV[2] + threshold] )


def mymask():
    return


def decomp_3_ch(frm):
    '''Split of the figure by channels'''

    frm_dec = np.zeros((frm.shape[0], frm.shape[1]*frm.shape[2]), dtype=np.uint8)

    for i in range(0, frm.shape[2]):
        frm_dec[:, i*frm.shape[1] : frm.shape[1]*(i+1)] = frm[:,:,i]
    cv2.imshow('Video decomposed ch', frm_dec)
    # cv2.waitKey(1000)
    # print(frm.shape)
    # print(frm_dec.shape)
    # print(frm)
    return frm_dec




if __name__ == '__main__':
    # myfun()
    # myfun2()
    myfun3()