import cv2
import numpy as np
import os
import time


def TemplateMatching():
    filename = 'plan.png'
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, None, fx = 1/6, fy=1/6)
    # cv2.imshow('img', img_resized)
    # cv2.waitKey(0)

    # index = 0 xrenovo rabotaet
    # index = 1 0.8 (4)
    index = 2


    tmp_arr = load_templates()
    res = cv2.matchTemplate(img_gray, tmp_arr[index], cv2.TM_CCOEFF_NORMED)

    cv2.imwrite('res.png', res)

    w, h = tmp_arr[index].shape[::-1]

    print(np.max(res))
    loc = np.where(res >0.99 * np.max(res))
    print(len(loc[0]))
    print(loc)

    for pt in zip(*loc[::-1]):
        print(pt)
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

    cv2.imwrite('out.jpg', img)

    # cv2.imshow("img", img)
    # cv2.waitKey(100)

    # cv2.imshow('img', img)
    # cv2.imshow('res', res)
    # cv2.waitKey(100)

    # loc = np.where(res >= 0.8)


    return


    cv2.imshow('res', res)
    cv2.waitKey(0)

    w, h = tmp_arr[0][::-1].shape

    loc = np.where(res>=0.8 )
    print(loc)

    print(zip(*[loc[::-1]]))
    for pt in zip(*[loc[::-1]]):
        print(pt)
        cv2.rectangle(img, pt, (pt[0] + w, pt[0] + h), (0, 255, 0), 3 )

def TemplateMatching2():

    # download original image
    filename = 'plan.png'
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # generate colors for each template
    get_Colors()

    # load templates with their rotations
    arr_rotated = get_rotated_templates()


    thresh = 0.9

    for index in range(len(arr_rotated)):
        if (index == 7):
            thresh=0.95
        elif(index == 2):
            thresh = 0.93
        #iterate over all templates
        for j in range(len(arr_rotated[index])):
            # iterate over all rotated versions of template

            # cv2.imshow('rotated by'+str(j*90), arr_rotated[index][j])

            res = cv2.matchTemplate(img_gray, arr_rotated[index][j], cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh * np.max(res) )

            # adding points in rectangles to the copy of original img

            w, h = arr_rotated[index][j].shape[::-1]
            for pt in zip(*loc[::-1]):
                # print(type(Param.colors[index][0]))
                # print(len((0, 255, 0)))
                # print(type(Param.colors[index][0]))

                mycolor = [int(Param.colors[index][0]), int(Param.colors[index][1]), int(Param.colors[index][2])]
                # mycolor = (Param.colors[index]).astype(int) # does not work. Why?
                cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), mycolor, 3)

                #put a text
                # point = (int(pt[1]), int(pt[0]))
                # print(type(point[0]))
                cv2.putText(img, str(index), pt, 0, 1, mycolor, 1)
                # cv2.putText(frm, 'frame number: ' + str(frm_count), (100, 100), 0, 2, [255, 255, 255], 5)

            # break
        # break
    cv2.imwrite('output.png', img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)





def get_Colors():
    '''
    Generates randomly colors for different templates
    :return: niente
    '''
    if (len(Param.colors) != 0):
        return
    Param.numberOfCalls_getColors += 1
    templates_Number = len(load_templates())
    Param.colors = []
    for i in range(templates_Number):
        color = np.array(  (np.random.choice(range(256), size = 3) ))
        Param.colors.append(color)
    # print(Param.colors)

def show_Colors():
    '''
    Shows colors for each template
    :return:
    '''
    if len(Param.colors)==0:
        get_Colors()
    for i in range(len(Param.colors)):
        # print(i)
        # print((Param.colors[i] * np.ones((100, 100, 3), np.uint8)).shape)
        cv2.imshow('Color: '+str(i), (np.ones((100, 100, 3), np.uint8) * Param.colors[i]).astype(np.uint8) )
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

def compare_templates():
    '''comparing of templates. Does not help'''

    tmp_arr = load_templates()
    for el in range(len(tmp_arr)):
        print(el, ':  ', tmp_arr[el].shape)

    for index1 in range(len(tmp_arr)):
        try:
            res = cv2.matchTemplate(tmp_arr[0], tmp_arr[index1], cv2.TM_CCOEFF_NORMED)
            print(index1, ':    ', res)
        except:
            print('Does not work for ', 0, ' and ', index1)

class Param: # this class keeps some important constants
    colors=[]
    numberOfCalls_getColors =0
    dict_to_rotate = {0, 7} # list of templates which must be rotated

def get_rotated_templates():
    '''To each template we add its rotations by \pm 90 and 180 degrees
        returns a list of lists
    '''
    tmp_arr_rot = []
    tmp_arr = load_templates()

    for index in range(len(tmp_arr)):
        if index in Param.dict_to_rotate:
            tmp_rotated_180 = tmp_arr[index][::-1, ::-1]
            tmp_rotated_90 = tmp_arr[index].transpose()
            tmp_rotated_270 = tmp_rotated_90[::-1, ::-1]
            tmp_arr_rot.append([tmp_arr[index], tmp_rotated_90, tmp_rotated_180, tmp_rotated_270])
        else:
            tmp_arr_rot.append([tmp_arr[index]])

    # #print - optional
    # for index in range(len(tmp_arr_rot)):
    #     for j in range(len(tmp_arr_rot[index])):
    #         cv2.imshow('rotated by '+str(90*j), tmp_arr_rot[index][j])
    #     cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    return tmp_arr_rot

def load_templates(folder='symbols'):
    '''
    Download plan and templates by iterating over a folder
    :return:
    '''

    filename = 'plan.png'
    img = cv2.imread(filename)
    # cv2.imshow('img', img)
    # cv2.waitKey(10)

    image_arr_templ = []
    folder_name = folder #'symbols'
    # sorted(os.listdir(folder_name))
    for filename in sorted(os.listdir(folder_name)):
        filenameFolder = folder_name + '/' + filename
        img_templ = cv2.imread(filenameFolder, cv2.IMREAD_GRAYSCALE)
        image_arr_templ.append(img_templ)
        # cv2.imshow("img_templ",img_templ)
        # print(filename)
        # cv2.waitKey(0)
    return image_arr_templ

if __name__ =='__main__':
    start = time.time()

    # load_templates()
    # TemplateMatching()
    # get_rotated_templates()
    # get_Colors()
    # show_Colors()
    TemplateMatching2()

    print(Param.numberOfCalls_getColors)
    print(time.time()-start, 'seconds')