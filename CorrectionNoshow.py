#!/usr/bin/python
#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import *
from util import *
from hed_net import *
from utils import *


from tensorflow import flags
from PIL import Image,ImageEnhance
from itertools import combinations
flags.DEFINE_string('image', './test_image/69.jpg',
                    'Image path to run hed, must be jpg image.')
flags.DEFINE_string('checkpoint_dir', './checkpoint',
                    'Checkpoint directory.')
flags.DEFINE_string('output_dir', './test_image',
                    'Output directory.')
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.image):
    print('image {} not exists, please retry' % FLAGS.image)
    exit()


def doc_scanning(imgdir,outdir):
    batch_size = 3
    im = Image.open(imgdir)

    # # 对比度增强
    # enh_con = ImageEnhance.Contrast(im)
    # contrast = 1
    # picc = enh_con.enhance(contrast)
    #
    # # 锐度增强
    # enh_sha = ImageEnhance.Sharpness(picc)
    # sharpness = 8
    # im = enh_sha.enhance(sharpness)
    # im.show()
    # # # -----    save
    # # path_new = os.path.join(save_dir, 'new' + file_name)
    # # pics.save(path_new)


    name = os.path.basename(imgdir)

    '''
    resize
    '''
    w = 700
    w_0,h_0 = im.size
    h = round(w*h_0/w_0)

    im = im.resize((w, h))
    # im.show()
    # img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    # # ~~~~~~~~~~~~~~~~~~~~ 保存线段值
    # linespath = os.path.join(outdir, os.path.basename(os.path.splitext(i)[0]) + 'line_w500.txt')
    # f = open(linespath, 'w')
    # ~~~~~~~~~~~~~~~~~~~~~ 写入线段txt  绘制直线

    '''
    HED predict
    '''
    tf.reset_default_graph()
    image_path_placeholder = tf.placeholder(tf.string)
    is_training_placeholder = tf.placeholder(tf.bool)
    feed_dict_to_use = {image_path_placeholder: imgdir,
                        is_training_placeholder: False}

    image_tensor = tf.read_file(image_path_placeholder)
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
    image_tensor = tf.image.resize_images(image_tensor, [const.image_height, const.image_width])

    image_float = tf.to_float(image_tensor)

    if const.use_batch_norm == True:
        image_float = image_float / 255.0
    else:
        # for VGG style HED net
        image_float = mean_image_subtraction(image_float, [R_MEAN, G_MEAN, B_MEAN])
    image_float = tf.expand_dims(image_float, axis=0)

    dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5 = mobilenet_v2_style_hed(image_float, batch_size,
                                                                    is_training_placeholder)

    global_init = tf.global_variables_initializer()

    # Saver
    hed_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hed')
    saver = tf.train.Saver(hed_weights)

    with tf.Session() as sess:
        sess.run(global_init)

        latest_ck_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_ck_file:
            print('restore from latest checkpoint file : {}'.format(latest_ck_file))
            saver.restore(sess, latest_ck_file)
        else:
            print('no checkpoint file to restore, exit()')
            exit()

        _dsn_fuse, \
        _dsn1, \
        _dsn2, \
        _dsn3, \
        _dsn4, \
        _dsn5 = sess.run([dsn_fuse,
                          dsn1, dsn2,
                          dsn3, dsn4,
                          dsn5],
                         feed_dict=feed_dict_to_use)


        '''
        HED 网络输出的 Tensor 中的像素值，并不是像 label image 那样落在 (0.0, 1.0) 这个区间范围内的。
        用 threshold 处理一下，就可以转换成对应 image 的矩阵，让像素值落在正常取值区间内
        '''
        threshold = 0.0

        dsn_fuse_image = np.where(_dsn_fuse[0] > threshold, 255, 0)
        # dsn_fuse_image_path = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + 'mask0.jpg')
        # dsn_fuse_image = _dsn_fuse[0].astype('uint8')
        dsn_fuse_image = dsn_fuse_image.astype('uint8')
        dsn_fuse_image = cv2.resize(dsn_fuse_image, (w, h), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(dsn_fuse_image_path, dsn_fuse_image)


        '''
        HoughLinesP 直线检测
        '''
        # dsn_fuse_image = cv2.imread(dsn_fuse_image_path, 0)
        resized = cv2.resize(dsn_fuse_image, (w,h), interpolation=cv2.INTER_AREA)
        lines = cv2.HoughLinesP(resized, 1, 1 * np.pi / 360, 100, minLineLength=0.25*w,
                                maxLineGap=0.01*w)  # 200 50 5 300 500

        # img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(img_extd, (x1, y1), (x2, y2), (255, 0, 0), 1)  # BGR
        #         # cv2.line(img_extd, (x1, y1), (x2, y2), (255, 0, 0), 1)  # BGR
        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + '_line.jpg')
        # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
        # img1.save(savepath)
        # img1.show()
        # # print("Line Num : ", len(lines))
        # print(lines)
        # print("type:",type(lines))
        # img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + '_HL.jpg')
        # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
        # img1.save(savepath)
        # img2 = Image.open(savepath)
        # img2.show()


        '''
        画直线
        延长直线
        '''
        img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        line_extd = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # print('hhhhhhhh:',x1, y1, x2, y2,w,h)
                x1n, y1n, x2n, y2n = extend(int(x1), int(y1), int(x2), int(y2), w, h)
                line_extd.append([x1n, y1n, x2n, y2n])
                # cv2.line(img_extd, (x1n, y1n), (x2n, y2n), (255, 0, 0), 1)  # BGR
                # cv2.line(img_extd, (x1, y1), (x2, y2), (255, 0, 0), 1)  # BGR
        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + '_longline.jpg')
        # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
        # img1.show()
        # img1.save(savepath)
        # img2 = Image.open(savepath)
        # img2.show()

        '''
        分组
        '''
        group = {}
        limit = 20
        n = 0
        for i in line_extd:
            x1n, y1n, x2n, y2n = i
            flag = 0
            if (abs(x1n-0)<10 and abs(y1n-0)<10 and abs(x2n-w)<10 and abs(y2n-0)<10) or \
                (abs(x1n - w) < 10 and abs(y1n - 0) < 10 and abs(x2n - 0) < 10 and abs(y2n - 0) < 10) or \
                (abs(x1n-w)<10 and abs(y1n-0)<10 and abs(x2n-w)<10 and abs(y2n-h)<10) or \
                (abs(x1n - w) < 10 and abs(y1n - h) < 10 and abs(x2n - w) < 10 and abs(y2n - 0) < 10) or \
                (abs(x1n - w) < 10 and abs(y1n - h) < 10 and abs(x2n - 0) < 10 and abs(y2n - h) < 10) or \
                (abs(x1n - 0) < 10 and abs(y1n - h) < 10 and abs(x2n - w) < 10 and abs(y2n - h) < 10) or \
                (abs(x1n - 0) < 10 and abs(y1n - h) < 10 and abs(x2n - 0) < 10 and abs(y2n - 0) < 10) or \
                (abs(x1n - 0) < 10 and abs(y1n - 0) < 10 and abs(x2n - 0) < 10 and abs(y2n - h) < 10):
                continue
            if not group:
                group[0] = [[x1n, y1n, x2n, y2n]]
                continue
            for j in group:
                bro = group[j][0]
                x1b, y1b, x2b, y2b = bro
                if ((abs(x1n - x1b) <= limit and abs(x2n - x2b) <= limit and abs(y1n - y1b) <= limit and abs(
                        y2n - y2b) <= limit) or (abs(x1n - x2b) <= limit and abs(x2n - x1b) <= limit and abs(y1n - y2b) <= limit and abs(
                            y2n - y1b) <= limit)):
                    group[j].append([x1n, y1n, x2n, y2n])
                    flag = 1
                    break
            if flag == 0:
                n += 1
                group[n] = [[x1n, y1n, x2n, y2n]]

        # for i in group:
        #     img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        #     for j in group[i]:
        #         x1n, y1n, x2n, y2n = j
        #         # print(x1n, y1n, x2n, y2n)
        #         cv2.line(img_extd, (x1n, y1n), (x2n, y2n), (255, 0, 0), 1)  # BGR

            # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) +'_linegroup'+'_'+str(i)+'_'+ '.jpg')
            # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
            # img1.save(savepath)
            # img2 = Image.open(savepath)
            # img2.show()
        # print('**********************************************8 is:\n',group[8])

        '''
        用每组的平均值代表本组，其他的都去掉 √
        '''
        img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        line_mean = []
        for i in group:
            r = np.array(group[i])
            r_mean = np.round(r.sum(axis=0) / len(group[i])).astype('int')
            x1n, y1n, x2n, y2n = list(r_mean)
            line_mean.append(list(r_mean))
            # cv2.line(img_extd, (x1n, y1n), (x2n, y2n), (255, 0, 0), 1)  # BGR

        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) +  'avg.jpg')
        # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
        # img1.save(savepath)
        # img2 = Image.open(savepath)
        # img2.show()
        # print('line_mean is :\n',line_mean)


        #求交点
        comb_lines = list(combinations(line_mean, 2))
        intersecs = []
        for c in comb_lines:
            x1, y1, x2, y2 = c[0]
            x3, y3, x4, y4 = c[1]
            [xx, yy] = findIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
            #             print(xx,yy)
            if xx > 0 and xx < w and yy > 0 and yy < h:
                if [xx, yy] not in intersecs:
                    intersecs.append([xx, yy])
            else:
                continue
        # print('intersecs is : \n')
        # print(intersecs)
        
        '''
        随机取四个点
        '''
        comb_points = list(combinations(intersecs, 4))
        areas = []
        for i in comb_points:
            comb_sg = list(i)
            polygon = np.array(order_points(np.array(list(i), dtype=np.int32)))
            area = polygon_area(polygon)
            #     print(area)
            if area < 0:
                area = -area  # 逆时针求得面积是负的
            comb_sg.append(area)
            areas.append(comb_sg)
        areas = sorted(areas, key=lambda x: x[-1], reverse=True)
        # print('areas unsorted:\n')
        # print(areas)
        candidate = areas[0][:4]

        '''
        将点按左上右上右下左下排序
        '''
        candidate = np.array(candidate, dtype=np.int32)
        candidate = order_points(candidate)
        # print('candidate is \n:', candidate)
        n = 0
        img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        for i in range(len(candidate)-1):
            x1, y1 = candidate[i]
            # print(x, y)
            x2 , y2 = candidate[i+1]
            cv2.line(img_extd, (x1, y1), (x2, y2), (255, 0, 0), 1)  # BGR
        x1 , y1 = candidate[0]
        x2 , y2 = candidate[-1]
        cv2.line(img_extd, (x1, y1), (x2, y2), (255, 0, 0), 1)  # BGR
            # cv2.drawMarker(img_extd, position=((round(x), round(y))), color=(0, 255, 0),
            #                thickness=1,
            #                #                    markerType=mark_type,
            #                line_type=cv2.LINE_8,
            #                markerSize=20)
            # img1 = Image.fromarray(cv2.cvtColor(img_extd, cv2.COLOR_BGR2RGB))
            # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) +'_'+str(n)+'_'+ 'points.jpg')
            # img1.save(savepath)
            # img2 = Image.open(savepath)
            # img2.show()
            # n+=1

        # '''
        # 透视变换
        # '''
        # tl,tr,br,bl=candidate
        # img_extd = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        # warped = four_point_transform(img_extd, tl, tr, br, bl)
        # img1 = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        # img1 = img1.resize((w, h))
        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + '_' +  'warped.jpg')
        # img1.save(savepath)
        # img2 = Image.open(savepath)
        # img2.show()


        # #判断是否有客户签名
        # img = cv2.imread(savepath);
        # # img1 = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # # img1.show()
        # # 转成灰度图片
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        # # 二值化 自适应阈值
        # # ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        # blocksize = 201#101
        # C = 20
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
        # img0 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        # savepath = os.path.join(outdir, os.path.basename(os.path.splitext(name)[0]) + '_' + 'sign.jpg')
        # img0.save(savepath)
        # img0 = Image.open(savepath)
        # img0.show()


        # # # 工单号
        # # cod = img[200:230,190:390]
        # # img0 = Image.fromarray(cv2.cvtColor(cod, cv2.COLOR_GRAY2RGB))
        # # img0.show()
        # #
        # # ## b.设置卷积核5*5
        # # kernel = np.ones((10, 10), np.uint8)
        # #
        # # ## c.图像的腐蚀，默认迭代次数
        # # img_cod = cv2.erode(cod, kernel)
        # # # img1 = Image.fromarray(cv2.cvtColor(img_trans, cv2.COLOR_GRAY2RGB))
        # # # img1.show()
        # # # print(img_cod)
        # # ratio_cod = np.sum(img_cod == 0) / img_cod.size
        # # print(ratio_cod)

        # #签名
        # # signature = img[1005:1175,245:525]
        # signature = img[round(0.8375*h):round(0.9792*h), round(0.2722*w):round(0.5833*w)]
        # img0 = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_GRAY2RGB))
        # img0.show()

        # ## b.设置卷积核5*5
        # kernel = np.ones((10, 10), np.uint8)

        # ## c.图像的腐蚀，默认迭代次数
        # img_sign = cv2.erode(signature, kernel)
        # img1 = Image.fromarray(cv2.cvtColor(img_sign, cv2.COLOR_GRAY2RGB))
        # img1.show()
        # # print(img_sign)
        # ratio_sign = np.sum(img_sign == 0) / img_sign.size
        # print(ratio_sign)



        # if ratio_sign>0.06:
        #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #     print('Signature get!')
        #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # else:
        #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #     print('Signature not found!')
        #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return img_extd



if __name__ == "__main__":
    # imgdir = r'./test_imgs_input/10.jpg'
    # outdir = r'./test_imgs_output_resize/'
    imgdir = r'./testdir_1206/4.jpg'
    outdir = r'./testdir_1206/'


    if os.path.isfile(imgdir):
        path = imgdir
        doc_scanning(path,outdir)

    elif os.path.isdir(imgdir):
        for i in os.listdir(imgdir):
            path = os.path.join(imgdir,i)
            doc_scanning(path,outdir)