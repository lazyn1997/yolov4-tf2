import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from yolo import YOLO
from tqdm import tqdm

# from write_xml import write_xml

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    yolo = YOLO()
    root = 'E:/postgraduate_data/fish/third/'
    # root = '/public/home/zhaoyaning/program/fish_recognition/Individual_recognition2/'
    output_dir = 'output/LYC_third_filter_416/videos'
    if output_dir:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = 1920, 1080
    if output_dir and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    txt_dir = '%s/result_many.txt' % output_dir
    txt = open(txt_dir, 'w')
    for date in ['4_22_split']:
        path = '%s%s/' % (root, date)
        fileList = os.listdir(path)
        acc = {}
        strip, acc_low, acc_high = 1, 0, 0
        for name in fileList:
            if name in ['001.mp4', '021.mp4', '025.mp4', '033.mp4', '045.mp4']:
                videoflag = name.replace('.mp4', '')
                videolabel = 'fish_%s' % videoflag
                if output_dir:
                    out_path = os.path.join(output_dir, name)
                    writer = cv2.VideoWriter(out_path, fourcc, 30, (width, height))
                cap = cv2.VideoCapture(path + name)
                predicts = {}
                pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps, total, detect, rightl, righth = 0.0, 0, 0, 0, 0
                while True:
                    t1 = time.time()
                    ref, frame0 = cap.read()
                    if not ref:
                        break
                    if total % strip == 0:
                        # 格式转变，BGRtoRGB
                        frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(np.uint8(frame))
                        frame, labels, boxes = yolo.detect_image(frame)
                        # RGBtoBGR满足opencv显示格式
                        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

                        if len(labels) > 0:
                            detect += 1
                            top_result = labels[0]
                            if top_result in predicts:
                                predicts[top_result] += 1
                            else:
                                predicts[top_result] = 1
                        if videolabel in labels:
                            righth += 1
                            if len(labels) == 1:
                                rightl += 1
                                # top, left, bottom, right = boxes[0]
                                # photoname = '%s_%d.png' % (videoflag, detect)
                                # photopath = 'E:\\Documents\\postgraduate\\fish_recognition_yolov4\\fish\\JPEGImages\\%s' \
                                #             % photoname
                                # xml_name = 'fish\\Annotations\\%s_%d.xml' % (videoflag, detect)
                                # write_xml(photoname, photopath, 'fish_%s' % videoflag, xml_name,
                                #           left, top, right, bottom)
                                # cv2.imwrite(photopath, frame0)
                    else:
                        frame = frame0
                        labels = []
                    fps = (fps + (1. / (time.time() - t1))) / 2
                    show_result = 'label: %s, predict: %s, fps: %.4f' % (videoflag, labels, fps)
                    pbar.update(1)
                    pbar.set_description('%s, Progress' % show_result)
                    frame = cv2.putText(frame, show_result, (0, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 255), 2)
                    cv2.namedWindow('fish', cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty('fish', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('fish', frame)
                    if output_dir:
                        writer.write(frame)
                    if cv2.waitKey(33) & 0xff == ord('q'):
                        break
                    total += 1
                cap.release()
                pbar.close()
                if output_dir:
                    writer.release()
                total_rightl = round(rightl / detect, 4)
                total_righth = round(righth / detect, 4)
                for pre in predicts:
                    predicts[pre] = round(predicts[pre] / detect, 4)
                predicts = sorted(predicts.items(), key=lambda x: x[1], reverse=True)
                tqdm.write('%s, right_low: %s, right_high: %s' % (name, total_rightl, total_righth))
                txt.write('%s, total frame is %d, detect frame is %d, right_low: %s, right_high: %s, all predicts is %s\n'
                          % (name, total, detect, total_rightl, total_righth, str(predicts)))
                acc[videoflag] = {'low': total_rightl}
                acc[videoflag]['high'] = total_righth
                acc_low += total_rightl
                acc_high += total_righth
        cv2.destroyAllWindows()
        acc['low'] = round(acc_low / 5, 4)
        acc['high'] = round(acc_high / 5, 4)
        tqdm.write('%s' % acc)
        txt.write('%s' % acc)
        txt.close()
