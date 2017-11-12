from styx_msgs.msg import TrafficLight
from scipy.misc import imresize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.stats import norm
import os
import collections

# Frozen inference graph files.
SSD_GRAPH_FILE = os.getcwd() + \
    '/ssd_mobilenet/model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

RESULT_DIR = os.getcwd() + "/ssd_mobilenet/test/"

VIZ_DEBUG = 0

COLOR_TO_CLASS = {"Red": 0,
                  "Yellow": 1,
                  "Green": 2,
                  "Unknown": 3}
CLASS_TO_RGBCOLOR = {0: (255, 0, 0),
                     1: (255, 255, 0),
                     2: (0, 255, 0),
                     3: (19, 139, 69)}
# ==============
# Utility funcs
# ==============


def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            if classes[i] == 10:    # Traffic light class is 10
                idxs.append(i)

    # If no boxes are greater than 'min_score', append the highest
    # probability box if it's classification = 10
    if len(idxs) == 0:
        max_score = max(scores)
        max_idx = [i for i, score in enumerate(scores) if score == max_score]
        max_idx = max_idx[0]
        if classes[max_idx] == 10:
            idxs.append(max_idx)


    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords


def draw_bboxes(img, bboxes, classes, probs=None):
    ''' Drawing bounding box'''
    if probs is None:   # if probability not available
        # ground truth
        probs = []
        for i in range(len(bboxes)):
            probs.append(-1)

    for i, box in enumerate(bboxes):
        # draw the bbox:
        xmin = int(box[1])
        ymin = int(box[0])
        xmax = int(box[3])
        ymax = int(box[2])

        # Draw only valid color
        if classes[i] < 3:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          CLASS_TO_RGBCOLOR[classes[i]], 2)

            if probs is not None:
                # write the detection probability on the bbox:
                # # make the top line of the bbox thicker:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymin - 12),
                              CLASS_TO_RGBCOLOR[classes[i]], -1)
                # # write the probaility in the top line of the bbox:
                prob_string = "%.2f" % probs[i]
                cv2.putText(img, prob_string, (int(xmin) + 2, int(ymin) - 2), 2, 0.4,
                            (255, 255, 255), 1)

    img_with_bboxes = img
    return img_with_bboxes


def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def crop_image(rgb_image, boxes):
    ''' Cropping image in a bounding boxes'''
    cropped_imgs = []
    for i in range(len(boxes)):
        ymin, xmin, ymax, xmax = boxes[i, ...]

        cropped = rgb_image[int(ymin): int(ymax), int(xmin):int(xmax), :]

        cropped_imgs.append(cropped)
        # cv2.imshow("cropped", cropped)
        # cv2.waitKey(0)
        if 0:
            if not os.path.exists(RESULT_DIR):
                os.makedirs(RESULT_DIR)
            cv2.imwrite(RESULT_DIR + "cropped" + str(i) + ".jpg", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    return cropped_imgs


def match_histogram(cropped_imgs):
    ''' Traffic light classification using histogram color matching '''
    results = []
    for cropped_img in cropped_imgs:
        RGB = [0., 0., 0.]
        # Show numpy histogram
        if cropped_img.shape[0] == 0:
            print()
            return

        for ch in range(3):
            flatten_ch0 = cropped_img[:, :, ch].flatten()
            color_freq = collections.Counter(flatten_ch0)

            total_val = np.sum(np.array(list(color_freq.values())))
            sorted_colors_tp = list(sorted(color_freq.items()))

            # calculate only the frequency of the 250-255 values
            for i in range(250, 256):
                RGB[ch] = RGB[ch] + color_freq[i]

            # Normalization
            RGB[ch] = float(RGB[ch]) / total_val

            if VIZ_DEBUG:
                if ch == 0:
                    strcolor = "r"
                elif ch == 1:
                    strcolor = "g"
                else:
                    strcolor = "b"
                sorted_colors_tp = np.array(sorted_colors_tp)
                plt.bar(sorted_colors_tp[:, 0], sorted_colors_tp[:, 1] /
                        total_val, color=strcolor, label="ch" + str(ch))
        if VIZ_DEBUG:
            plt.legend()
            plt.show()

        max_val = max(RGB)
        max_idx = [i for i, ch in enumerate(RGB) if ch == max_val]
        #print("RGB: ", RGB)
        #print("Max idx : %s with val %f " % (max_idx, max_val))
        red_thres_ratio = 1.0
        green_thres_ratio = 1.0
        yellow_thres_ratio = 0.4
        blue_thres = 0.07

        redgreen_ratio = RGB[0] / RGB[1] if RGB[1] > 0. else 10

        greenblue_ratio = RGB[1] / RGB[2] if RGB[2] > 0. else 10

        if RGB[2] < blue_thres:
            if redgreen_ratio > red_thres_ratio:
                results.append(COLOR_TO_CLASS["Red"])
            elif redgreen_ratio >= yellow_thres_ratio:
                results.append(COLOR_TO_CLASS["Yellow"])
            else:
                results.append(COLOR_TO_CLASS["Green"])
        else:
            if greenblue_ratio > green_thres_ratio:
                results.append(COLOR_TO_CLASS["Green"])
            else:
                results.append(COLOR_TO_CLASS["Yellow"])

            # Not appending unknown
            # results.append(COLOR_TO_CLASS["Unknown"])

        # Comparison to PIL histogram
        # ----------------------------------------
        # RGB_dict={"Red":0, "Yellow":0, "Green":0}
        # print(type(cropped_img.histogram()))
        # # https://stackoverflow.com/questions/22460577/understanding-histogram-in-pillow
        # for i, value in enumerate(cropped_img.histogram()):
        #     #print(i, value) # i: 0 -767
        #     if i > 0 and i <15:
        #         RGB_dict["Red"]= value
        #     elif i>500 and i < 515:
        #         RGB_dict["Green"]= value
        #     elif i >760:
        #         RGB_dict["Blue"]= value
        # #print(max(RGB_dict, key=RGB_dict.get))
        # if VIZ_DEBUG:
        #     plt.plot(cropped_img.histogram())
        #     plt.show()
    return results


class TLClassifier(object):
    def __init__(self):
        # Start code based on https://github.com/udacity/CarND-Object-Detection-Lab
        self.detection_graph = load_graph(SSD_GRAPH_FILE)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
            'detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
            'detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')

        self.sess = tf.Session(graph=self.detection_graph)
        #summary_writer = tf.summary.FileWriter("./logs", sess.graph)

    def get_classification(self, bgr_img, DEBUG=False):

        b, g, r = cv2.split(bgr_img)       # get b,g,r
        rgb_img = cv2.merge([r, g, b])     # switch it to rgb

        image_np = np.expand_dims(rgb_img, 0)

        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        if DEBUG:
            print("Scores: ", scores)
            print("Classes: ", classes)
        confidence_cutoff = 0.27
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(
            confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        height, width = rgb_img.shape[0], rgb_img.shape[1]
        box_coords = to_image_coords(boxes, height, width)
        if DEBUG:
            print("boxes: ", boxes)
            print("Box coord: ", box_coords)

        #Cropped image
        if len(box_coords) > 0:
            cropped_imgs = crop_image(rgb_img, box_coords)
            det_colors = match_histogram(cropped_imgs)

            if det_colors is None:
                return 3

            color_freq = collections.Counter(det_colors)
            final_color = color_freq.most_common(1)[0]

            max_prob = max(scores)
            print("Final result: %d, probs: %f" %
                  (final_color[0], max_prob))

            bbox_img = draw_bboxes(rgb_img, box_coords, det_colors, scores)

            if DEBUG:
                if not os.path.exists(RESULT_DIR):
                    os.makedirs(RESULT_DIR)
                cv2.imwrite(RESULT_DIR + "ros_detection.jpg", cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
                cv2.imshow("bla", cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
            return final_color[0], max_prob
        else:
            return 3, 0

