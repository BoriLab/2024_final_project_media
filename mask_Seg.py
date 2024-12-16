import warnings
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from models.detector import face_detector
from models.parser import face_parser

def mask_img(path, l_val=81.2, a_val=8.4, b_val=23.1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["SM_FRAMEWORK"] = 'tf.keras'
    warnings.filterwarnings("ignore")

    def resize_image(im, max_size=768):
        if np.max(im.shape) > max_size:
            ratio = max_size / np.max(im.shape)
            print(f"Resize image to ({int(im.shape[1]*ratio)}, {int(im.shape[0]*ratio)}).")
            return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
        return im

    def convert_lab_to_opencv(l, a, b):
        """LAB 값을 OpenCV 범위로 변환"""
        L_opencv = int((l / 100) * 255)  # L 범위: 0~255
        a_opencv = int(a + 128)          # a 범위: 0~255
        b_opencv = int(b + 128)          # b 범위: 0~255
        return [L_opencv, a_opencv, b_opencv]

    # 이미지 로드
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Image not found at {path}")
    im = im[..., ::-1]  # BGR -> RGB 변환
    im = resize_image(im)

    fd = face_detector.FaceAlignmentDetector(
        lmd_weights_path="./models/detector/FAN/2DFAN-4_keras.h5"
    )

    bboxes = fd.detect_face(im, with_landmarks=False)
    assert len(bboxes) > 0, "No face detected."

    prs = face_parser.FaceParser()
    out = prs.parse_face(im)

    skin = np.array(out[0])
    skin_index = np.where(skin == 1)

    # LAB 값 변환 및 적용
    lab_color = convert_lab_to_opencv(l_val, a_val, b_val)
    lab_image = cv2.cvtColor(im, cv2.COLOR_RGB2Lab)  # RGB -> LAB 변환
    lab_image[skin_index] = lab_color
    result_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)  # LAB -> RGB 변환

    # 결과 저장 및 표시
    output_path = "img/reference2.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"Masked image saved to {output_path}")
    # plt.imshow(result_image)
    # plt.show()


