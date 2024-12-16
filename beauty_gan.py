# import dlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import tensorflow as tf
# import numpy as np
# import cv2
# import face_parsing


# def beauty():
#     detector = dlib.get_frontal_face_detector()
#     sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')


#     def align_faces(img):
#         dets = detector(img, 1)

#         objs = dlib.full_object_detections()

#         for detection in dets:
#             s = sp(img, detection)
#             objs.append(s)

#         faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)

#         return faces


#     # test
#     test_img = dlib.load_rgb_image('skin_test/12.jpg')

#     test_faces = align_faces(test_img)

#     fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(20, 16))
#     axes[0].imshow(test_img)

#     for i, face in enumerate(test_faces):
#         axes[i + 1].imshow(face)

#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     saver = tf.train.import_meta_graph('models/model.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('models'))
#     graph = tf.get_default_graph()

#     X = graph.get_tensor_by_name('X:0') # source
#     Y = graph.get_tensor_by_name('Y:0') # reference
#     Xs = graph.get_tensor_by_name('generator/xs:0') # output

#     def preprocess(img):
#         return img.astype(np.float32) / 127.5 - 1.

#     def postprocess(img):
#         return ((img + 1.) * 127.5).astype(np.uint8)

#     img1 = dlib.load_rgb_image('skin_test/12.jpg')
#     img1_faces = align_faces(img1)

#     img2 = dlib.load_rgb_image('add_mask.jpg')
#     img2_faces = align_faces(img2)

#     fig, axes = plt.subplots(1, 2, figsize=(16, 10))
#     axes[0].imshow(img1_faces[0])
#     axes[1].imshow(img2_faces[0])

#     src_img = img1_faces[0]
#     ref_img = img2_faces[0]

#     X_img = preprocess(src_img)
#     X_img = np.expand_dims(X_img, axis=0)

#     Y_img = preprocess(ref_img)
#     Y_img = np.expand_dims(Y_img, axis=0)

#     output = sess.run(Xs, feed_dict={
#         X: X_img,
#         Y: Y_img
#     })
#     output_img = postprocess(output[0])

#     fig, axes = plt.subplots(1, 3, figsize=(20, 10))
#     axes[0].set_title('Source')
#     axes[0].imshow(src_img)
#     axes[1].set_title('Reference')
#     axes[1].imshow(ref_img)
#     axes[2].set_title('Result')
#     axes[2].imshow(output_img)
#     im = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
#     cv2.imwrite("img/result.jpg", im)
    
    
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import cv2
import os

# 얼굴 정렬 및 스타일 변환을 위한 개선된 코드

def beauty(test_image_path, reference_image_path, output_dir="img", model_dir="models"):
    # 모델 및 파일 경로 설정
    shape_predictor_path = os.path.join(model_dir, 'shape_predictor_5_face_landmarks.dat')
    model_meta_path = os.path.join(model_dir, 'model.meta')

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # dlib 설정
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(shape_predictor_path)

    def align_faces(img):
        """얼굴 정렬 함수"""
        dets = detector(img, 1)
        objs = dlib.full_object_detections()
        for detection in dets:
            s = sp(img, detection)
            objs.append(s)
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
        return faces

    # 입력 이미지 로드 및 얼굴 정렬
    test_img = dlib.load_rgb_image(test_image_path)
    ref_img = dlib.load_rgb_image(reference_image_path)

    test_faces = align_faces(test_img)
    ref_faces = align_faces(ref_img)

    # 얼굴이 검출되지 않았을 경우
    if len(test_faces) == 0 or len(ref_faces) == 0:
        print("No faces detected in one of the images.")
        return

    # 얼굴 시각화
    fig, axes = plt.subplots(1, len(test_faces) + 1, figsize=(20, 16))
    axes[0].imshow(test_img)
    for i, face in enumerate(test_faces):
        axes[i + 1].imshow(face)

    # TensorFlow 세션 및 모델 로드
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.global_variables_initializer().run()
        saver = tf.compat.v1.train.import_meta_graph(model_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.compat.v1.get_default_graph()

        # 텐서 추출
        X = graph.get_tensor_by_name('X:0')  # 소스
        Y = graph.get_tensor_by_name('Y:0')  # 참조
        Xs = graph.get_tensor_by_name('generator/xs:0')  # 출력

        def preprocess(img):
            """이미지 전처리"""
            return img.astype(np.float32) / 127.5 - 1.

        def postprocess(img):
            """이미지 후처리"""
            return ((img + 1.) * 127.5).astype(np.uint8)

        # 변환할 소스 및 참조 이미지 전처리
        src_img = preprocess(test_faces[0])
        src_img = np.expand_dims(src_img, axis=0)

        ref_img = preprocess(ref_faces[0])
        ref_img = np.expand_dims(ref_img, axis=0)

        # 스타일 변환 실행
        output = sess.run(Xs, feed_dict={
            X: src_img,
            Y: ref_img
        })

        # 변환 결과 후처리
        output_img = postprocess(output[0])

        # 결과 시각화
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        axes[0].set_title('Source')
        axes[0].imshow(test_faces[0])
        axes[1].set_title('Reference')
        axes[1].imshow(ref_faces[0])
        axes[2].set_title('Result')
        axes[2].imshow(output_img)

        # 결과 저장
        output_path = os.path.join(output_dir, "result2.jpg")
        output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_img_bgr)
        print(f"Result saved to {output_path}")

# 실행
if __name__ == "__main__":
    test_image_path = "./input_img.jpg"
    reference_image_path = "../img/reference.jpg"
    beauty(test_image_path, reference_image_path)
 