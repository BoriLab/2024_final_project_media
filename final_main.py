
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import pymysql
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976

# RGB 색상을 LAB 색 공간으로 변환하는 함수입니다. LAB 색 공간은 색상 비교에 유용
def rgb2lab(inputColor):    
    RGB = [float(value) / 255 for value in inputColor]
    RGB = [((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92 for value in RGB]
    RGB = [value * 100 for value in RGB]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ = [X / 95.047, Y / 100.0, Z / 108.883]
    XYZ = [value ** (1 / 3) if value > 0.008856 else (7.787 * value) + (16 / 116) for value in XYZ]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])
    return [round(L, 4), round(a, 4), round(b, 4)]

# KMeans 클러스터링 결과를 히스토그램 형태로 반환합니다. 각 클러스터의 비율을 계산합니다.
def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    hist, _ = np.histogram(clt.labels_, bins=numLabels)
    return (hist / hist.sum()).astype("float")

# 클러스터링된 색상을 바 차트로 시각화합니다. 색상 분포를 확인하는 데 유용합니다.
def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    for percent, color in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
    return bar

# 데이터베이스에서 화장품 데이터를 가져와 LAB 값 기반으로 색상 차이를 계산합니다. DB 사용 여부는 플래그로 제어됩니다.
def query_db_and_calculate(lab_code, use_db=False):
    if not use_db:
        print("Database functionality is disabled.")
        return None

    # DB 연결하기
    db = pymysql.connect(host="127.0.0.1", user="root", password="1234", db="condb", charset="utf8")

    # DB 커서 만들기
    cursor = db.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT * FROM condb.cosmetic;"
    cursor.execute(sql)
    table = []
    result = cursor.fetchall()
    for record in result:
        table.append(list(record.values()))
    db.close()

    print(table)
    print(len(table))
    delta_e_list = []
    cosmetics_lab_list = []
    skin_rgb = LabColor(lab_l=lab_code[0], lab_a=lab_code[1], lab_b=lab_code[2])
    for i in range(len(table)):
        cosmetics_lab_list.append(LabColor(lab_l=table[i][5], lab_a=table[i][6], lab_b=table[i][7]))
        delta_e = delta_e_cie1976(skin_rgb, cosmetics_lab_list[i])
        delta_e_list.append(delta_e)
        table[i].append(delta_e)

    table.sort(key=lambda x: x[-1])
    del table[5:]
    return table

def calculate_average_color_lab(img):
    # 이미지의 평균 RGB 값을 계산하고 이를 LAB 색 공간으로 변환합니다.
    mean_color_per_row = np.mean(img, axis=0)
    mean_color = np.mean(mean_color_per_row, axis=0)
    mean_color_lab = rgb2lab(mean_color)
    return mean_color_lab


def get_data(path, use_db=False):  # 이미지 경로를 받아 얼굴 검출, 피부색 추출, 그리고 DB 처리를 수행하는 메인 함수입니다.
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    # cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    FaceDetector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    faces = FaceDetector(gray)  # 얼굴을 탐지합니다.
    if not faces:
        print("No faces detected.")
        return None

    # 얼굴 인식
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        img = image[y1:y2, x1:x2]  # 사각형으로 그린부분 잘라서 img 변수에 저장
        # cv2.imshow("Cropped Face", img)
        cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w, _ = img.shape
    circle_size = (h + w) // 30

    # 이제 68개의 점을 그릴건데
    # 위에서 구한 circle_size로 그릴거임
    for face in FaceDetector(gray):
        shape = predictor(gray, face)
        for n in range(0, 68):
            x = shape.part(n).x
            y = shape.part(n).y
            cv2.circle(img, (x, y), circle_size, (0, 0, 0), -1)

    # cv2.imshow("Landmarks Drawn", img)
    cv2.waitKey(0)

    face_img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([30, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_msk = cv2.inRange(face_img_ycrcb, lower, upper)
    skin = cv2.bitwise_and(img, img, mask=skin_msk)

    # cv2.imshow("Skin Mask Applied", skin)
    cv2.waitKey(0)

    average_color = calculate_average_color_lab(skin)
    print(f"Average lab color of the face region: {average_color}")  # 평균 RGB 색상을 출력합니다.

    image = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))

    # 클러스터 개수를 설정합니다.
    k = 2
    clt = KMeans(n_clusters=k, random_state=42)
    clt.fit(image)

    rgb_code_list = []
    for center in clt.cluster_centers_:
        if int(center[0]) >= 94 and int(center[1]) >= 61 and int(center[2]) >= 40:
            rgb_code_list.append(center)

    if not rgb_code_list:
        print("No valid skin tones found.")
        return None

    rgb_code = rgb_code_list[0]
    lab_code = rgb2lab(rgb_code)
    hsv_code = colorsys.rgb_to_hsv(rgb_code[0] / 255, rgb_code[1] / 255, rgb_code[2] / 255)

    # 피부 톤 분류 기준을 설정합니다.
    vbs_zero_point = [65.1587, 17.6091, 0.3487762]
    if lab_code[0] > vbs_zero_point[0] and lab_code[2] > vbs_zero_point[1] and hsv_code[1] > vbs_zero_point[2]:
        skin_tone = "Sprint warm bright"
    elif lab_code[0] > vbs_zero_point[0] and lab_code[2] > vbs_zero_point[1] and hsv_code[1] < vbs_zero_point[2]:
        skin_tone = "Spring warm light"
    elif lab_code[0] > vbs_zero_point[0] and lab_code[2] < vbs_zero_point[1] and hsv_code[1] < vbs_zero_point[2]:
        skin_tone = "Summer cool light"
    elif lab_code[0] < vbs_zero_point[0] and lab_code[2] < vbs_zero_point[1] and hsv_code[1] < vbs_zero_point[2]:
        skin_tone = "Summer cool mute"
    elif lab_code[0] < vbs_zero_point[0] and lab_code[2] > vbs_zero_point[1] and hsv_code[1] < vbs_zero_point[2]:
        skin_tone = "Autumn warm mute"
    elif lab_code[0] < vbs_zero_point[0] and lab_code[2] > vbs_zero_point[1] and hsv_code[1] > vbs_zero_point[2]:
        skin_tone = "Autumn warm deep"
    elif lab_code[0] < vbs_zero_point[0] and lab_code[2] < vbs_zero_point[1] and hsv_code[1] > vbs_zero_point[2]:
        skin_tone = "Winter cool deep"
    elif lab_code[0] > vbs_zero_point[0] and lab_code[2] < vbs_zero_point[1] and hsv_code[1] > vbs_zero_point[2]:
        skin_tone = "Winter cool bright"
    else:
        skin_tone = "None"

    db_results = query_db_and_calculate(lab_code, use_db)
    if db_results:
        print("Top cosmetic matches:")
        for item in db_results:
            print(item)

    print(f"Detected skin tone: {skin_tone}")  # 결과를 출력합니다.
    return lab_code

if __name__ == "__main__":
    image_path = "./input_img.jpg"
    use_db = False  # 로컬 더미 db이기 때문에 제출시 false
    try:
        skin_tone = get_data(image_path, use_db)
        if skin_tone:
            print(f"Detected skin tone: {skin_tone}")
        else:
            print("No skin tone detected.")
    except Exception as e:
        print(f"An error occurred: {e}")
