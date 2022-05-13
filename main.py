from typing import List, Any
from apscheduler.schedulers.background import BackgroundScheduler
import pytesseract
import cv2
from io import BytesIO
# from utils import models
import paho.mqtt.client as mqtt
import requests, time, queue, re, datetime, json, os, ssl
from threading import Thread
from pytz import timezone, utc
from collections import Counter
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# 이전 차량 번호 (ex. 서울00가0000) :
old_car_number_format = re.compile(r'[가-힣]{2}\d{2}[가-힣]\d{4}')
# 신규 차량 번호 (ex. 23호 6144) :
new_car_number_format = re.compile(r'\d{2,3}[가-힣]\s\d{4}')
new_car_number_format2 = re.compile(r'\d{2,3}[가-힣]\d{4}')
car_numbers = []


def main():
    cam1 = 'rtsp://admin:123456@192.168.1.103/streaming/channel/101'
    cam1_queue = queue.Queue()
    cam1_receive_thread = Thread(target=receive_image, args=(cam1, cam1_queue))
    cam1_receive_thread.start()
    time.sleep(5)
    second_scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    second_scheduler.add_job(ocr_process1, 'interval', args=(cam1_queue,), seconds=10)
    # second_scheduler.add_job(load_image, 'interval', seconds=5)

    second_scheduler.start()


def ocr_process1(video_que):
    global car_numbers
    img = get_image(video_que)
    cv2.imwrite('D:/test111.jpg', img)
    # use easyOCR
    # reader = easyocr.Reader(['ko', 'en'], gpu=False)
    # result = reader.readtext(img)
    # ocr_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # ocr_img = ocr_img[400:1100, :]
    # result = pytesseract.image_to_data(img)
    # result = pytesseract.image_to_string(Image.open('test.png'))
    # result = result.split('\n')
    # print(result)
    # car_number_list = []
    # for line in result:
    #     (position, value, percent) = tuple(line)
    #     if percent >= 0.7:
    #         car_number = new_car_number_format.findall(value)
    #         if len(car_number) == 0:
    #             car_number = old_car_number_format.findall(value)
    #         if len(car_number) != 0:
    #             car_number_list.append(car_number)
    # print(f'car_number_list : {car_number_list}')
    # # [0] 여기 바꿔야 겠음
    # if len(car_number_list) != 0:
    #     car_numbers.append(car_number_list[0])


def receive_image(src, q):
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    q.put(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(src)
            continue
        if frame is None:
            cap.release()
            cap = cv2.VideoCapture(src)
            continue
        q.put(frame)


def get_image(q):
    if q.empty() is not True:
        frame = q.get()
        q.queue.clear()
        return frame


def car_reco_test():
    # temp
    number = 0

    img_ori = cv2.imread('D:/test2.jpg')
    height, width, channel = img_ori.shape

    # grayscale
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    # morphology
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_top_hat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structuring_element)
    img_black_hat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structuring_element)
    img_grayscale_plus_top_hat = cv2.add(img_gray, img_top_hat)
    img_gray = cv2.subtract(img_grayscale_plus_top_hat, img_black_hat)
    # GaussianBlur (원본 이미지, 필터 크기, 표준 편차)
    img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
    # 이미지 구별 쉽게  (0, 255)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    # 윤곽선 그리기
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    # 모든 칸투어 찾기
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(contour_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []
    for contour in contours:
        # 칸투어로 사각형 그리기
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 번호판 크기 가정
    MIN_AREA = 80  # 최소 넓이 80? 200?
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소 위드 헤이트
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 가로 세로 최소 최대 비율

    # 번호판일 수 있는 칸투어
    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    # visualize possible contours
    contour_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in possible_contours:
        cv2.rectangle(contour_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255),
                      thickness=2)

    # 숫자 배열 모양 정하기
    MAX_DIAG_MULTIPLYER = 5  # diag 중심점끼리의 거리가 length의 5배 안쪽으로
    MAX_ANGLE_DIFF = 12.0  # contour 간 각도 최대
    MAX_AREA_DIFF = 0.5  # contour 간 면적 차이가 0.5 이하
    MAX_WIDTH_DIFF = 0.8 # contour 간 위드 차이가 0.8 이하
    MAX_HEIGHT_DIFF = 0.2 # contour 간 헤이트 차이가 0.2 이하
    MIN_N_MATCHED = 5  # 글자 최소 숫자

    # 문자 찾기 재귀함수
    def find_chars(contour_list):
        # 찾기 결과
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                # 같은 인덱스 칸투어는 비교 대상 X
                if d1['idx'] == d2['idx']:
                    continue

                # 중심점 사이의 거리
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                # 첫 번째 칸투어 대각선 길이
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                # 벡터 사이 거리
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

                # 각도 계산
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))

                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            # 최소 문자 개수 이하 제외
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            # 제외 칸투어
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            # 같은 값만 추출
            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            # 재귀함수
            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    # 문자를 찾은 index
    result_idx = find_chars(possible_contours)

    print(f"result_idx: {len(result_idx)}")

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            # cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255), thickness=2)

    # 수평 맞추기
    PLATE_WIDTH_PADDING = 1.3  # 위드 패딩
    PLATE_HEIGHT_PADDING = 1.5  # 헤이트 패딩
    MIN_PLATE_RATIO = 3  # 최소 비율
    MAX_PLATE_RATIO = 10  # 최대 비율

    # 차번 이미지, 정보 담기
    plate_images = []
    plate_information = []
    for i, matched_chars in enumerate(matched_result):
        # x 방향 순차 정렬
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        # x 중심 좌표
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        # y 중심 좌표
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        # 번호판 간격 삼각형 기준 세타각 구하기
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  # 삼각형 높이
        triangle_hypotenuse = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        # 라디안 값을 각도로
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenuse))
        # 수평 맞춰 이미지 회전
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        # 원본 -> 와핑된 이미지
        img_rotated = cv2.warpAffine(img_ori, M=rotation_matrix, dsize=(width, height))

        # 이미지 크롭
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        # 번호판 비율 확인
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO \
                or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_images.append(img_cropped)
        plate_information.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    print(f"plate_images len: {len(plate_images)}")

    # 차번 검출
    car_number_list = []
    for i, plate_img in enumerate(plate_images):

        # debug 스트 표시 부분
        # plt.imshow(plate_img)
        # plt.show()
        number += 1
        cv2.imwrite(rf'D:\temp\test{number}.jpg', plate_img)
        # debug 스트 표시 부분

        # case1 pytesseract
        # 파일로 저장 했다 읽으면 판독 성능이 올라 간다고 하여 적용
        cv2.imwrite(rf'D:\temp.jpg', plate_img)
        from PIL import Image
        image = Image.open(rf'D:\temp.jpg')
        config = '-l kor --oem 3 --psm 7'
        chars = pytesseract.image_to_string(image, config=config)
        car_number = new_car_number_format.findall(chars)
        if len(car_number) == 0:
            car_number = new_car_number_format2.findall(chars)
        if len(car_number) == 0:
            car_number = old_car_number_format.findall(chars)
        if len(car_number) != 0:
            car_number_list.append(car_number)

        # case2 easyOCR
        # reader = easyocr.Reader(['ko', 'en'], gpu=False)
        # result = reader.readtext(rf'D:\temp.jpg')
        # print(result)
        # for line in result:
        #     (position, value, percent) = tuple(line)
        #     if percent >= 0.7:
        #         car_number = new_car_number_format.findall(value)
        #         if len(car_number) == 0:
        #             car_number = new_car_number_format2.findall(value)
        #         if len(car_number) == 0:
        #             car_number = old_car_number_format.findall(value)
        #         if len(car_number) != 0:
        #             car_number_list.append(car_number)

        # print
    for car_number in car_number_list:
        print(car_number)


def load_image():
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    result = reader.readtext('D:/test.jpg')
    car_number_list = []
    for line in result:
        (position, value, percent) = tuple(line)
        if percent >= 0.7:
            car_number = new_car_number_format.findall(value)
            if len(car_number) == 0:
                car_number = new_car_number_format2.findall(value)
            if len(car_number) == 0:
                car_number = old_car_number_format.findall(value)
            if len(car_number) != 0:
                car_number_list.append(car_number)
    for car_number in car_number_list:
        print(car_number)


car_reco_test()