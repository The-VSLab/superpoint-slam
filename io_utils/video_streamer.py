import glob
import os

import cv2
import numpy as np


class VideoStreamer(object):
    """이미지 스트림 처리를 도와주는 클래스. 세 가지 유형의 입력 가능:
      1.) USB 웹캠
      2.) 이미지 디렉토리 ('img_glob'과 일치하는 디렉토리 내 파일들)
      3.) 비디오 파일 (예: .mp4 또는 .avi 파일)
    """

    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        # "basedir" 문자열이 "camera"라는 단어이면 웹캠 사용
        if basedir == "camera/" or basedir == "camera":
            print("==> 웹캠 입력 처리 중.")
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # 비디오로 열기 시도
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4 : len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == ".mp4"):
                raise IOError("비디오 파일을 열 수 없습니다")
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != ".txt"):
                print("==> 비디오 입력 처리 중.")
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[:: self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print("==> 이미지 디렉토리 입력 처리 중.")
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[:: self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError(
                        "이미지를 찾을 수 없습니다 (잘못된 '--img_glob' 파라미터일 수 있음)"
                    )

    def read_image(self, impath, img_size):
        """이미지를 흑백으로 읽고 img_size로 크기 조정합니다.
        입력
          impath: 입력 이미지 경로
          img_size: 크기 조정 크기를 지정하는 (W, H) 튜플
        반환
          grayim: [0, 1] 범위의 값을 가진 H x W 크기의 float32 numpy 배열
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception("이미지 읽기 오류 %s" % impath)
        # OpenCV를 통해 이미지 크기 조정
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = (grayim.astype("float32") / 255.0)
        return grayim

    def next_frame(self):
        """다음 프레임을 반환하고 내부 카운터를 증가시킵니다.
        반환
           image: 다음 H x W 이미지
           status: 이미지가 로드되었는지 여부에 따른 True 또는 False
        """
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if ret is False:
                print("VideoStreamer: 카메라에서 이미지를 가져올 수 없습니다 (잘못된 --camid일 수 있음)")
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(
                input_image, (self.sizer[1], self.sizer[0]), interpolation=cv2.INTER_AREA
            )
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            input_image = input_image.astype("float") / 255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # 내부 카운터 증가
        self.i = self.i + 1
        input_image = input_image.astype("float32")
        return (input_image, True)
