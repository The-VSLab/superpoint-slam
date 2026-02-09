import os
import sys
from pathlib import Path

import torch
import time

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.superpoint_frontend import SuperPointFrontend
from io_utils.video_streamer import VideoStreamer
from tracking.point_tracker import PointTracker, myjet


def add_demo_args(parser):
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="실행 장치 선택 (auto/cpu/cuda/mps)",
    )
    parser.add_argument("--img_glob", type=str, default="*.png")
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--show_extra", action="store_true")
    parser.add_argument("--H", type=int, default=120)
    parser.add_argument("--W", type=int, default=160)
    parser.add_argument("--display_scale", type=int, default=2)
    parser.add_argument("--min_length", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=5)
    parser.add_argument("--nms_dist", type=int, default=4)
    parser.add_argument("--conf_thresh", type=float, default=0.015)
    parser.add_argument("--nn_thresh", type=float, default=0.7)
    parser.add_argument("--camid", type=int, default=0)
    parser.add_argument("--waitkey", type=int, default=1)
    parser.add_argument("--cuda", action="store_true", help="(호환) CUDA 사용")
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--write_dir", type=str, default="tracker_outputs/")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--save_npy_dir", type=str, default="npy_outputs/")


def select_device(device_opt, cuda_flag):
    if cuda_flag:
        return "cuda"
    if device_opt != "auto":
        return device_opt
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_demo(opt):
    # 다양한 소스에서 입력 이미지를 로드하는 데 도움이 되는 클래스
    vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    device = select_device(opt.device, opt.cuda)
    use_cuda = device == "cuda"

    print("==> 사전 학습된 네트워크 로딩 중.")
    # SuperPoint 네트워크를 실행하고 출력을 처리하는 클래스
    fe = SuperPointFrontend(
        weights_path=opt.weights,
        nms_dist=opt.nms_dist,
        conf_thresh=opt.conf_thresh,
        nn_thresh=opt.nn_thresh,
        cuda=use_cuda,
    )
    print("==> 사전 학습된 네트워크 로딩 완료.")

    # 연속된 점 매칭을 트랙으로 병합하는 데 도움이 되는 클래스
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

    # 데모를 표시할 창 생성
    if not opt.no_display:
        win = "SuperPoint V2 Tracker (MobileNet)"
        cv2.namedWindow(win)
    else:
        print("시각화 건너뛰기, GUI를 표시하지 않습니다.")

    # 시각화를 위한 폰트 파라미터
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    # 원하는 경우 출력 디렉토리 생성
    if opt.write:
        print("==> 출력을 %s에 저장합니다" % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)
    if opt.save_npy:
        if not os.path.exists(opt.save_npy_dir):
            os.makedirs(opt.save_npy_dir)

    print("==> 데모 실행 중.")
    while True:

        start = time.time()

        # 새로운 이미지 가져오기
        img, status = vs.next_frame()
        if status is False:
            break

        # 특징점과 디스크립터 가져오기
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()

        # 추적기에 특징점과 디스크립터 추가
        tracker.update(pts, desc)

        # 모든 프레임에서 성공적으로 매칭된 점들의 트랙 가져오기
        tracks = tracker.get_tracks(opt.min_length)

        # 주요 출력 - 입력 이미지 위에 점 트랙 오버레이 표시
        out1 = (np.dstack((img, img, img)) * 255.0).astype("uint8")
        if tracks.shape[0] > 0:
            tracks[:, 1] /= float(fe.nn_thresh)  # 트랙 점수를 [0,1]로 정규화
            tracker.draw_tracks(out1, tracks)
        if opt.show_extra:
            cv2.putText(out1, "Point Tracks", font_pt, font, font_sc, font_clr, lineType=16)

        # 추가 출력 -- 현재 점 검출 표시
        out2 = (np.dstack((img, img, img)) * 255.0).astype("uint8")
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, "Raw Point Detections", font_pt, font, font_sc, font_clr, lineType=16)

        # 추가 출력 -- 점 신뢰도 히트맵 표시
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 0.00001)
            out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype("int"), :]
            out3 = (out3 * 255).astype("uint8")
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, "Raw Point Confidences", font_pt, font, font_sc, font_clr, lineType=16)

        # 최종 출력 크기 조정
        if opt.show_extra:
            out = np.hstack((out1, out2, out3))
            out = cv2.resize(out, (3 * opt.display_scale * opt.W, opt.display_scale * opt.H))
        else:
            out = cv2.resize(out1, (opt.display_scale * opt.W, opt.display_scale * opt.H))

        # 화면에 시각화 이미지 표시
        if not opt.no_display:
            cv2.imshow(win, out)
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord("q"):
                print("종료, 'q' 키가 눌렸습니다.")
                break

        # 선택적으로 이미지를 디스크에 저장
        if opt.write:
            out_file = os.path.join(opt.write_dir, "frame_%05d.png" % vs.i)
            print("이미지를 %s에 저장 중" % out_file)
            cv2.imwrite(out_file, out)
        # 선택적으로 npy 결과 저장
        if opt.save_npy:
            base = os.path.join(opt.save_npy_dir, "frame_%05d" % vs.i)
            np.save(base + "_pts.npy", pts)
            if desc is not None:
                np.save(base + "_desc.npy", desc)
            if heatmap is not None:
                np.save(base + "_heatmap.npy", heatmap)

        end = time.time()
        net_t = 1.0 / float(end1 - start1)
        total_t = 1.0 / float(end - start)
        if opt.show_extra:
            print(
                "이미지 %d 처리 완료 (네트워크+후처리: %.2f FPS, 전체: %.2f FPS)."
                % (vs.i, net_t, total_t)
            )

    # 남아있는 모든 창 닫기
    cv2.destroyAllWindows()

    print("==> 데모 완료.")
