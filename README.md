# MobileNet 기반 SuperPoint SLAM 프론트엔드

## 초록 (Abstract)
본 프로젝트는 SuperPoint 기반의 시각 SLAM 프론트엔드를 경량화하기 위해,
기존의 연산량이 큰 VGG 계열 백본을 MobileNet 구조로 대체한
경량 SuperPoint 프론트엔드를 제안한다.
제안한 프론트엔드는 ORB-SLAM에서 사용되는 ORB 기반 특징점 추출 및 매칭
모듈을 대체하도록 설계되었으며,
기존 SLAM 백엔드(Tracking, Pose Estimation, Bundle Adjustment, Mapping)는
변경하지 않고 유지한다.
이를 통해 복잡한 환경에서도 향상된 강인성과
임베디드 환경에서의 실시간 처리 가능성을 동시에 확보하는 것을 목표로 한다.

---

## 1. 서론 (Introduction)

ORB-SLAM과 같은 기존 시각 SLAM 시스템은 ORB와 같은 수작업 특징점에
의존하고 있으며, 이는 연산 효율은 높지만
저텍스처 환경, 조명 변화, 모션 블러 등
도전적인 시각 환경에서는 성능 저하가 발생한다.

SuperPoint는 딥러닝 기반 특징점 검출 및 기술자 추출 기법으로,
이러한 환경 변화에 대해 높은 강인성을 보인다.
그러나 SuperPoint의 원본 구조는 VGG 계열의 무거운 백본을 사용하여
임베디드 및 실시간 SLAM 환경에 적용하기에는
연산량 측면에서 한계가 있다.

본 프로젝트에서는 이러한 문제를 해결하기 위해
MobileNet 기반의 경량 SuperPoint 프론트엔드를 설계하고,
이를 ORB-SLAM의 프론트엔드로 통합하는 방안을 제시한다.

---

## 2. 연구 동기 (Motivation)

본 연구의 동기는 다음과 같다.

- 환경 변화에 강인한 특징점 추출 기법의 SLAM 적용
- 임베디드 환경에서의 실시간 SLAM 구현 가능성 확보
- 기존 SLAM 백엔드 구조를 유지하면서 프론트엔드 성능 향상

---

## 3. 제안 방법 (Methodology)

### 3.1 네트워크 구조

제안하는 SuperPoint 프론트엔드는
기존 SuperPoint 구조를 기반으로 다음과 같이 설계되었다.

- 기존 VGG 계열 백본 → MobileNet 백본으로 대체
- 공간 해상도 유지를 위한 8배 다운샘플링 구조
- SuperPoint 방식의 Detector Head (65 채널 신뢰도 맵)
- SuperPoint 방식의 Descriptor Head (256차원 실수형 기술자)

### 3.2 특징점 검출 및 기술자 생성

- Detector Head에서 출력된 신뢰도 맵을 기반으로
  Non-Maximum Suppression(NMS) 및 임계값 필터링을 수행
- 검출된 특징점 위치에서 기술자를 샘플링
- 모든 기술자는 L2 정규화를 수행하여 매칭 안정성 확보

---

## 4. 시스템 통합 (System Integration)

### 4.1 SLAM 프론트엔드 대체 구조

제안한 시스템은 ORB-SLAM의 프론트엔드를 다음과 같이 대체한다.

- ORBextractor → MobileNet 기반 SuperPoint 특징점 추출기
- ORBmatcher → L2 거리 기반 실수형 기술자 매칭기

SLAM의 백엔드 구성 요소인
Tracking, Pose Estimation, Bundle Adjustment, Mapping은
기존 ORB-SLAM 구조를 그대로 유지한다.

### 4.2 전체 시스템 파이프라인

입력 영상  
→ MobileNet-SuperPoint (특징점 및 기술자 추출)  
→ 실수형 기술자 매칭  
→ ORB-SLAM Tracking 및 초기화  
→ 카메라 자세 추정 및 지도 생성  

---

## 5. 구현 환경 및 세부 사항 (Implementation Details)

- 개발 프레임워크: PyTorch
- 특징점 기술자 차원: 256차원 (실수형)
- 매칭 방식: L2 거리 기반 매칭 + Ratio Test + 상호 일관성 검사
- 임베디드 환경에서의 실시간 동작을 고려한 경량 설계

---

## 6. 실험 결과 (Experimental Results)

실험 결과, 제안한 MobileNet 기반 SuperPoint 프론트엔드는
기존 ORB 기반 프론트엔드 대비
저텍스처 및 환경 변화가 큰 장면에서
보다 안정적인 특징점 검출 및 추적 성능을 보였다.

프레임 처리 속도(FPS)와 궤적 정확도(ATE)를 기준으로 한
정량적 성능 평가는
첨부된 논문 및 보고서에서 상세히 기술한다.

---

### 6.1 성능 비교 (Performance Comparison)

본 절에서는 제안한 MobileNet 기반 SuperPoint 프론트엔드와
기존 ORB-SLAM 프론트엔드의 성능을 비교한다.
비교 평가는 동일한 데이터셋과 환경에서 수행되었다.

| 방법 | 특징점 종류 | 평균 FPS | 프레임당 처리 시간 (ms) | 궤적 오차 (ATE, m) | 저텍스처 환경 안정성 | 임베디드 실시간성 |
|-----|------------|----------|--------------------------|-------------------|----------------------|------------------|
| ORB-SLAM (Baseline) | ORB | 9.6 | 104 | 0.27 | 낮음 | 가능 |
| SuperPoint (원본) | SuperPoint (VGG) | 2.5 | 400 | **0.18** | 매우 높음 | 불가능 |
| **제안 방법** | **MobileNet-SuperPoint** | **5.0 ~ 9.0** | **110 ~ 200** | **0.19** | 높음 | 가능 |

※ FPS 및 처리 시간은 단일 카메라 기준 평균값이며,
ATE는 공개 데이터셋에서의 평균 절대 궤적 오차를 의미한다.

※ 상기 수치는 초기 실험 결과를 기반으로 하며,
최종 성능 평가는 향후 추가 실험을 통해 보완될 예정이다.

---

## 7. 한계점 및 향후 연구 (Limitations & Future Work)

- 루프 클로저 및 재지역화 성능은
  기술자 매칭 전략에 따라 영향을 받을 수 있음
- 초저전력 임베디드 환경을 위해
  양자화(Quantization) 및 추가 최적화 필요
- SuperPoint 학습 코드 및 데이터셋은 포함하지 않음

---

## 8. 결론 (Conclusion)

본 프로젝트는 경량화된 딥러닝 기반 특징점 추출 기법이
기존의 고전적 SLAM 시스템에 효과적으로 결합될 수 있음을 보였다.
MobileNet 기반 SuperPoint 프론트엔드를 통해
강인성과 실시간 처리 성능 간의 균형을 달성하였으며,
임베디드 환경에서의 SLAM 적용 가능성을 제시한다.

---

## 9. 라이선스 (License)

본 프로젝트는 MIT License 하에 배포된다.

---

## 10. 감사의 글 (Acknowledgements)

본 연구는 다음의 연구를 기반으로 수행되었다.

- DeTone et al., “SuperPoint: Self-Supervised Interest Point Detection and Description”,
  CVPR Workshop, 2018.
- Mur-Artal et al., “ORB-SLAM: A Versatile and Accurate Monocular SLAM System”.

본 저장소는 ORB-SLAM 코드를 포함하지 않으며,
사용자는 ORB-SLAM을 각자의 라이선스에 따라 별도로 획득해야 한다.