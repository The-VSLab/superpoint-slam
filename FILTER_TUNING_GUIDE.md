# SuperPoint 2D SLAM - 포인트 필터링 튜닝 가이드

## 문제점

2D 지도가 그림자와 하늘로 인해 잘못된 특징점을 잡아 지도가 왜곡되는 문제

## 해결방법

### 1. 새로운 필터들 추가됨

- **그림자 필터** (`filter_shadow_points`): 어두운 영역의 거짓 특징점 제거
- **밝기/명암비 필터** (`filter_by_brightness_contrast`): 불안정한 특징점 제거
- **하늘 필터 강화**: 더 정확한 색상/밝기 감지

### 2. 실행 방법

#### 기본 설정 (모든 필터 활성화)

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth
```

#### 필터 조정 예시

**그림자가 많은 환경** → 그림자 필터 강화:

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --enable_shadow_filter --enable_brightness_filter
```

**하늘이 많은 환경** → 하늘/밝기 필터 강화:

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --enable_sky_filter --enable_brightness_filter
```

**필터 비활성화** (기본 설정으로 복원):

PowerShell:

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --no-enable_shadow_filter --no-enable_sky_filter
```

cmd (개행):

```cmd
python scripts/superpoint_app_2D.py ^
    --mode slam2d ^
    --input test2.mp4 ^
    --weights weights/superpoint_v2_mobilenet.pth ^
    --no-enable_shadow_filter ^
    --no-enable_sky_filter
```

### 3. 필터 옵션 설명

| 옵션                            | 사용법               | 설명                      |
| ------------------------------- | -------------------- | ------------------------- |
| `--enable_sky_filter`           | 플래그만 사용 (값 X) | 하늘/구름 영역 필터링     |
| `--no-enable_sky_filter`        | 플래그만 사용 (값 X) | 하늘 필터링 비활성화      |
| `--enable_shadow_filter`        | 플래그만 사용 (값 X) | 그림자/어두운 영역 필터링 |
| `--no-enable_shadow_filter`     | 플래그만 사용 (값 X) | 그림자 필터링 비활성화    |
| `--enable_brightness_filter`    | 플래그만 사용 (값 X) | 밝기/명암비 기반 필터링   |
| `--no-enable_brightness_filter` | 플래그만 사용 (값 X) | 밝기 필터링 비활성화      |
| `--enable_line_filter`          | 플래그만 사용 (값 X) | 직선/전선 필터링          |
| `--enable_stat_filter`          | 플래그만 사용 (값 X) | 통계적 이상치 제거        |

**⚠️ 중요**: boolean 플래그는 값을 입력하지 않습니다!

- ❌ 잘못된 사용: `--enable_shadow_filter true`
- ✅ 올바른 사용: `--enable_shadow_filter`
- ✅ 비활성화: `--no-enable_shadow_filter`

### 4. 필터 설정값 수정 (Python 코드)

만약 더 정밀한 조정이 필요하면 `tracking/point_filter.py`의 다음 값들을 변경:

**그림자 필터** (line ~95):

```python
def filter_shadow_points(self, frame, kpts,
                        brightness_thresh: float = 0.2,  # ← 0.2에서 0.3으로 증가하면 더 엄격
                        saturation_thresh: float = 0.3):  # ← 증가하면 더 관대해짐
```

**하늘 필터** (line ~25):

```python
def filter_sky_points(self, frame, kpts,
                     brightness_thresh: float = 0.5):  # ← 조정하여 밝기 기준 변경
```

**밝기/명암비 필터** (line ~140):

```python
def filter_by_brightness_contrast(self, frame, kpts,
                                 min_brightness: float = 0.15,      # ← 최소 밝기
                                 max_brightness: float = 0.95,      # ← 최대 밝기
                                 min_contrast: float = 5.0):        # ← 최소 명암비
```

### 5. 권장 설정

#### 실외 쌜냥한 환경 (구름/하늘 많음)

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --enable_sky_filter --enable_brightness_filter --slam_conf_thresh 0.005 --slam_nms_dist 5
```

#### 실내 환경 (그림자 많음)

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --enable_shadow_filter --enable_brightness_filter --slam_conf_thresh 0.003 --slam_nms_dist 4
```

#### 균형잡힌 설정 (기본)

```powershell
python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --enable_sky_filter --enable_shadow_filter --enable_brightness_filter --enable_line_filter --enable_stat_filter --slam_conf_thresh 0.003 --slam_nms_dist 4
```

## 결과 확인

- **특징점 감소 ✓**: 잘못된 특징점이 제거되어야 함
- **지도 품질 개선 ✓**: 지도가 더 정확해졌는지 확인
- **궤적 안정성 ✓**: 궤적이 더 부드럽고 정확해졌는지 확인

## 추가 최적화

1. **conf_thresh 조정**: 신뢰도 임계값을 높여 애초부터 낮은 신뢰도 점 제거

   ```powershell
   python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --slam_conf_thresh 0.01
   ```

2. **max_kpts 감소**: 더 적은 수의 고품질 특징점 사용

   ```powershell
   python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --max_kpts 300
   ```

3. **nms_dist 증가**: 밀집된 특징점 제거
   ```powershell
   python scripts/superpoint_app_2D.py --mode slam2d --input test2.mp4 --weights weights/superpoint_v2_mobilenet.pth --slam_nms_dist 5
   ```

## 디버깅

특징점 필터링 전/후 분석이 필요하면:

1. `visual_slam_2d.py` 라인 342에서 로깅 추가:

   ```python
   print(f"Before filter: {len(kpts)} points")
   kpts_filtered, desc_filtered = self.point_filter.apply_all_filters(...)
   print(f"After filter: {len(kpts_filtered)} points")
   ```

2. 각 필터별 효과 분석이 필요하면 `apply_all_filters` 각 단계의 마스크 출력 추가
