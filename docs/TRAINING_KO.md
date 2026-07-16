# SuperPoint (MobileNet) 학습 가이드

현재 레포의 학습 진입점은 두 가지입니다.

| 스크립트 | 용도 | 출력 |
|----------|------|------|
| `learning/finetune_descriptor.py` | **디스크립터 헤드만 재증류** (검출/트래킹 성능 보존, ~2분) | `weights/v14_desc_ft.pth` |
| `learning/train_superpoint.py` | 전체 teacher-student 증류 (2-Phase) | `checkpoints/v14_latest.pth` |

> 과거 문서에 있던 `scripts/train_synthetic.py`, `scripts/train_superpoint.py`는 제거된 스크립트입니다.

---

## 1) 디스크립터 헤드 재증류 (권장 진입점)

### 배경: 디스크립터 붕괴 사고

`weights/v14_latest.pth`는 검출 성능은 정상이지만 **디스크립터 헤드가 붕괴**된
상태였습니다 (한 프레임 안의 키포인트 기술자들이 사실상 동일 벡터 — 유사도
p95=0.999, 인접 프레임 매칭 inlier 7%). 이 상태에서는 루프 클로저·재지역화 등
기술자 매칭 기반 기능이 전부 동작하지 않습니다.

### 복구 방법

백본과 검출 헤드를 **동결**하고 디스크립터 헤드(convDa/convDb, 74.6K 파라미터)만
재초기화하여 원본 SuperPoint(VGG) teacher로부터 다시 증류합니다.

```bash
./.venv/bin/python learning/finetune_descriptor.py \
    --epochs 2 --batch_size 8 --stride 3 --lr 1e-3
```

- **데이터**: `dataset/training/<seq>/image_0` (KITTI 그레이스케일) — teacher 증류라 **라벨 불필요**
- 평가 시퀀스(07, 08)는 기본적으로 학습에서 제외 (`--exclude`)
- Teacher: `weights/superpoint_v1.pth` (MagicLeap 오리지널)
- 결과 (KITTI 07 실측): 인접 프레임 inlier 7% → **89%**, 루프 클로저 검출 성공

### 검증 방법

학습 후 인접 프레임 매칭 품질을 확인하세요 — Essential inlier 비율이 70%를
넘어야 정상입니다. 60% 미만이면 epochs를 늘리거나 lr을 낮춰 재시도.

---

## 2) 전체 학습: 2-Phase Teacher-Student 증류

SuperPoint VGG 원본(Teacher)의 성능을 MobileNetV2(Student)로 이식하는 2단계 학습입니다.

### 왜 2단계인가

MobileNetV2 백본은 ImageNet 사전학습 지능(BatchNorm 통계 등)을 갖고 있습니다.
처음부터 백본을 열고 학습하면 무작위 초기화된 Head의 거대한 초기 그래디언트가
백본을 오염시키는 **Catastrophic Forgetting**이 발생합니다.

### Phase 1: 뼈대 동결 (Frozen Backbone)

- `learning/train_config.yml`: `freeze_backbone: true`, `epochs: 20`, `lr: 1.0e-4`
- Head(Detector + Descriptor)만 학습 → 안정된 기초 가중치

### Phase 2: 정밀 튜닝 (Safe Fine-Tuning)

- `freeze_backbone: false` (BatchNorm은 코드에서 자동 동결 유지)
- `resume: true`, `resume_from:` Phase 1 가중치
- `epochs: 30`, `lr: 1.0e-5`

### 실행

```bash
uv run python learning/train_superpoint.py --config learning/train_config.yml
```

### ⚠️ 디스크립터 붕괴 방지 체크리스트 (필독)

과거 v14_latest의 붕괴 사고를 반복하지 않기 위한 안전장치:

1. **`use_teacher_desc: true` + `desc_weight > 0` 확인** — desc loss가 0이면
   디스크립터 헤드에 그래디언트가 전혀 가지 않아 붕괴합니다.
2. 현재 코드는 teacher가 desc를 반환하지 않으면 **에러를 발생**시킵니다
   (조용한 실패 방지 가드 적용됨).
3. 학습 중 로그의 **`desc=` 손실이 0에 머물면 즉시 중단**하고 원인 확인.
4. 학습 후 반드시 인접 프레임 매칭 테스트 (위 1절 검증 방법)로 확인.
5. `resume_from`으로 다른 실험의 체크포인트를 이어받을 때는 그 체크포인트의
   디스크립터 품질부터 검증할 것 (붕괴 상태를 상속받을 수 있음).

### 참고

- Teacher 가중치: `weights/superpoint_v1.pth` (`weights/` 우선 탐색, 루트 폴백)
- Validation에서 `Prec`(정밀도)·`MaxP`(최대 확신도) > 0.8 상승을 모니터링
- `val_dir: dataset/test` (KITTI 테스트 스플릿) — desc/det val loss 추적용
