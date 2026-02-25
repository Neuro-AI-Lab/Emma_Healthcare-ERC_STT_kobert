# STT 기반 감정 분류 플랫폼 (STT-Emotion_Model)

본 프로젝트는 음성 데이터를 전사(STT)하고, 이를 KoBERT 모델을 활용하여 7가지 감정으로 분류하는 통합 파이프라인입니다. 시계열 데이터의 특징을 추출하고, 전이 학습(Transfer Learning)을 통해 타겟 데이터셋에 최적화된 성능을 도출하도록 설계되었습니다.

---
- **참여 기관**: 광운대학교 신경공학 및 인공지능 연구실 (NeuroAI Lab)
- **개발자**:
  - 이준영 연구원
    - leejykw2025@kw.ac.kr
  - 이동혁 연구원
  - 김대현 연구원
- **활용 모델**:
  - **STT**: OpenAI Whisper
  - **Emotion Classification**: Pre-train KoBert
---

## 1. 감정 분류 클래스 (7개)
데이터셋은 다음 7가지 감정을 분류하도록 구성되어 있습니다:
- `happy` (0), `neutral` (1), `fearful` (2), `disgust` (3), `surprise` (4), `sad` (5), `angry` (6)

## 2. Configuration 설정 (config.json)
프로젝트의 주요 파라미터 및 경로는 `config.json`에서 관리됩니다:

| Parameter | Description                     |
| :--- |:--------------------------------|
| `bert_model` | 기본으로 사용할 사전 학습된 BERT 모델명        |
| `fine_tuning_weight` | 전이 학습된 가중치가 저장되거나 로드될 경로        |
| `stt_whisper_model` | 사용할 Whisper 모델의 크기 (기본: medium) |
| `fine_tuning_data_path` | 전이 학습용 원천 데이터셋 경로               |
| `save_data_path` | 전처리가 완료된 TSV 파일들이 저장될 경로        |
| `max_length` | 모델 입력 텍스트의 최대 길이 (기본: 128)      |
| `batch_size` | 학습 시 사용할 배치 크기                  |
| `learning_rate` | 모델 최적화를 위한 학습률                  |
| `epochs` | 학습 반복 횟수                        |
| `fold` | 교차 검증을 위한 K-Fold 수              |

## 3. Main.py Argument 상세 설명

`main.py`는 명령행 인자를 통해 전처리, 학습, 평가를 제어합니다.

| Argument                    | Type | Default | 상세 기능 설명                                   |
|:----------------------------| :--- | :--- |:-------------------------------------------|
| `--set_config`              | `str` | `config.json` | 설정(경로, 시드, 파라미터) 파일 지정                     |
| `--create_fine_tuning_data` | `flag` | - | AI HUB 데이터를 전이 학습용 TSV로 전처리                |
| `--create_test_data`        | `flag` | - | 테스트 음성 파일을 Whisper로 전사하여 데이터셋 구축           |
| `--set_bert_model`          | `flag` | - | KoBERT 모델 및 토크나이저 초기화                      |
| `--set_fine_tuning_weight`  | `flag` | - | 1차 학습 가중치(`fine_tuning_weight`) 로드 활성화     |
| `--fine_tuning`             | `flag` | - | **[Base Tuning]** AI HUB 데이터를 이용한 전이 학습 수행 |
| `--target_evaluation`       | `flag` | - | **[Target Tuning]** K-Fold 검증 및 최적 모델 추출   |
| `--predict`                 | `str` | - | 특정 `.wav` 파일의 감정 상태를 즉시 예측                 |
| `--predict_no_str`          | `str` | - | 특정 텍스트 str의 감정 상태를 즉시 예측                   |

## 4. 학습 알고리즘 및 최적 모델 저장 (src/train.py)

프로젝트는 2단계 미세 조정(Fine-tuning) 전략을 취합니다:

### Stage 1: Base Fine-tuning (전이 학습)
* **목적**: 범용 언어 모델을 감정 분석 도메인에 적응시킴.
* **결과물**: `./Result/kobert_fine_tuning_final/`에 가중치 저장.

### Stage 2: Target Evaluation (Group K-Fold)
* **방식**: 피험자(Subject)를 기준으로 데이터를 분할하여 5-Fold 검증 수행.
* **최적 모델 저장**: 각 Fold의 성능을 비교하여 **가장 높은 Accuracy를 기록한 가중치**를 자동으로 선별하여 `kobert_final_weight` 경로에 저장합니다.

## 5. 실행 가이드 (Usage)
- **예측 시**: 별도의 학습 없이 결과만 확인하고 싶으시다면 `Step 3`를 바로 실행하시면 됩니다.
- **학습 데이터 추가**: 학습데이터를 추가하고 싶다면 Dataset/test_dataset/ 폴더에 데이터 추가후 `step 2`부터 진행하시면 됩니다.
### Step 1: 필요한 라이브러리 설치
- 현재 본 모델은 `Python 3.8`을 사용하고 있습니다.
- 사전에 필요한 라이브러리는 `requirements.txt`에 저장해 두었습니다.
- 최종 학습 모델 가중치가 필요합니다. 아래의 구글드라이브 Zip파일을 해제하여 Result 폴더에 넣어주세요.
- https://drive.google.com/file/d/1IJ0sZnrBkjMpm_FbTu7un4X3WYjCyAgd/view?usp=drive_link
```bash
pip install -r requirements.txt
```
### Step 1: 데이터 전처리 및 전이 학습
- 이미 fine_tuning data가 Dataset/preprocess_data/ 폴더에 들어있다면 --create_fine_tuning_data 인자는 제외후 사용하시면 됩니다.
```bash
python main.py --create_fine_tuning_data --set_bert_model --fine_tuning
```

### Step 2: 타겟 데이터 평가 및 최적 가중치 추출
- 음성 데이터를 전사하고, K-Fold 교차 검증을 통해 가장 성능이 좋은 모델을 저장합니다.
- - 이미 test data가 Dataset/preprocess_data/ 폴더에 들어있다면 --create_test_data 인자는 제외후 사용하시면 됩니다.
```bash
python main.py --create_test_data --set_bert_model --set_fine_tuning_weight --target_evaluation
```

### Step 3: 실시간 감정 예측
- 학습된 모델을 사용하여 특정 음성 데이터의 감정을 즉시 분석합니다.

**음성**
```bash
python main.py --predict "음성 데이터 경로"
```
**텍스트**
```bash
python main.py --predict "음성 데이터 경로"
```
## 6. 결과 산출물 (Results)
- 최적 가중치: ./Result/kobert_final_weight/ (Best Fold Model)
- 평가 지표: ./Result/fold/kobert_fold_results.csv
- 예측 및 피험자 정보: predict_results.plk, k_fold_sub.plk


