from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, BartForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from datasets import Dataset
import pandas as pd
import torch
import warnings
import os
import pickle
import whisper

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Whisper(STT) 모델 로드
def load_stt_model(config):
    return whisper.load_model(config['stt_whisper_model'])

def set_model(model_name='monologg/kobert', fine_tuning_weight=None):
    """모델과 토크나이저를 초기화하고 필요한 경우 가중치를 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # KoBERT 토큰 구조에 맞춰 토큰 설정
    tokenizer.eos_token = "[SEP]"
    tokenizer.eos_token_id = 3

    if fine_tuning_weight:
        # 저장된 체크포인트에서 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuning_weight, num_labels=7,
                                                                   local_files_only=True)
    else:
        # 사전 학습된 기본 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
    return model, tokenizer


def preprocess_fn(dataset, tokenizer, max_length):
    """텍스트 데이터 토큰화하는 전처리 함수"""
    # 텍스트 정제 (NaN 방지 및 끝부분 공백 정리)
    if isinstance(dataset, str):
        inputs = [dataset.strip().rstrip('.')]
    else:
        inputs = [str(t).strip().rstrip('.') for t in dataset["text"]]

    return tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=True,
        return_tensors='pt'
    )

def compute_metrics(pred):
    """모델 성능 측정을 위한 지표 계산 (Accuracy, F1, Recall, Precision)"""
    labels = pred.label_ids
    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else:
        logits = pred.predictions

    preds = logits.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision_res, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {'accuracy': acc, 'f1': f1, 'recall':recall, 'precision':precision_res}

def fine_tuning(model, tokenizer, config, dataset=None):
    """fine tuning용 데이터를 활용하여 기본 Kobert 모델 학습을 수행"""
    if dataset is None:
        try:
            dataset = pd.read_csv(config['save_data_path'] + 'fine_tuning_data.tsv', sep='\t')
        except:
            print('데이터셋 파일을 찾을 수 없어 학습을 중단합니다.')
            return

    # 전처리 및 데이터셋 분할 (9:1)
    fine_tuning_ds = Dataset.from_pandas(dataset).map(preprocess_fn, batched=True, fn_kwargs={
        "tokenizer": tokenizer,
        "max_length": config['max_length'],
    })
    fine_tuning_ds = fine_tuning_ds.train_test_split(test_size=0.1)

    # 학습 파라미터 설정
    training_args_stage1 = TrainingArguments(
        seed=config['seed'],
        output_dir="./Result/kobert_fine_tuning",
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        logging_steps=50,
        metric_for_best_model = "accuracy",
        greater_is_better = True,
        save_total_limit = 1,
    )
    trainer_stage1 = Trainer(
        model=model,
        args=training_args_stage1,
        train_dataset=fine_tuning_ds["train"],
        eval_dataset=fine_tuning_ds["test"],
        compute_metrics=compute_metrics
    )

    trainer_stage1.train()
    model.save_pretrained("./Result/kobert_fine_tuning_final")
    print('KoBERT 파인튜닝 모델이 "./Result/kobert_fine_tuning_final"에 저장이 완료되었습니다.')

def target_evaluation(tokenizer, config, dataset=None):
    """실증 데이터를 활용하여 2차 학습 및 결과"""
    if dataset is None:
        try:
            dataset = pd.read_csv(config['save_data_path']+'test_data.tsv', sep='\t')
        except:
            print('데이터셋 파일을 찾을 수 없어 학습을 중단합니다.')
            return

    gkf = GroupKFold(n_splits=config['fold'])
    subjects = dataset['subject'].values

    best_accuracy = 0.0
    final_model_save_path = os.path.join("./Result/", "kobert_final_weight")
    fold_results = []
    k_fold_sub = []
    pred_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(dataset, groups=subjects)):
        print(f"\n# ====== Fold {fold + 1} 시작 ===== #")

        # 전처리 및 데이터셋 분할 (9:1)
        train_fold_df = dataset.iloc[train_idx]
        val_fold_df = dataset.iloc[val_idx]

        train_fold_ds = Dataset.from_pandas(train_fold_df).map(preprocess_fn, batched=True, fn_kwargs={
        "tokenizer": tokenizer,
        "max_length": config['max_length'],
    })
        val_fold_ds = Dataset.from_pandas(val_fold_df).map(preprocess_fn, batched=True, fn_kwargs={
        "tokenizer": tokenizer,
        "max_length": config['max_length'],
    })
        model = AutoModelForSequenceClassification.from_pretrained(
            config['fine_tuning_weight'],
            num_labels=7
        )

        training_args_fold = TrainingArguments(
            seed=config['seed'],
            output_dir=f"./Result/fold/kobert_fold_{fold + 1}",
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=config['learning_rate'],

            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,

            label_smoothing_factor=0.1,
            fp16=True,
            gradient_checkpointing=True,
            logging_steps=10
        )

        # 4. 트레이너 실행
        trainer = Trainer(
            model=model,
            args=training_args_fold,
            train_dataset=train_fold_ds,
            eval_dataset=val_fold_ds,
            compute_metrics=compute_metrics
        )

        trainer.train()
        check_fold_df = dataset.iloc[val_idx]
        subject_fold_map = check_fold_df['subject'].unique().tolist()

        eval_result = trainer.evaluate()
        current_accuracy = eval_result['eval_accuracy']
        # Best 모델 체크 및 저장
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            model.save_pretrained(final_model_save_path)
            print(f"** Fold {fold + 1}이 현재까지 최고 성능({best_accuracy:.4f})을 기록하여 저장되었습니다. **")

        fold_results.append({'fold':fold + 1,
                             'accuracy':eval_result['eval_accuracy'],
                             'f1_score': eval_result['eval_f1'],
                             'recall': eval_result['eval_recall'],
                             'precision': eval_result['eval_precision']})
        k_fold_sub.append(subject_fold_map)
        pred_result = trainer.predict(val_fold_ds)
        pred_results.append(pred_result)
    with open('./Result/fold/k_fold_sub.plk', 'wb') as f:
        pickle.dump(k_fold_sub, f)
    with open('./Result/fold/predict_results.plk', 'wb') as f:
        pickle.dump(pred_results, f)
    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv('./Result/fold/kobert_fold_results.csv', index=False)
    print(f"\n최종 모델이 {final_model_save_path} 에 저장되었습니다. (Best Accuracy: {best_accuracy:.4f})")

def predict_result(audio_path, config):

    final_model_path = os.path.join("Result/kobert_final_weight")

    if not os.path.exists(final_model_path):
        print(f"최종 모델을 찾을 수 없습니다: {final_model_path}")
        print("Target Evaluation를 먼저 완료하여 모델을 저장해주세요.")
        return

    model, tokenizer = set_model(config['bert_model'],final_model_path )
    model.eval()

    stt_model = load_stt_model(config)
    STT = stt_model.transcribe(audio_path, language='ko')['text'].strip()
    token = preprocess_fn(STT, tokenizer, config['max_length'])

    with torch.no_grad():
        outputs = model(**token)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    # 4. 결과 출력 (ID를 감정 텍스트로 변환)
    # config.json의 emotion_map을 역순으로 탐색하여 감정 이름을 가져옵니다.
    id_to_emotion = {v: k for k, v in config['emotion_map'].items()}
    result_emotion = id_to_emotion.get(prediction, "Unknown")

    print(f"\n# ====== 예측 결과 ====== #")
    print(f"입력 파일: {audio_path}")
    print(f"분석 문장: {STT}")
    print(f"예측 감정: {result_emotion} (ID: {prediction})")
    print(f"========================== #")

    return result_emotion, logits

def predict_result_without_stt(text, config):
    print(type(text))
    final_model_path = os.path.join("Result/kobert_final_weight")

    if not os.path.exists(final_model_path):
        print(f"최종 모델을 찾을 수 없습니다: {final_model_path}")
        print("Target Evaluation를 먼저 완료하여 모델을 저장해주세요.")
        return

    model, tokenizer = set_model(config['bert_model'],final_model_path )
    model.eval()
    token = preprocess_fn(str(text), tokenizer, config['max_length'])

    with torch.no_grad():
        outputs = model(**token)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    # 4. 결과 출력 (ID를 감정 텍스트로 변환)
    # config.json의 emotion_map을 역순으로 탐색하여 감정 이름을 가져옵니다.
    id_to_emotion = {v: k for k, v in config['emotion_map'].items()}
    result_emotion = id_to_emotion.get(prediction, "Unknown")
    print(f"\n# ====== 예측 결과 ====== #")
    print(f"문장: {str(text)}")
    print(f"예측 감정: {result_emotion} (ID: {prediction})")
    print(f"========================== #")

    return result_emotion, logits