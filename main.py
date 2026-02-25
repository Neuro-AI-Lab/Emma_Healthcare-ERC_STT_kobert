import argparse
import torch
import random
import numpy as np
import os
import json
from src.train import set_model, fine_tuning, target_evaluation, predict_result, predict_result_without_stt
from src.create_dataset import create_ai_hub_fine_tuning_data, create_test_data

def set_seed(seed=2025):
    """실험 결과의 재현성을 위해 난수 시드를 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드가 {seed}로 고정되었습니다.")



def main():
    fine_tuning_data = None
    tokenizer = None
    test_data = None
    model = None

    parser = argparse.ArgumentParser()
    # 실행 옵션 설정
    parser.add_argument('--set_config', type=str, default='config.json', help='설정 파일 경로')
    parser.add_argument('--create_fine_tuning_data', action='store_true', help='학습 데이터 생성')
    parser.add_argument('--create_test_data', action='store_true', help='테스트 데이터 생성')
    parser.add_argument('--set_bert_model', action='store_true', help='BERT 모델 로드 활성화')
    parser.add_argument('--set_fine_tuning_weight', action='store_true', help='학습된 가중치 적용 여부')
    parser.add_argument('--fine_tuning', action='store_true', help='파인튜닝 학습 시작')
    parser.add_argument('--target_evaluation', action='store_true', help='타겟 평가(K-Fold) 시작')
    parser.add_argument('--predict', type=str, help='예측용 오디오 파일 경로 입력')
    parser.add_argument('--predict_without_stt', type=str, help='예측용 텍스트 입력')
    args = parser.parse_args()

    # 설정값 로드
    with open(args.set_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    set_seed(seed=config['seed'])

    # 데이터 생성 단계
    if args.create_fine_tuning_data:
        os.makedirs(config['save_data_path'], exist_ok=True)
        fine_tuning_data = create_ai_hub_fine_tuning_data(config)

    if args.create_test_data:
        os.makedirs(config['save_data_path'], exist_ok=True)
        test_data = create_test_data(config)

    # 모델 준비
    if args.set_bert_model:
        if args.set_fine_tuning_weight:
            model, tokenizer = set_model(config['bert_model'], config['fine_tuning_weight'])
        else:
            model, tokenizer = set_model(config['bert_model'])

    # 학습 및 평가 로직
    if args.fine_tuning:
        if args.set_fine_tuning_weight:
            print("이미 학습된 가중치가 설정되어 있어 학습을 건너뜁니다.")
        else:
            if fine_tuning_data is not None:
                fine_tuning(model, tokenizer, config, fine_tuning_data)
            else:
                fine_tuning(model, tokenizer, config)

    if args.target_evaluation:
        if test_data is not None:
            target_evaluation(tokenizer, config, test_data)
        else:
            target_evaluation(tokenizer, config)

    if args.predict:
        result, logits = predict_result(args.predict, config)
        return result, logits

    if args.predict_without_stt:
        result, logits = predict_result_without_stt(args.predict_without_stt, config)
        return result, logits

if __name__ == '__main__':
    predict_result_val, logits = main()