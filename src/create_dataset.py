import pandas as pd
import os
import whisper
from tqdm import tqdm


def load_stt_model(config):
    return whisper.load_model(config['stt_whisper_model'])

def create_ai_hub_fine_tuning_data(config):
    """AI HUB 데이터를 전처리하여 학습용 TSV 파일을 생성합니다."""

    # 데이터 경로 설정 및 불러오기
    subject_path = config['fine_tuning_data_path'] + 'voice/'
    label_path = config['fine_tuning_data_path'] + 'label/'
    subject_list = os.listdir(subject_path)
    label_list = os.listdir(label_path)
    save_path = config['save_data_path'] + 'fine_tuning_data.tsv'
    result = []
    for label in label_list:
        print(f'{label} 데이터 전처리 중...')
        temp_df = pd.read_csv(label_path + label, encoding='cp949')
        temp_df['상황'] = temp_df['상황'].replace({'anger':'angry', 'fear':'fearful', 'sadness':'sad', 'happiness':'happy'})
        for i in range(len(temp_df)):
            subject_id = temp_df.iloc[i]['wav_id'] + '.wav'
            gt_stt = temp_df.iloc[i]['발화문']
            temp_label = temp_df.iloc[i]['상황']
            if subject_id in subject_list:
                result.append({'labels': config['emotion_map'][temp_label], 'text': gt_stt})

    data_info_df = pd.DataFrame(result)
    data_info_df.to_csv(save_path, sep='\t', index=False)
    print(f'전처리된 데이터셋이 {save_path}에 저장되었습니다.')
    return data_info_df

def create_test_data(config):
    """테스트용 음성 파일을 Whisper STT를 통해 텍스트로 변환합니다."""

    stt_model = load_stt_model(config)
    sample_list = os.listdir(config['test_data_path'])
    save_path = config['save_data_path'] + 'test_data.tsv'

    result = []

    for sample in tqdm(sample_list):
        parts = sample.replace('.wav', '').split('_')
        subject_id = parts[0]
        emotion = parts[1]
        temp_path = config['test_data_path'] + sample
        if emotion in config['emotion_map']:
            STT = stt_model.transcribe(temp_path, language='ko')["text"].strip()
            result.append({'labels': config['emotion_map'][emotion], 'text': STT, 'subject': subject_id})

    data_info_df = pd.DataFrame(result)
    data_info_df.to_csv(save_path, sep='\t', index=False)
    print(f'테스트 데이터셋이 {save_path}에 저장되었습니다.')
    return data_info_df