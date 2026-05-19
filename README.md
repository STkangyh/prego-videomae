# PREGO + VideoMAE v2: Improved Action Recognition for Procedural Mistake Detection

## Overview
PREGO는 절차적 1인칭 영상에서 실수를 온라인으로 감지하는 모델입니다. 현재 인식된 행동과 LLM 기반 심볼릭 추론 모듈이 예측한 다음 행동을 비교해 실수를 탐지합니다.
이 레포에서 비디오 backbone을 교체하면서 성능 병목이 얼마나 해소되는지를 정량적으로 분석하는 것을 목표로 합니다.

- 원본 논문: PREGO (CVPR 2024)
- 원본 레포: aleflabo/PREGO

## Motivation
원본 논문의 Oracle 실험(정답 행동 라벨을 직접 입력하는 조건)에서 PREGO-LLAMA는 일반 설정 대비 F1-score 기준 최대 11% 향상이 가능함을 확인했습니다. 이는 LLM 추론 모듈이 아닌, 비디오 기반 행동 인식 모듈이 현재 성능의 주요 병목임을 암시합니다. 
따라서 원본 논문에서 사용하는 비디오 backbone 외에 다른 모델을 이용해서 이 병목을 해소해보고자 합니다.
또한, 추가 데이터셋을 만들어서 실험 데이터로 주로 사용하고 있는 Assembly101, Epic-tent 외에도 새로운 클래스(한식 요리과정 등)을 학습시켜서 스마트 글래스 같은 웨어러블 기기에 추가할만한 기능을 만들어보고 싶습니다.
