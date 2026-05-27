# PREGO-VideoMAE
### Egocentric Procedural Assistance for Wearable Vision

## Overview
PREGO-VideoMAE is a wearable procedural assistance system designed to help users perform multi-step tasks using egocentric video understanding.

Inspired by wearable platforms such as Apple Vision Pro and smart glasses, the system observes first-person video, recognizes the user's current procedural step, tracks workflow progress, and provides corrective guidance when mistakes occur.

The goal of this project is to create a practical computer vision assistant for real-time procedural support.

Example use cases:
- Cooking assistance
- DIY assembly guidance
- Laboratory procedure support
- Rehabilitation exercise coaching

---

## Motivation
Procedural tasks require users to follow ordered sequences of actions correctly.

Mistakes such as:
- skipping required steps
- performing actions in the wrong order
- repeating unnecessary actions
- deviating from expected workflows

can reduce efficiency, lead to task failure, or create safety risks.

Existing procedural mistake detection methods often rely on complex symbolic reasoning pipelines, making them difficult to implement and deploy in lightweight wearable environments.

This project explores whether modern video understanding models can provide a simpler and more practical alternative.

---

## Project Goal
Build a wearable procedural assistant capable of:

- recognizing the user's current action
- tracking procedural progress
- detecting workflow deviations
- recommending corrective next actions

Example outputs:

Normal execution:

Current Step: Pour Water
Next Step: Add Coffee


---

# 한글 README

# PREGO-VideoMAE
### 웨어러블 비전을 위한 Egocentric Procedural Assistance System

## 프로젝트 개요
PREGO-VideoMAE는 사용자의 1인칭(egocentric) 영상을 이해하여 다단계 절차 작업을 보조하는 웨어러블 procedural assistance 시스템입니다.
Apple Vision Pro와 스마트 글래스 같은 웨어러블 플랫폼에서 영감을 받아, 사용자의 현재 행동을 인식하고 절차 진행 상태를 추적하며, 잘못된 수행이 감지될 경우 교정 가이드를 제공하는 것을 목표로 합니다.

본 프로젝트의 목표는 실시간 procedural support가 가능한 실용적인 컴퓨터 비전 어시스턴트를 구현하는 것입니다.

활용 예시:
- 요리 보조
- DIY 조립 가이드
- 실험 절차 지원
- 재활 운동 코칭

---

## 연구 동기
절차 기반 작업은 정해진 순서대로 행동을 수행해야 합니다.

다음과 같은 실수가 발생할 수 있습니다.

- 필수 단계 누락
- 잘못된 순서 수행
- 불필요한 행동 반복
- 절차 흐름 이탈

이러한 실수는 작업 실패, 비효율, 안전 문제로 이어질 수 있습니다.

기존 procedural mistake detection 연구들은 복잡한 symbolic reasoning 구조를 사용하는 경우가 많아 실제 웨어러블 환경에 적용하기 어렵습니다.

본 프로젝트는 현대적인 video understanding 모델을 활용하여 더 단순하고 실용적인 대안을 탐색합니다.

---

## 프로젝트 목표
다음 기능을 수행하는 wearable assistant를 구현합니다.

- 사용자의 현재 행동 인식
- 절차 진행 상태 추적
- workflow 이탈 감지
- 다음 올바른 행동 추천

예시 출력:

정상 수행:

Current Step: Pour Water
Next Step: Add Coffee