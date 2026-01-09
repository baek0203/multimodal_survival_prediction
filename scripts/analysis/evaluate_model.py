"""
모델 평가 스크립트
- C-index (Concordance Index) 계산
- Kaplan-Meier 생존 곡선
- Risk group 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("멀티모달 모델 평가")
print("=" * 60)

# ============================================================
# 1. 예측 결과 로드
# ============================================================
print(f"\n{'=' * 60}")
print("예측 결과 로드")
print(f"{'=' * 60}")

df = pd.read_csv('results/test_predictions.csv')
print(f"\n테스트 환자 수: {len(df)}명")
print(f"\nColumns: {list(df.columns)}")
print(f"\n처음 5개 샘플:")
print(df.head())

# ============================================================
# 2. C-index 계산
# ============================================================
print(f"\n{'=' * 60}")
print("C-index 계산")
print(f"{'=' * 60}")

c_index = concordance_index(
    df['survival_time'],
    -df['risk_score'],  # Negative because higher risk = shorter survival
    df['event']
)

print(f"\nC-index: {c_index:.4f}")
print(f"  (0.5 = random, 1.0 = perfect)")

# ============================================================
# 3. Risk Group 분류
# ============================================================
print(f"\n{'=' * 60}")
print("Risk Group 분류")
print(f"{'=' * 60}")

# Median risk score를 기준으로 High/Low risk 분류
median_risk = df['risk_score'].median()
df['risk_group'] = df['risk_score'].apply(
    lambda x: 'High Risk' if x > median_risk else 'Low Risk'
)

print(f"\nMedian risk score: {median_risk:.4f}")
print(f"\nRisk group 분포:")
print(df['risk_group'].value_counts())

# ============================================================
# 4. Kaplan-Meier 생존 곡선
# ============================================================
print(f"\n{'=' * 60}")
print("Kaplan-Meier 생존 곡선 생성")
print(f"{'=' * 60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KM 객체 생성
kmf = KaplanMeierFitter()

# (1) Risk group별 생존 곡선
ax = axes[0]

for group in ['Low Risk', 'High Risk']:
    mask = df['risk_group'] == group
    kmf.fit(
        df[mask]['survival_time'],
        df[mask]['event'],
        label=group
    )
    kmf.plot_survival_function(ax=ax)

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# (2) Event 유무별 생존 곡선
ax = axes[1]

for event_label, event_val in [('Censored', 0), ('Death', 1)]:
    mask = df['event'] == event_val
    if mask.sum() > 0:
        kmf.fit(
            df[mask]['survival_time'],
            df[mask]['event'],
            label=event_label
        )
        kmf.plot_survival_function(ax=ax)

ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Kaplan-Meier Survival Curves by Event Type', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/kaplan_meier_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ 저장: results/kaplan_meier_curves.png")

# ============================================================
# 5. Risk Score 분포 시각화
# ============================================================
print(f"\n{'=' * 60}")
print("Risk Score 분포 시각화")
print(f"{'=' * 60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1) Histogram
ax = axes[0]
ax.hist(df[df['risk_group'] == 'Low Risk']['risk_score'],
        bins=15, alpha=0.6, label='Low Risk', color='blue')
ax.hist(df[df['risk_group'] == 'High Risk']['risk_score'],
        bins=15, alpha=0.6, label='High Risk', color='red')
ax.axvline(median_risk, color='black', linestyle='--', label='Median')
ax.set_xlabel('Risk Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Risk Score Distribution', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# (2) Box plot
ax = axes[1]
df.boxplot(column='risk_score', by='risk_group', ax=ax)
ax.set_xlabel('Risk Group', fontsize=12)
ax.set_ylabel('Risk Score', fontsize=12)
ax.set_title('Risk Score by Group', fontsize=14)
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('results/risk_score_distribution.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/risk_score_distribution.png")

# ============================================================
# 6. 생존 시간 vs Risk Score
# ============================================================
print(f"\n{'=' * 60}")
print("생존 시간 vs Risk Score 시각화")
print(f"{'=' * 60}")

fig, ax = plt.subplots(figsize=(10, 6))

# Event별 색상 구분
colors = df['event'].map({0: 'blue', 1: 'red'})
labels = df['event'].map({0: 'Censored', 1: 'Death'})

for event_val, color, label in [(0, 'blue', 'Censored'), (1, 'red', 'Death')]:
    mask = df['event'] == event_val
    ax.scatter(
        df[mask]['risk_score'],
        df[mask]['survival_time'],
        c=color,
        alpha=0.6,
        s=100,
        label=label,
        edgecolors='black',
        linewidths=0.5
    )

ax.set_xlabel('Risk Score', fontsize=12)
ax.set_ylabel('Survival Time (days)', fontsize=12)
ax.set_title('Survival Time vs Risk Score', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/survival_vs_risk.png', dpi=150, bbox_inches='tight')
print("✓ 저장: results/survival_vs_risk.png")

# ============================================================
# 7. 통계 요약
# ============================================================
print(f"\n{'=' * 60}")
print("통계 요약")
print(f"{'=' * 60}")

summary = {
    'test_patients': len(df),
    'deaths': int(df['event'].sum()),
    'censored': int((1 - df['event']).sum()),
    'c_index': float(c_index),
    'median_survival_time': float(df['survival_time'].median()),
    'median_risk_score': float(median_risk),
    'risk_groups': {
        'low_risk': int((df['risk_group'] == 'Low Risk').sum()),
        'high_risk': int((df['risk_group'] == 'High Risk').sum())
    }
}

print(f"\n테스트 환자: {summary['test_patients']}명")
print(f"  Death: {summary['deaths']}명")
print(f"  Censored: {summary['censored']}명")
print(f"\nC-index: {summary['c_index']:.4f}")
print(f"Median survival time: {summary['median_survival_time']:.1f} days")
print(f"Median risk score: {summary['median_risk_score']:.4f}")
print(f"\nRisk groups:")
print(f"  Low risk: {summary['risk_groups']['low_risk']}명")
print(f"  High risk: {summary['risk_groups']['high_risk']}명")

# 저장
import json
with open('results/evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ 저장: results/evaluation_summary.json")

# ============================================================
# 8. Risk group별 생존 통계
# ============================================================
print(f"\n{'=' * 60}")
print("Risk Group별 생존 통계")
print(f"{'=' * 60}")

for group in ['Low Risk', 'High Risk']:
    mask = df['risk_group'] == group
    group_data = df[mask]

    print(f"\n{group}:")
    print(f"  환자 수: {len(group_data)}명")
    print(f"  Death: {int(group_data['event'].sum())}명")
    print(f"  Censored: {int((1 - group_data['event']).sum())}명")
    print(f"  평균 생존 시간: {group_data['survival_time'].mean():.1f} days")
    print(f"  중앙 생존 시간: {group_data['survival_time'].median():.1f} days")
    print(f"  평균 risk score: {group_data['risk_score'].mean():.4f}")

print(f"\n{'=' * 60}")
print("평가 완료!")
print(f"{'=' * 60}")
print(f"\n저장된 파일:")
print(f"  - results/kaplan_meier_curves.png")
print(f"  - results/risk_score_distribution.png")
print(f"  - results/survival_vs_risk.png")
print(f"  - results/evaluation_summary.json")
