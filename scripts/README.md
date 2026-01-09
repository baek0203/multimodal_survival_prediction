# TCGA-OV & TCIA-OV 멀티모달 데이터 다운로드 스크립트

샘플링된 환자 데이터를 자동으로 다운로드하는 Python 스크립트 모음입니다.

## 📋 사전 준비

### 1. Python 패키지 설치
```bash
pip install tcia-utils requests pandas numpy tqdm
```

### 2. GDC Client 설치 (Linux/WSL)
```bash
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip
chmod +x gdc-client
# gdc-client를 현재 디렉토리 또는 PATH에 위치시키기
```

### 3. 디렉토리 구조 생성
```bash
mkdir -p data/{imaging,genomic,clinical}
```

---

## 🚀 실행 순서

### 1단계: 환자 샘플링
```bash
python scripts/01_sample_patients.py
```

**동작:**
- TCGA와 TCIA에서 전체 환자 목록 조회
- 두 데이터베이스 모두에 데이터가 있는 환자만 선택
- 30명 랜덤 샘플링 (코드에서 조정 가능)
- `data/sampled_patients.csv` 생성

**출력 파일:**
- `data/sampled_patients.csv` - 샘플링된 환자 ID 리스트
- `data/all_common_patients.csv` - 전체 공통 환자 리스트
- `data/sampling_summary.json` - 샘플링 요약 정보

---

### 2단계: TCGA 유전체 데이터 다운로드
```bash
python scripts/02_download_tcga.py
```

**동작:**
- 샘플링된 환자의 RNA-seq, Mutation, CNV 데이터 다운로드
- GDC API로 파일 목록 조회
- gdc-client로 자동 다운로드
- 임상 데이터도 함께 수집

**출력 파일:**
- `data/genomic/rnaseq/` - RNA-seq 데이터
- `data/genomic/mutation/` - Mutation 데이터 (MAF 파일)
- `data/genomic/cnv/` - Copy Number Variation 데이터
- `data/clinical/tcga_ov_sampled_clinical.csv` - 임상 데이터
- `data/gdc_manifest_*.txt` - 다운로드 매니페스트 파일들

**예상 용량:** 200-500 MB

---

### 3단계: TCIA 영상 데이터 다운로드
```bash
python scripts/03_download_tcia.py
```

**동작:**
- 샘플링된 환자의 의료 영상(CT/MRI) 다운로드
- tcia-utils 사용
- 환자별로 시리즈 정보 수집
- 다운로드 전 용량 추정 및 확인

**출력 파일:**
- `data/imaging/dicom/[환자ID]/` - DICOM 영상 파일들
- `data/imaging/metadata/tcia_ov_sampled_metadata.csv` - 영상 메타데이터
- `data/imaging/metadata/patient_series_summary.csv` - 환자별 시리즈 요약
- `data/imaging/download_summary.json` - 다운로드 요약

**예상 용량:** 5-10 GB

**참고:** 다운로드 시작 전 용량을 확인하고 승인을 요청합니다.

---

### 4단계: 데이터 검증
```bash
python scripts/04_validate_data.py
```

**동작:**
- 각 환자별 데이터 완전성 확인
- 멀티모달 데이터(영상 + RNA-seq + 임상) 매칭
- 디스크 사용량 계산
- 최종 요약 리포트 생성

**출력 파일:**
- `data/validation_results.csv` - 환자별 데이터 완전성
- `data/multimodal_patients.csv` - 멀티모달 데이터 완전한 환자 리스트
- `data/data_summary.json` - 최종 데이터 요약

---

## 📊 데이터 구조

다운로드 완료 후 디렉토리 구조:

```
data/
├── sampled_patients.csv              # 샘플링된 환자 ID
├── multimodal_patients.csv           # 멀티모달 완전한 환자 ID
├── validation_results.csv            # 검증 결과
├── data_summary.json                 # 최종 요약
│
├── imaging/
│   ├── dicom/
│   │   ├── TCGA-OV-XX-XXXX/         # 환자별 DICOM 파일
│   │   └── ...
│   └── metadata/
│       ├── tcia_ov_sampled_metadata.csv
│       └── patient_series_summary.csv
│
├── genomic/
│   ├── rnaseq/                       # RNA-seq 데이터
│   ├── mutation/                     # Mutation 데이터
│   └── cnv/                          # CNV 데이터
│
└── clinical/
    └── tcga_ov_sampled_clinical.csv  # 임상 데이터
```

---

## ⚙️ 커스터마이징

### 샘플 크기 변경
`01_sample_patients.py`에서:
```python
SAMPLE_SIZE = 30  # 원하는 숫자로 변경 (예: 50)
```

### 특정 데이터만 다운로드
개별 스크립트만 실행:
- 유전체 데이터만: `02_download_tcga.py`
- 영상 데이터만: `03_download_tcia.py`

### 재현성 보장
랜덤 시드 고정 (이미 적용됨):
```python
random.seed(42)
```

---

## 🔧 문제 해결

### tcia-utils 설치 오류
```bash
pip install --upgrade pip
pip install tcia-utils
```

### gdc-client를 찾을 수 없음
스크립트가 다음 경로에서 gdc-client를 찾습니다:
- `./gdc-client`
- `gdc-client` (PATH)
- `../gdc-client`

수동으로 실행:
```bash
./gdc-client download -m data/gdc_manifest_*.txt -d data/genomic/
```

### TCIA 연결 실패
- 네트워크 연결 확인
- VPN 사용 시 비활성화 후 재시도
- TCIA 서버 상태 확인: https://www.cancerimagingarchive.net/

### 다운로드 중단 시
각 스크립트는 재실행 가능합니다:
- gdc-client는 이미 다운로드된 파일을 건너뜀
- tcia-utils도 기존 파일 확인

---

## 📈 예상 실행 시간

| 단계 | 예상 시간 | 비고 |
|------|----------|------|
| 1. 샘플링 | 1-2분 | API 호출만 |
| 2. TCGA 다운로드 | 10-30분 | 네트워크 속도에 따라 |
| 3. TCIA 다운로드 | 1-3시간 | 영상 크기에 따라 |
| 4. 검증 | 1-5분 | 로컬 파일 확인만 |

**총 예상 시간:** 1.5 - 4시간

---

## 💾 예상 디스크 사용량

30명 환자 기준:
- 영상 데이터: 5-10 GB
- RNA-seq: 100-200 MB
- Mutation: 50-100 MB
- CNV: 50-100 MB
- 임상: <1 MB

**총합: 약 6-11 GB**

---

## 📝 다음 단계

데이터 다운로드 완료 후:

1. **데이터 전처리**
   - DICOM → NIfTI 변환
   - 영상 정규화
   - RNA-seq 정규화

2. **멀티모달 통합**
   - `data/multimodal_patients.csv` 활용
   - 환자별 데이터 페어링
   - Feature 추출

3. **모델 개발**
   - 멀티모달 융합 전략
   - 딥러닝 모델 구현

---

## 📚 참고 자료

- [TCIA Portal](https://www.cancerimagingarchive.net/)
- [GDC Portal](https://portal.gdc.cancer.gov/)
- [tcia-utils Documentation](https://github.com/kirbyju/tcia_utils)
- [GDC API Documentation](https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/)