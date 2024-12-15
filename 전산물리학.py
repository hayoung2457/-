import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv("data000.csv", encoding='EUC-KR')

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

score = [col for col in data.columns if '비율' in col]
labels = [col.replace(' 비율', '') for col in score]

def predictions():
    while True:  # 터미널에서 사용자가 직접 입력할 수 있게 하는 프로그램
        department = input("\n단과대학을 입력하세요: ").strip()
        major = input("전공을 입력하세요: ").strip()
        
        dept_data = data[(data['단과대학'] == department) & (data['전공'] == major)]
        if dept_data.empty:
            print(f"\n{department} {major} 은/는 존재하지 않습니다. 다시 입력해주세요.")
            continue
        break

    grade_data = dept_data[['연도'] + score].dropna()  # 비어있는 셀을 삭제하기 위해 dropna() 코드 사용
    x = grade_data[['연도']]
    y = grade_data[score]

    N = len(x)
    n = int(N * (3/4))

    shuffle_x = x.sample(frac=1, random_state=42).reset_index(drop=True)  # 데이터프레임 형태의 데이터를 무작위 섞기
    shuffle_y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    x_train = shuffle_x[:n]
    y_train = shuffle_y[:n]

    x_test = shuffle_x[n:]
    y_test = shuffle_y[n:]

    predicted_2021_2022 = []
    y_test_data = []
    y_pred_data = []
    for col in score:
        grade_model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # 모델 생성
        grade_model.fit(x_train, y_train[col])

        y_pred = grade_model.predict(x_test)
        y_test_data.extend(y_test[col].tolist())  # mse와 r2 값 계산을 위해 데이터를 리스트 형태로 변환
        y_pred_data.extend(y_pred.tolist())
        
        predicted_2021_2022.append(grade_model)  # 모든 성적 항목에 대한 모델을 저장하기 위해 사용
    
    mse = mean_squared_error(y_test_data, y_pred_data)
    r2 = r2_score(y_test_data, y_pred_data)

    print(f"\n성적 테스트 데이터 예측 정확도:")
    print(f" - MSE: {mse:.4f}")
    print(f" - R²: {r2:.4f}")

    x_2023 = pd.DataFrame([[2023]], columns=['연도'])
    predicted_2023 = [model.predict(x_2023)[0] for model in predicted_2021_2022]  # 모델로 데이터 예측
    predicted_2023 = np.clip(predicted_2023, 0, None)  # 예측 값에 음수가 나와서 음수를 0으로 대체
    total_sum = sum(predicted_2023)  # 음수를 0으로 대체하면 100%가 되지 않으니 정규화 진행행
    predicted_2023 = [value / total_sum * 100 for value in predicted_2023]

    print(f"\n2023년 {department} {major}의 성적 분포 예측:")
    for col, value in zip(score, predicted_2023):
        print(f"{col.replace(' 비율', '')}: {value:.2f}%") 

    plt.bar(labels, predicted_2023, color='#F7CBCA')
    plt.xlabel('성적')
    plt.ylabel('예측 비율 (%)')
    plt.title(f"{department} {major} 2023년 성적 분포 예측")
    plt.legend([f"MSE: {mse:.4f}, R²: {r2:.4f}"], loc='upper right')
    plt.show()

    employment_data = dept_data[['연도', '취업률'] + score].dropna()
    x = employment_data[['연도'] + score]
    y = employment_data['취업률']

    N = len(x)
    n = int(N * (5/6))

    shuffle_x = x.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffle_y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    employment_x_train = shuffle_x[:n]
    employment_y_train = shuffle_y[:n]

    employment_x_test = shuffle_x[n:]
    employment_y_test = shuffle_y[n:]

    employment_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    employment_model.fit(employment_x_train, employment_y_train)

    employment_y_pred = employment_model.predict(employment_x_test)

    mse = mean_squared_error(employment_y_test, employment_y_pred)
    mae = mean_absolute_error(employment_y_test, employment_y_pred)

    print(f"\n취업률 테스트 데이터 예측 정확도(GradientBoostingRegressor):")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")

    x = employment_data['연도'].values  # polyfit을 계산하기 위해 numpy 배열로 변환
    y = employment_data['취업률'].values

    coefficients = np.polyfit(x, y, deg=1)
    trend_line = np.poly1d(coefficients)

    y_pred = trend_line(x)
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"\n취업률 테스트 데이터 예측 정확도(Polyfit):")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")

    scores_2023_actual = pd.DataFrame(
       [[2023] + dept_data[dept_data['연도'] == 2023][score].iloc[0].tolist()],  # 데이터를 리스트 형태로 변환환
       columns=['연도'] + score
    )
    employment_2023_actual = employment_model.predict(scores_2023_actual)[0]

    print(f"\n2023년 {department} {major}의 취업률 예측: {employment_2023_actual:.2f}%")

while True:
    predictions()
    while True:
        cont = input("\n다른 학과를 확인하시겠습니까? (예/아니요): ").strip().lower()  # 공백, 대소문자 입력값 조정
        if cont in ['예', '아니요']:
            break
        print("\n잘못된 입력입니다. '예' 또는 '아니요'를 입력해주세요.")
    if cont == '아니요':
        print("\n전산물리학 종강!\n")
        break