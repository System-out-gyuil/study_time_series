# 🚩 Time series financial market project

## 삼성전자 주식 분석 및 예측

- 2024년 6월 17일 기준 작성
- 단위는 원화 단위로 진행하였습니다.

### 목차

### 1. 데이터 불러오기

- yfinance를 통해 삼성전자의 데이터를 불러와서 사용하였고, GLD(금ETF)와 비교를 위해 함께 다운로드 하였습니다.

- 2010년 10월 1일부터 2024년 6월 16일까지의 데이터를 가져왔습니다.

  <details>
      <summary>데이터 불러오기 코드보기</summary>
      
        import yfinance as yf
        import pandas as pd

        # 종목 코드와 기간 설정
        symbols = ['005930.KS', 'GLD']
        start_date = '2010-01-01'
        end_date = '2024-06-16'

        # 삼성전자와 GLD 데이터 다운로드
        data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

        # USD/KRW 환율 데이터 다운로드 (FX 코드 사용)
        usd_krw = yf.download('KRW=X', start=start_date, end=end_date)['Adj Close']

        # GLD 데이터를 원화로 변환
        data['GLD_KRW'] = data['GLD'] * usd_krw

        data = data.round(4)
        w_df = data[['005930.KS', 'GLD_KRW']]
        w_df

  </details>

<br/>

### 2. 삼성전자와 금ETF의 주가 확인

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/add59bfc-b8db-4e32-b682-d1becd4fb384">

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/58041493-199c-425e-86cd-6b84c5603b94">

- 삼성전자의 주가를 확인해보면 2012년 이후로 스마트폰의 시대가 시작되면서  
  주가가 조금씩 오르기 시작하다 갤럭시s 시리즈와 노트 시리지의 큰 인기로  
  2016년에서 2018년에 크게 상승하는 모습을 보입니다.  
  또한 코로나바이러스로 인해 2020년도 초반에 큰 폭으로 감소하였다가 플립, 폴드 등 폴더블폰의 인기에 힘입어
  2021년정도에 큰 폭으로 주가가 상승하는 모습을 보입니다.

- 금ETF의 경우 한번에 큰 폭으로 변화하는 모습은 잘 보이지 않으며 2014년부터 2019년도에 아주 잠잠하다가 2021년도 이후로 계속 증가하는 추세를 보입니다.

- 삼성전자와 금 ETF의 주가를 비교해보았을 때 미묘하게 비슷한 양상을 띄는 부분이 있는 듯 합니다.

<br/>

### 3. 차분 후 주가 비교

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/091ca9bf-e981-4929-a515-99002ae23af6">

- 차분 후 주가를 확인해 보았을 때 삼성전자의 경우 2019년도 까지는 분산이 일정한 편이었으나
  그 이후 점점 분산이 크게 나타나는 것을 확인할 수 있었습니다.

- GLD의 경우 2014년도부터 2019년까지의 분산이 일정하며 2013년 하반기에 큰 분산이 나타났으며,  
  크게 눈에 띄는 구간은 없었습니다.

<br/>

### 4. 변동성 확인

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/dd3418cf-3aa1-4c60-a3e9-fec03b1b3f4c" width="500px">

- 확실히 안정적인 GLD보다 삼성전자의 변동성이 크게 나타난 모습입니다.

<br/>

### 5. 수익률 계산

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/37a11219-8f33-47ba-af8f-f16cb78af96d" width="250px">

  <details>
    <summary>수익률 계산 코드 보기</summary>
  
        import numpy as np

        # 수익률 계산
        rate_w_df = np.log(w_df / w_df.shift(1))
        rate_w_df

  </details>

  <br/>

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/83b16ecb-ebb1-4d28-a1aa-dcc38a9f9c70">

- 수익률의 변동성을 확인해보았을 때 또한 위의 분산과 같은 모습을 보였습니다.

### 6. 연율화(영업일 252일으로 계산)

> 삼성전자 : 0.1365  
> GLD : 0.0635

- 연율화를 통해 수익을 비교해보았을 때, 삼성전자의 연간 수익율이 그렇게 높지 않다는 것을 확인할 수 있습니다.
- GLD또한 금 ETF답게 큰 수익율이 나오지 않는 종목임을 미뤄보아 삼성전자의 수익율이 높지 않다는 것을 알 수 있습니다.

<br/>

### 7. VIF스코어

- 위에서 삼성전자와 GLD의 주가를 비교해보았을 때 살짝 비슷한 느낌이 있어서 상관관계 확인을 위해 VIF스코어를 확인해 보았습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/611b3fa2-8c6c-4ab9-918d-e34f39046644">

  <details>
    <summary>VIF스코어 확인 코드 보기</summary>

        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        def get_vif(features):
            vif = pd.DataFrame()
            vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
            vif['feature'] = features.columns
            return vif

        rate_w_df = rate_w_df.dropna()
        get_vif(rate_w_df)

  </details>

  <br/>

  - vif스코어를 통해 확인해 보았을 경우 GLD와 삼성전자는 상관관계가 거의 없다는 것을 확인할 수 있었습니다.

<br/>

### 8. 로그 변환

- 데이터 분포 조정(원본 - 상단, 로그변환 - 하단)

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/7a2b356b-85db-415e-94a0-51191aa7787b">

  - 데이터 비교의 편의성과 신뢰성 향상을 위해 로그 변환을 통해 정규분포 형태로 조정해주었습니다.

<br/>

### 9. 일간 수익률과 월간 수익률

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/f2507613-f526-4f5c-8a22-d1f1d73b9ca0">

- 금 ETF의 경우 수익률의 변화가 거의 보이지 않는 모습이며
- 삼성전자의 경우 상승하는 추세를 보이고 있으며, 주가와 마찬가지로 21년도정도에 큰 폭으로 상승했던 모습을 보입니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/555139b8-1401-44d6-b478-ce33d7e9c1ae">

- 일간 수익률보다 조금 더 평탄화되었으며, 가독성이 좋아졌습니다.

<br/>

### 10. 이동 평균

- 최근 1년간의 데이터를 20일 기준으로 최소, 최대, 평균을 구해준 후 시각화를 통해 확인하였습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/a880878b-c50c-4d07-8f3e-0e58d1f8afe4">

  <details>
    <summary>이동평균 코드 보기</summary>

        window = 20

        # 20일간의 데이터를 최소, 최대, 평균을 구해줌
        s_df['min'] = s_df['Samsung'].rolling(window=window).min()
        s_df['mean'] = s_df['Samsung'].rolling(window=window).mean()
        s_df['max'] = s_df['Samsung'].rolling(window=window).max()

        s_df.dropna()

        import matplotlib.pyplot as plt

        # 최소, 최대, 평균값 시각화
        ax = s_df[['min', 'mean', 'max']].iloc[-252:].plot(figsize=(12, 6), style=['g--', 'r--', 'g--'], lw=0.8)
        s_df['Samsung'].iloc[-252:].plot(ax=ax)
        plt.title("Samsung 20-Day Moving Average Price Movement")
        plt.show()

  </details>

  <br/>

  - 이동평균을 보았을 때 2024년 2월과 4월에 최대값과 최소값이 크게 차이나는 것을 확인할 수 있었으며  
    조금씩 상승하는 추세를 보입니다.

- 다음은 삼성전자의 주가 기술분석입니다.
- 단기 이동평균인 SMA1(주황 선)이 장기 이동평균선SMA2(초록 선)을 아래에서 위로 교차할 때 매수를 권장하고  
  반대로 단기 이동평균이 장기이동평균을 위애서 아래로 교차할 때 매도를 권장합니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/a867f761-5266-4f9d-8108-01f3a87e014d">

  <details>
    <summary>주가 기술분석 코드 보기</summary>

        # 삼성전자 주가 기술 분석
        # 골든 크로스, 데드 크로스
        s_df.dropna(inplace=True)

        s_df['positions'] = np.where(s_df['SMA1'] > s_df['SMA2'], 1, -1)  # 1: buy , -1: sell /

        ax = s_df[['Samsung', 'SMA1', 'SMA2', 'positions']].plot(figsize=(15, 8), secondary_y='positions')
        ax.get_legend().set_bbox_to_anchor((-0.05, 1))

        plt.title("Samsung Trading Window based on Technical Analysis")
        plt.show()

  </details>

  <br/>

  - 골든크로스와 데드크로스가 자주 발생한 모습입니다.  
    이는 삼성전자가 대기업이기 때문에 크고 작은 일에 변동이 많이 나타나기 때문으로 추정됩니다.

### 11. Auto arima

- auto arima로 예측하기 전에 오래된 데이터는 제외하고 학습시키기 위해 2022년 이후의 데이터만 가지고 진행하였습니다.

  - 학습 데이터와 테스트 데이터 분리 및 시각화

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/962faec2-d563-43d9-94c4-6a647560f48d" width="500px">

  <details>
    <summary>데이터 분리 및 시각화 코드 보기</summary>

        # 데이터의 80퍼센트를 train, 나머지를 test데이터로 분리
        y_train = s_df.Samsung[:int(0.8 * len(s_df))]
        y_test = s_df.Samsung[int(0.8 * len(s_df)):]

        y_train.plot()
        y_test.plot()

  </details>

  <br/>

- 차분 횟수 구하기

- ndiffs를 통해 최적 차분횟수를 구해주었습니다.

  <details>
    <summary>ndiffs 코드 보기</summary>

        from pmdarima.arima import ndiffs
        # KPSS(Kwaiatkowski-Phillips-Schmidt-Shin)
        # 차분을 진행하는 것이 필요할 지 결정하기 위해 사용하는 한 가지 검정 방법
        # 영가설(귀무가설)을 "데이터에 정상성이 나타난다."로 설정한 뒤
        # 영가설이 거짓이라는 증거를 찾는 알고리즘이다.

        # alpha: 유의수준(기각 영가설 설정, 주로 0.05로 설정)
        # test: 단위근 검정 방법 설정
        # max_d: 최대차분 횟수 설정
        kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
        pp_diffs = ndiffs(y_train, alpha=0.05, test='pp', max_d=6)

        n_diffs = max(kpss_diffs, adf_diffs, pp_diffs)

        # 최적 차분 횟수
        print(f'd = {n_diffs}')

  </details>

  <br/>

  - 위 코드의 결과에 따라 차분 횟수는 2로 설정 후 진행하였습니다.

<br/>

- 모델 생성

  - auto arima를 사용하여 모델을 생성해주었습니다.

  <details>
    <summary>auto arima 모델 생성 코드 보기</summary>

        import pmdarima as pm

        model = pm.auto_arima(y=y_train,
                              d=2,
                              start_p=0, max_p=10,
                              start_q=0, max_q=10,
                              m=1, seasonal=False,
                              stepwise=True,
                              trace=True)

  </details>

  <br/>

- 생성된 모델을 train데이터로 학습 후 summary 결과

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/73384ecd-e513-4e37-b034-96ba02f7d4bc" width="600px">

  - Prob(Q), 융-박스 검정 통계량이 0.00으로 데이터가 서로 독립적이지 않고, 동일하지 않은 분포를 가졌다고 판단이 됩니다.
  - Prob(H), 이분산성 검정 통계량이 0.08으로 잔차의 분산이 일정하다고 판단됩니다.
  - Prob(JB), 자크-베라 검정 통계량이 0.00으로 일정한 평균과 분산을 따르지 않는다고 판단됩니다.
  - Skew가 -0.01로 거의 0에 근접한 값을 보아 분포가 거의 대칭적이라고 판단되며,
  - Kurtosis는 4.83으로 첨도가 높아 분포의 꼬리가 두껍게 나타날것이라 판단됩니다.

<br/>

- 모델의 plot_diagnostics를 통해 시각화로 확인해보았습니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/370978bd-5ba5-4de1-b939-3c818d3db3c4">

  - 분산이 조금씩 튀는값이 있지만 어느정도는 일정합니다.
  - 편향도 나쁘지않지만 양쪽 끝에 조금 튀는값을 확인하였습니다.
  - 분포는 정규분포 형태로 잘 나타난것을 확인하였고, 값이 조금은 일정한 범위내에 있다고 판단하였습니다.

<br/>

- 해당 모델로 predict한 결과

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/740a0eb8-f68a-443a-929d-d5ed1c4234ea">

  - 예측이 정답과 거의 유사한 모습을 확인할 수 있었습니다.

  - 오차가 약 1.5%정도로 잘 맞췄다고 판단이 됩니다.

### 12. prophet을 통한 미래 예측

- Prophet을 이용해 모델을 학습하고 make_future_dataframe를 통해 미래값을 예측합니다.

  <details>
    <summary>Prophet 모델 생성 및 예측 코드 보기</summary>

        from prophet import Prophet

        model = Prophet().fit(pre_s_df)
        future = model.make_future_dataframe(periods=365)

        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

  </details>

  <br/>

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/0db0d773-7e12-4d79-8cb8-388a9a81fc87">

  - 정확하게 맞춘다고 할 순 없지만 어느정도 추세는 맞는 모습이며, 계속 오를것이라 예측하였다.

- 신뢰구간을 통한 결과 시각화

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/19f01aaa-eb8d-4193-a02a-f5e5b7c9b0ae">

  - 신뢰구간을 벗어나는 값이(검은 점) 꽤나 있지만 대부분은 신뢰구간에 잘 들어가있는 모습을 보입니다.

- 추세와 요일, 월간 그래프

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/486cd408-ce71-432f-a2cd-ab84729f19c2">

  - 22년 7월에 떨어졌다가 23년도 이후로는 꾸준히 상승하는 추세를 보였으며 미래에도 꾸준히 상승하는 추세를 보입니다.
  - 월요일에 수치가 떨어지고 금요일에 수치가 상승하는 모습을 보입니다.
  - 4월에 수치가 최고점을 찍고 10월에 가장 낮은 수치를 보입니다.
  - 이는 4월에 개학 혹은 개강을 하며 구매율이 크게 올랐다가  
    겨울이벤트 등이 오기 전에 지속적으로 수치가 감소하다가 10월달에 저점을 보이는 것으로 추정하였습니다.

### 13. Prophet 미세조정

- 위의 결과에서 신뢰구간을 벗어나는 값이 많고 예측을 조금 더 정확하게 해보기 위해 미세조정

- Prophet의 diagnostics와 itertools를 이용하여 미세조정을 진행하였습니다.

  <details>
    <summary>Prophet 미세조정 코드 보기</summary>

        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
        import itertools

        # changepoint_prior_scale: trend의 변화하는 크기를 반영하는 정도이다, 0.05가 default
        # seasonality_prior_scale: 계절성을 반영하는 단위이다.
        # seasonality_mode: 계절성으로 나타나는 효과를 더해 나갈지, 곱해 나갈지 정한다.
        search_space = {
            'changepoint_prior_scale': [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            'seasonality_prior_scale': [0.05, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }

        # itertools.product(): 각 요소들의 모든 경우의 수 조합으로 생성
        param_combinded = [dict(zip(search_space.keys(), v)) for v in itertools.product(*search_space.values())]

        train_len = int(len(pre_s_df) * 0.8)
        test_len = int(len(pre_s_df) * 0.2)

        train_size = f'{train_len} days'
        test_size = f'{test_len} days'
        train_df = pre_s_df.iloc[: train_len]
        test_df = pre_s_df.iloc[train_len: ]

        mapes = []
        for param in param_combinded:
            model = Prophet(**param)
            model.fit(train_df)

            # 'threads' 옵션은 메모리 사용량은 낮지만 CPU 바운드 작업에는 효과적이지 않을 수 있다.
            # 'dask' 옵션은 대규모의 데이터를 처리하는 데 효과적이다.
            # 'processes' 옵션은 각각의 작업을 별도의 프로세스로 실행하기 때문에 CPU 바운드 작업에 효과적이지만,
            # 메모리 사용량이 높을 수 있다.
            cv_df = cross_validation(model, initial=train_size, period='20 days', horizon=test_size, parallel='processes')
            df_p = performance_metrics(cv_df, rolling_window=1)
            mapes.append(df_p['mape'].values[0])

        tuning_result = pd.DataFrame(param_combinded)
        tuning_result['mape'] = mapes

  </details>

  <br/>

- 위에서 담아놓은 각 파라미터들마다의 로스값을 확인하여 가장 로스값이 적은 파라미터를 선택하여 모델에 사용합니다.

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/cab7e7f7-7561-4c22-893f-dd83a2f1bae2" width="500px">

  - changepoint_prior_scale: 0.05, seasonality_prior_scale: 1.00, seasonality_mode: additive  
    위의 파라미터 조합이 mape값 0.084로 가장 낮았기에 해당 파라미터로 진행

- 모델 학습

  - 최적의 파라미터를 넣고 훈련과 미래 값 예측은 동일하게 진행하였습니다.

  <details>
    <summary>Prophet 미세조정 후 학습 및 예측 코드 보기</summary>

        model = Prophet(changepoint_prior_scale=0.05,
                seasonality_prior_scale=1,
                seasonality_mode='additive')

        model.fit(pre_s_df)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

  </details>

  <br/>

- 학습 후 예측 결과

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/1b19ceed-9046-4737-8670-f5b2d9b3cccf">

  - additive임에도 모델은 조금 계절성이 있다고 판단하였는지 예측 결과가 조금 구불구불하게 나타난 모습입니다.

- 신뢰구간과 함께 확인

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/822e8afa-7202-4ad7-bd83-e3d96968c193">
  - 미세조정 후 신뢰구간

  <br/>

  <img src="https://github.com/System-out-gyuil/study_data_analysis/assets/120631088/19f01aaa-eb8d-4193-a02a-f5e5b7c9b0ae">
  - 미세조정 전 신뢰구간

  <br/>

  - 미세 조정 이후에도 이전과의 차이가 보이지 않는것을 확인하였습니다.

### 13. 느낀점

-
