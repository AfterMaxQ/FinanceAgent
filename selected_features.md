# VIF特征筛选报告

- 阈值: 5.0
- 用于计算的样本行数: 100255
- 初始迭代次数: 16

## 保留的特征
- Volume
- GSPC_Close
- GSPC_LogReturn
- LogReturn
- MACDh_12_26_9
- MACDs_12_26_9
- Intraday_Range
- Trend_Strength
- Candle_Body
- Month_Sin
- Sentiment_Score
- BBB_20_2.0_2.0
- BBP_20_2.0_2.0
- ATR_14
- OBV
- Beta_60
- Alpha_60
- Sharpe_60

## 迭代VIF明细（按最大VIF排序）
### 迭代 1
- 最大VIF特征: SMA_20
- 最大VIF值: inf

| 特征 | VIF |
| --- | ---: |
| SMA_20 | inf |
| BBU_20_2.0_2.0 | inf |
| BBL_20_2.0_2.0 | inf |
| BBM_20_2.0_2.0 | inf |
| EMA_12 | 723985.4097 |
| EMA_26 | 518852.6945 |
| SMA_5 | 98609.4782 |
| High | 75998.8877 |
| Low | 60064.6991 |
| Close | 56529.4048 |
| Open | 55284.7608 |
| Adj Close | 543.8970 |
| MACDs_12_26_9 | 203.5961 |
| MACD_12_26_9 | 193.3493 |
| MACDh_12_26_9 | 28.2309 |
| ATR_14 | 20.5501 |
| OBV | 16.8945 |
| RSI_14 | 11.0937 |
| BBP_20_2.0_2.0 | 7.3385 |
| Volume | 7.2780 |
| Volatility_20 | 5.8425 |
| BBB_20_2.0_2.0 | 4.6050 |
| LogReturn | 3.6397 |
| Sharpe_60 | 3.5318 |
| Trend_Strength | 3.5208 |
| Candle_Body | 3.2767 |
| Intraday_Range | 3.0139 |
| Alpha_60 | 2.3375 |
| GSPC_LogReturn | 1.3395 |
| Beta_60 | 1.1967 |
| GSPC_Close | 1.1848 |
| Month_Sin | 1.0181 |
| Sentiment_Score | 1.0027 |

### 迭代 2
- 最大VIF特征: BBL_20_2.0_2.0
- 最大VIF值: inf

| 特征 | VIF |
| --- | ---: |
| BBU_20_2.0_2.0 | inf |
| BBL_20_2.0_2.0 | inf |
| BBM_20_2.0_2.0 | inf |
| EMA_12 | 723985.4097 |
| EMA_26 | 518852.6945 |
| SMA_5 | 98609.4782 |
| High | 75998.8877 |
| Low | 60064.6991 |
| Close | 56529.4048 |
| Open | 55284.7608 |
| Adj Close | 543.8970 |
| MACDs_12_26_9 | 203.5961 |
| MACD_12_26_9 | 193.3493 |
| MACDh_12_26_9 | 28.2309 |
| ATR_14 | 20.5501 |
| OBV | 16.8945 |
| RSI_14 | 11.0937 |
| BBP_20_2.0_2.0 | 7.3385 |
| Volume | 7.2780 |
| Volatility_20 | 5.8425 |
| BBB_20_2.0_2.0 | 4.6050 |
| LogReturn | 3.6397 |
| Sharpe_60 | 3.5318 |
| Trend_Strength | 3.5208 |
| Candle_Body | 3.2767 |
| Intraday_Range | 3.0139 |
| Alpha_60 | 2.3375 |
| GSPC_LogReturn | 1.3395 |
| Beta_60 | 1.1967 |
| GSPC_Close | 1.1848 |
| Month_Sin | 1.0181 |
| Sentiment_Score | 1.0027 |

### 迭代 3
- 最大VIF特征: EMA_12
- 最大VIF值: 723985.4097

| 特征 | VIF |
| --- | ---: |
| EMA_12 | 723985.4097 |
| EMA_26 | 518852.6945 |
| BBM_20_2.0_2.0 | 128275.1904 |
| SMA_5 | 98609.4782 |
| High | 75998.8877 |
| Low | 60064.6991 |
| Close | 56529.4048 |
| Open | 55284.7608 |
| BBU_20_2.0_2.0 | 5526.1627 |
| Adj Close | 543.8970 |
| MACDs_12_26_9 | 203.5961 |
| MACD_12_26_9 | 193.3493 |
| MACDh_12_26_9 | 28.2309 |
| ATR_14 | 20.5501 |
| OBV | 16.8945 |
| RSI_14 | 11.0937 |
| BBP_20_2.0_2.0 | 7.3385 |
| Volume | 7.2780 |
| Volatility_20 | 5.8425 |
| BBB_20_2.0_2.0 | 4.6050 |
| LogReturn | 3.6397 |
| Sharpe_60 | 3.5318 |
| Trend_Strength | 3.5208 |
| Candle_Body | 3.2767 |
| Intraday_Range | 3.0139 |
| Alpha_60 | 2.3375 |
| GSPC_LogReturn | 1.3395 |
| Beta_60 | 1.1967 |
| GSPC_Close | 1.1848 |
| Month_Sin | 1.0181 |
| Sentiment_Score | 1.0027 |

### 迭代 4
- 最大VIF特征: EMA_26
- 最大VIF值: 151610.2340

| 特征 | VIF |
| --- | ---: |
| EMA_26 | 151610.2340 |
| BBM_20_2.0_2.0 | 128052.4186 |
| High | 75976.6853 |
| Low | 60035.5151 |
| Open | 55074.6603 |
| Close | 54308.0045 |
| SMA_5 | 51272.8496 |
| BBU_20_2.0_2.0 | 5208.7539 |
| Adj Close | 539.3855 |
| MACD_12_26_9 | 192.0186 |
| MACDs_12_26_9 | 178.7985 |
| MACDh_12_26_9 | 27.6489 |
| ATR_14 | 20.2778 |
| OBV | 16.8929 |
| RSI_14 | 11.0781 |
| BBP_20_2.0_2.0 | 7.3257 |
| Volume | 7.2767 |
| Volatility_20 | 5.8193 |
| BBB_20_2.0_2.0 | 4.5843 |
| LogReturn | 3.6345 |
| Trend_Strength | 3.5156 |
| Sharpe_60 | 3.5116 |
| Candle_Body | 3.2741 |
| Intraday_Range | 3.0136 |
| Alpha_60 | 2.3367 |
| GSPC_LogReturn | 1.3394 |
| Beta_60 | 1.1967 |
| GSPC_Close | 1.1848 |
| Month_Sin | 1.0181 |
| Sentiment_Score | 1.0027 |

### 迭代 5
- 最大VIF特征: High
- 最大VIF值: 75975.3679

| 特征 | VIF |
| --- | ---: |
| High | 75975.3679 |
| Low | 60027.9140 |
| BBM_20_2.0_2.0 | 57104.1831 |
| Open | 54867.7085 |
| Close | 54206.4378 |
| SMA_5 | 31921.4314 |
| BBU_20_2.0_2.0 | 5169.4085 |
| Adj Close | 538.9782 |
| MACD_12_26_9 | 184.8976 |
| MACDs_12_26_9 | 177.3799 |
| MACDh_12_26_9 | 26.8933 |
| ATR_14 | 20.2456 |
| OBV | 16.8581 |
| RSI_14 | 11.0726 |
| BBP_20_2.0_2.0 | 7.3244 |
| Volume | 7.2711 |
| Volatility_20 | 5.8181 |
| BBB_20_2.0_2.0 | 4.5842 |
| LogReturn | 3.6284 |
| Trend_Strength | 3.5152 |
| Sharpe_60 | 3.5071 |
| Candle_Body | 3.2699 |
| Intraday_Range | 3.0135 |
| Alpha_60 | 2.3324 |
| GSPC_LogReturn | 1.3391 |
| Beta_60 | 1.1966 |
| GSPC_Close | 1.1847 |
| Month_Sin | 1.0180 |
| Sentiment_Score | 1.0027 |

### 迭代 6
- 最大VIF特征: Low
- 最大VIF值: 58458.2989

| 特征 | VIF |
| --- | ---: |
| Low | 58458.2989 |
| BBM_20_2.0_2.0 | 56958.3336 |
| SMA_5 | 31651.8483 |
| Open | 30338.8338 |
| Close | 30250.7034 |
| BBU_20_2.0_2.0 | 5168.5940 |
| Adj Close | 537.5993 |
| MACD_12_26_9 | 184.8836 |
| MACDs_12_26_9 | 177.3645 |
| MACDh_12_26_9 | 26.7117 |
| ATR_14 | 17.7998 |
| OBV | 16.8297 |
| RSI_14 | 11.0707 |
| BBP_20_2.0_2.0 | 7.3239 |
| Volume | 7.1694 |
| Volatility_20 | 5.7439 |
| BBB_20_2.0_2.0 | 4.5830 |
| LogReturn | 3.6275 |
| Sharpe_60 | 3.5070 |
| Trend_Strength | 3.4858 |
| Candle_Body | 3.2671 |
| Intraday_Range | 2.8353 |
| Alpha_60 | 2.3324 |
| GSPC_LogReturn | 1.3359 |
| Beta_60 | 1.1966 |
| GSPC_Close | 1.1816 |
| Month_Sin | 1.0179 |
| Sentiment_Score | 1.0027 |

### 迭代 7
- 最大VIF特征: BBM_20_2.0_2.0
- 最大VIF值: 56851.7057

| 特征 | VIF |
| --- | ---: |
| BBM_20_2.0_2.0 | 56851.7057 |
| SMA_5 | 30920.3376 |
| Open | 15436.6199 |
| Close | 12444.0375 |
| BBU_20_2.0_2.0 | 5156.0600 |
| Adj Close | 534.8031 |
| MACD_12_26_9 | 184.5074 |
| MACDs_12_26_9 | 176.6914 |
| MACDh_12_26_9 | 26.3753 |
| OBV | 16.6576 |
| ATR_14 | 14.4105 |
| RSI_14 | 11.0706 |
| BBP_20_2.0_2.0 | 7.3232 |
| Volume | 6.8905 |
| Volatility_20 | 5.6147 |
| BBB_20_2.0_2.0 | 4.5829 |
| LogReturn | 3.6269 |
| Sharpe_60 | 3.5062 |
| Trend_Strength | 3.4616 |
| Candle_Body | 3.2667 |
| Intraday_Range | 2.6252 |
| Alpha_60 | 2.3321 |
| GSPC_LogReturn | 1.3339 |
| Beta_60 | 1.1962 |
| GSPC_Close | 1.1785 |
| Month_Sin | 1.0176 |
| Sentiment_Score | 1.0027 |

### 迭代 8
- 最大VIF特征: Open
- 最大VIF值: 15136.8038

| 特征 | VIF |
| --- | ---: |
| Open | 15136.8038 |
| SMA_5 | 13727.8315 |
| Close | 10426.8917 |
| BBU_20_2.0_2.0 | 3128.6551 |
| Adj Close | 533.5522 |
| MACD_12_26_9 | 182.0206 |
| MACDs_12_26_9 | 175.1480 |
| OBV | 16.6176 |
| MACDh_12_26_9 | 15.9390 |
| ATR_14 | 13.3661 |
| RSI_14 | 11.0533 |
| BBP_20_2.0_2.0 | 7.3190 |
| Volume | 6.8903 |
| Volatility_20 | 5.5652 |
| BBB_20_2.0_2.0 | 4.4130 |
| LogReturn | 3.6243 |
| Sharpe_60 | 3.5056 |
| Trend_Strength | 3.4440 |
| Candle_Body | 3.2640 |
| Intraday_Range | 2.6204 |
| Alpha_60 | 2.3214 |
| GSPC_LogReturn | 1.3337 |
| Beta_60 | 1.1962 |
| GSPC_Close | 1.1785 |
| Month_Sin | 1.0175 |
| Sentiment_Score | 1.0027 |

### 迭代 9
- 最大VIF特征: SMA_5
- 最大VIF值: 8615.8696

| 特征 | VIF |
| --- | ---: |
| SMA_5 | 8615.8696 |
| Close | 7974.3791 |
| BBU_20_2.0_2.0 | 3123.5041 |
| Adj Close | 533.5359 |
| MACD_12_26_9 | 180.7650 |
| MACDs_12_26_9 | 174.0054 |
| OBV | 16.5961 |
| MACDh_12_26_9 | 15.9378 |
| ATR_14 | 13.2828 |
| RSI_14 | 11.0451 |
| BBP_20_2.0_2.0 | 7.3134 |
| Volume | 6.8894 |
| Volatility_20 | 5.5651 |
| BBB_20_2.0_2.0 | 4.4127 |
| LogReturn | 3.5540 |
| Sharpe_60 | 3.5052 |
| Trend_Strength | 3.4348 |
| Candle_Body | 2.9984 |
| Intraday_Range | 2.6204 |
| Alpha_60 | 2.3204 |
| GSPC_LogReturn | 1.3218 |
| Beta_60 | 1.1961 |
| GSPC_Close | 1.1785 |
| Month_Sin | 1.0175 |
| Sentiment_Score | 1.0027 |

### 迭代 10
- 最大VIF特征: Close
- 最大VIF值: 2953.7060

| 特征 | VIF |
| --- | ---: |
| Close | 2953.7060 |
| BBU_20_2.0_2.0 | 2607.2798 |
| Adj Close | 533.2583 |
| MACD_12_26_9 | 176.5836 |
| MACDs_12_26_9 | 168.0920 |
| OBV | 16.5892 |
| MACDh_12_26_9 | 15.6292 |
| ATR_14 | 12.9891 |
| RSI_14 | 11.0451 |
| BBP_20_2.0_2.0 | 7.2880 |
| Volume | 6.8850 |
| Volatility_20 | 5.5231 |
| BBB_20_2.0_2.0 | 4.2532 |
| LogReturn | 3.5075 |
| Sharpe_60 | 3.5032 |
| Trend_Strength | 3.4190 |
| Candle_Body | 2.9983 |
| Intraday_Range | 2.6130 |
| Alpha_60 | 2.3176 |
| GSPC_LogReturn | 1.3185 |
| Beta_60 | 1.1961 |
| GSPC_Close | 1.1785 |
| Month_Sin | 1.0175 |
| Sentiment_Score | 1.0027 |

### 迭代 11
- 最大VIF特征: Adj Close
- 最大VIF值: 439.3318

| 特征 | VIF |
| --- | ---: |
| Adj Close | 439.3318 |
| BBU_20_2.0_2.0 | 422.5543 |
| MACD_12_26_9 | 175.9005 |
| MACDs_12_26_9 | 165.4275 |
| OBV | 16.2500 |
| MACDh_12_26_9 | 13.7409 |
| RSI_14 | 11.0051 |
| ATR_14 | 10.1414 |
| BBP_20_2.0_2.0 | 7.2866 |
| Volume | 6.7176 |
| Volatility_20 | 5.4332 |
| BBB_20_2.0_2.0 | 3.7597 |
| Sharpe_60 | 3.5032 |
| LogReturn | 3.5012 |
| Trend_Strength | 3.4081 |
| Candle_Body | 2.9980 |
| Intraday_Range | 2.6067 |
| Alpha_60 | 2.3147 |
| GSPC_LogReturn | 1.3154 |
| Beta_60 | 1.1959 |
| GSPC_Close | 1.1735 |
| Month_Sin | 1.0169 |
| Sentiment_Score | 1.0027 |

### 迭代 12
- 最大VIF特征: MACD_12_26_9
- 最大VIF值: 175.8702

| 特征 | VIF |
| --- | ---: |
| MACD_12_26_9 | 175.8702 |
| MACDs_12_26_9 | 165.3250 |
| BBU_20_2.0_2.0 | 39.7427 |
| OBV | 14.6450 |
| MACDh_12_26_9 | 13.5291 |
| RSI_14 | 10.9824 |
| ATR_14 | 8.7895 |
| BBP_20_2.0_2.0 | 7.2850 |
| Volatility_20 | 5.2938 |
| Volume | 5.2539 |
| BBB_20_2.0_2.0 | 3.7034 |
| Sharpe_60 | 3.5032 |
| LogReturn | 3.4993 |
| Trend_Strength | 3.4068 |
| Candle_Body | 2.9980 |
| Intraday_Range | 2.5982 |
| Alpha_60 | 2.3145 |
| GSPC_LogReturn | 1.3145 |
| Beta_60 | 1.1958 |
| GSPC_Close | 1.1646 |
| Month_Sin | 1.0169 |
| Sentiment_Score | 1.0026 |

### 迭代 13
- 最大VIF特征: BBU_20_2.0_2.0
- 最大VIF值: 39.7262

| 特征 | VIF |
| --- | ---: |
| BBU_20_2.0_2.0 | 39.7262 |
| OBV | 14.6445 |
| RSI_14 | 10.9824 |
| ATR_14 | 8.7892 |
| BBP_20_2.0_2.0 | 7.2830 |
| Volatility_20 | 5.2916 |
| Volume | 5.2463 |
| BBB_20_2.0_2.0 | 3.7033 |
| Sharpe_60 | 3.5022 |
| LogReturn | 3.4992 |
| Trend_Strength | 3.4065 |
| Candle_Body | 2.9979 |
| Intraday_Range | 2.5980 |
| Alpha_60 | 2.3145 |
| MACDs_12_26_9 | 2.1168 |
| GSPC_LogReturn | 1.3145 |
| MACDh_12_26_9 | 1.2079 |
| Beta_60 | 1.1957 |
| GSPC_Close | 1.1646 |
| Month_Sin | 1.0168 |
| Sentiment_Score | 1.0026 |

### 迭代 14
- 最大VIF特征: RSI_14
- 最大VIF值: 10.9501

| 特征 | VIF |
| --- | ---: |
| RSI_14 | 10.9501 |
| BBP_20_2.0_2.0 | 7.2640 |
| Volatility_20 | 5.0927 |
| ATR_14 | 3.9967 |
| Volume | 3.8034 |
| BBB_20_2.0_2.0 | 3.6978 |
| Sharpe_60 | 3.5003 |
| LogReturn | 3.4988 |
| Trend_Strength | 3.4064 |
| Candle_Body | 2.9979 |
| Intraday_Range | 2.5747 |
| OBV | 2.4610 |
| Alpha_60 | 2.3074 |
| MACDs_12_26_9 | 1.8819 |
| GSPC_LogReturn | 1.3140 |
| MACDh_12_26_9 | 1.1976 |
| Beta_60 | 1.1860 |
| GSPC_Close | 1.1395 |
| Month_Sin | 1.0164 |
| Sentiment_Score | 1.0026 |

### 迭代 15
- 最大VIF特征: Volatility_20
- 最大VIF值: 5.0552

| 特征 | VIF |
| --- | ---: |
| Volatility_20 | 5.0552 |
| ATR_14 | 3.9612 |
| Volume | 3.7959 |
| BBB_20_2.0_2.0 | 3.6668 |
| LogReturn | 3.4798 |
| Trend_Strength | 3.0664 |
| Candle_Body | 2.9977 |
| BBP_20_2.0_2.0 | 2.9968 |
| Intraday_Range | 2.5736 |
| OBV | 2.4555 |
| Sharpe_60 | 2.4334 |
| Alpha_60 | 2.3045 |
| MACDs_12_26_9 | 1.8292 |
| GSPC_LogReturn | 1.3137 |
| MACDh_12_26_9 | 1.1975 |
| Beta_60 | 1.1860 |
| GSPC_Close | 1.1393 |
| Month_Sin | 1.0154 |
| Sentiment_Score | 1.0026 |

### 迭代 16
- 最大VIF特征: ATR_14
- 最大VIF值: 3.9042

| 特征 | VIF |
| --- | ---: |
| ATR_14 | 3.9042 |
| Volume | 3.7653 |
| LogReturn | 3.4798 |
| Trend_Strength | 3.0509 |
| Candle_Body | 2.9972 |
| BBP_20_2.0_2.0 | 2.9936 |
| OBV | 2.4524 |
| Sharpe_60 | 2.3979 |
| Alpha_60 | 2.2998 |
| Intraday_Range | 2.0481 |
| BBB_20_2.0_2.0 | 1.9998 |
| MACDs_12_26_9 | 1.8255 |
| GSPC_LogReturn | 1.3104 |
| MACDh_12_26_9 | 1.1955 |
| Beta_60 | 1.1541 |
| GSPC_Close | 1.1219 |
| Month_Sin | 1.0154 |
| Sentiment_Score | 1.0024 |

## 最终特征VIF
| 特征 | VIF |
| --- | ---: |
| ATR_14 | 3.9042 |
| Volume | 3.7653 |
| LogReturn | 3.4798 |
| Trend_Strength | 3.0509 |
| Candle_Body | 2.9972 |
| BBP_20_2.0_2.0 | 2.9936 |
| OBV | 2.4524 |
| Sharpe_60 | 2.3979 |
| Alpha_60 | 2.2998 |
| Intraday_Range | 2.0481 |
| BBB_20_2.0_2.0 | 1.9998 |
| MACDs_12_26_9 | 1.8255 |
| GSPC_LogReturn | 1.3104 |
| MACDh_12_26_9 | 1.1955 |
| Beta_60 | 1.1541 |
| GSPC_Close | 1.1219 |
| Month_Sin | 1.0154 |
| Sentiment_Score | 1.0024 |