# Data Splits Documentation

**⚠️ TEMPLATE - Fill in your actual dates before publication**

## Training Set

- **Start Date**: [YYYY-MM-DD] (e.g., 2020-01-01)
- **End Date**: [YYYY-MM-DD] (e.g., 2023-12-31)
- **Duration**: [X years/months] (e.g., 4 years)
- **Trading Days**: [X days] (e.g., 1008 trading days)
- **Tickers**: [List or count] (e.g., 15 tickers: AAPL, TSLA, NVDA, ...)
- **Total Samples**: [X samples] (e.g., 15,120 samples = 15 tickers × 1008 days)

## Validation Set (if applicable)

- **Start Date**: [YYYY-MM-DD] (e.g., 2024-01-01)
- **End Date**: [YYYY-MM-DD] (e.g., 2024-06-30)
- **Duration**: [X months] (e.g., 6 months)
- **Trading Days**: [X days] (e.g., 126 trading days)
- **Gap from Training**: [X days] (e.g., 1 day gap on 2023-12-31 to 2024-01-01)

## Test Set

- **Start Date**: [YYYY-MM-DD] (e.g., 2024-07-01)
- **End Date**: [YYYY-MM-DD] (e.g., 2024-12-31)
- **Duration**: [X months] (e.g., 6 months)
- **Trading Days**: [X days] (e.g., 126 trading days)
- **Gap from Validation**: [X days] (e.g., 1 day gap on 2024-06-30 to 2024-07-01)

## Temporal Validation

- **Method**: Walk-forward validation
- **Train Window**: [X days] (e.g., 252 trading days = 1 year)
- **Test Window**: [X days] (e.g., 21 trading days = 1 month)
- **Step Size**: [X days] (e.g., 21 trading days)
- **Total Splits**: [X splits] (e.g., 12 splits)

## Data Coverage

- **Total Date Range**: [START] to [END] (e.g., 2020-01-01 to 2024-12-31)
- **Total Trading Days**: [X days]
- **Market Regimes Covered**:
  - Bull market: [Date range]
  - Bear market: [Date range]
  - High volatility: [Date range]
  - Low volatility: [Date range]

## Class Balance

- **Training Set**:
  - Movement = 0 (DOWN): [X samples] ([X%])
  - Movement = 1 (UP): [X samples] ([X%])
  
- **Test Set**:
  - Movement = 0 (DOWN): [X samples] ([X%])
  - Movement = 1 (UP): [X samples] ([X%])

## Notes

- All splits use **strict temporal ordering** (no random shuffling)
- **Gap periods** between train/val/test prevent leakage
- **No overlap** between any sets
- All dates are **trading days only** (excludes weekends/holidays)

---

**To fill this out:**
1. Run your data generation script
2. Extract the actual dates from your train/test split
3. Calculate statistics (sample counts, class balance)
4. Replace all [PLACEHOLDER] values with actual data

