# Assignment 04 Interpretation Memo

**Student Name:** [Tallulah Pascucci]
**Date:** [2/13/26]
**Assignment:** REIT Annual Returns and Predictors (Simple Linear Regression)

---

## 1. Regression Overview

You estimated **three** simple OLS regressions of REIT *annual* returns on different predictors:

| Model | Y Variable | X Variable | Interpretation Focus |
|-------|------------|------------|----------------------|
| 1 | ret (annual) | div12m_me | Dividend yield |
| 2 | ret (annual) | prime_rate | Interest rate sensitivity |
| 3 | ret (annual) | ffo_at_reit | FFO to assets (fundamental performance) |

For each model, summarize the key results in the sections below.

---

## 2. Coefficient Comparison (All Three Regressions)

**Model 1: ret ~ div12m_me**
- Intercept (β₀): 0.1082 (SE: 0.006, p-value: 0.000)
- Slope (β₁): -0.0687 (SE: 0.032, p-value: 0.035)
- R²: 0.002 | N: 2527

**Model 2: ret ~ prime_rate**
- Intercept (β₀): 0.1998 (SE: 0.016, p-value: 0.000)
- Slope (β₁): -0.0194 (SE: 0.003, p-value: 0.000)
- R²: 0.016 | N: 2527

**Model 3: ret ~ ffo_at_reit**
- Intercept (β₀): 0.0973 (SE: 0.009, p-value: 0.000)
- Slope (β₁): 0.5770 (SE: 0.567, p-value: 0.309)
- R²: 0.000 | N: 2518

*Note: Model 3 may have fewer observations if ffo_at_reit has missing values; statsmodels drops those rows.*

---

## 3. Slope Interpretation (Economic Units)

**Dividend Yield (div12m_me):**
- A 1 percentage point increase in dividend yield (12-month dividends / market equity) is associated with a [slope value] change in annual return.
- [Your interpretation: Is higher dividend yield associated with higher or lower returns? Why might this be?] 
A 1 percentage point increase in dividend yield is associated with a -0.000687 change in annual return. Higher dividend yield is associated with lower returns even though the magnitude is small.

**Prime Loan Rate (prime_rate):**
- A 1 percentage point increase in the year-end prime rate is associated with a [slope value] change in annual return.
- [Your interpretation: Does the evidence suggest REIT returns are sensitive to interest rates? In which direction?]
A 1 percentage point increase in the year-end prime rate is associated with a -0.000194 change in annual return. The evidence suggests REIT returns are sensitive to interest rates and higher rates are linked to lower returns.

**FFO to Assets (ffo_at_reit):**
- A 1 unit increase in FFO/Assets (fundamental performance) is associated with a [slope value] change in annual return.
- [Your interpretation: Do more profitable REITs (higher FFO/Assets) earn higher returns?]
A 1 unit increase in FFO/Assets is associated with a 0.5770 change in annual return. The relationship is not reliable in this data.

---

## 4. Statistical Significance

For each slope, at the 5% significance level:
- **div12m_me:** Significant — Dividend yield has a small negative relationship with annual returns.
- **prime_rate:** Significant — Higher prime rates are associated with lower annual returns.
- **ffo_at_reit:** Not significant — No clear linear relationship with returns at the 5% level.

**Which predictor has the strongest statistical evidence of a relationship with annual returns?** prime_rate

---

## 5. Model Fit (R-squared)

Compare R² across the three models:
- [Your interpretation: Which predictor explains the most variation in annual returns? Is R² high or low in general? What does this suggest about other factors driving REIT returns?]
prime_rate explains the most variation but all R² values are very low. This suggests most variation in REIT annual returns is driven by other factors not shown.

---

## 6. Omitted Variables

By using only one predictor at a time, we might be omitting:
- [Market return]: Shows broad market movements that affect all REITs.
- [Leverage]: Higher leverage can amplify returns and risk.
- [Property size]: Different sizes have different risk and return profiles.

**Potential bias:** If omitted variables are correlated with both the X variable and ret, our slope estimates may be biased. [Brief discussion of direction if possible]
If higher dividend yield or interest rates coincide with weaker conditions, the negative slopes could reflect omitted risk, biasing coefficients more negative.
---

## 7. Summary and Next Steps

**Key Takeaway:**
[2-3 sentences summarizing which predictor(s) show the strongest relationship with REIT annual returns and whether the evidence is consistent with economic theory]
Prime rates show the strongest relationship with REIT annual returns, with higher rates linked to lower returns, consistent with discount-rate and financing-cost. Dividend yield has a small negative relationship, while Assets does not show a relationship. Overall, the evidence explains a small amount of annual return variation.

**What we would do next:**
- Extend to multiple regression (include two or more predictors)
- Test for heteroskedasticity and other OLS assumption violations
- Examine whether relationships vary by time period or REIT sector

---

## Reproducibility Checklist
- [x] Script runs end-to-end without errors
- [x] Regression output saved to `Results/regression_div12m_me.txt`, `regression_prime_rate.txt`, `regression_ffo_at_reit.txt`
- [x] Scatter plots saved to `Results/scatter_div12m_me.png`, `scatter_prime_rate.png`, `scatter_ffo_at_reit.png`
- [x] Report accurately reflects regression results
- [x] All interpretations are in economic units (not just statistical jargon)
