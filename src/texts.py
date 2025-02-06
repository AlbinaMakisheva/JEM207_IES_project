def introduction_text():
    return """
        This project focuses on analyzing the impact of key events on stock prices using event-driven analysis. 
        We examine the relationship between the global COVID-19 pandemic and the stock prices of Pfizer Inc. 
        by investigating the impact of significant events, such as the WHO declaring a pandemic and the approval of the COVID-19 vaccine.
        
        The project involves multiple types of analyses:
        - **Regression Analysis:** We perform multiple linear regression to understand the relationships between stock prices and COVID-related variables.
        - **Event Impact Analysis:** We assess the direct impact of major events on stock returns.
        - **Logistic Regression & Random Forest:** We build models to classify stock movements and identify important features.

        The datasets used in this analysis include global COVID-19 case data and stock price data for Pfizer. These datasets are combined to enable the analysis of event-driven changes in stock behavior.
    """

def stock_price_analysis_text():
    return """
        - This graph shows **Pfizer's stock price (USD) over time**, with key COVID-19 events marked by vertical dashed lines.
        - **Observations:**
            - **WHO Declares Pandemic (March 2020)**: Pfizer's stock initially experienced **volatility**, showing no immediate uptrend.
            - **First Vaccine Approval (December 2020)**: The stock **rallied significantly**, possibly indicating investor confidence in vaccine-driven revenue.
            - **Vaccination Threshold Reached (July 2021)**: The stock peaked, likely due to strong vaccine sales expectations.
            - **Relaxation of Lockdowns (May 2022)**: Stock **began to decline**, reflecting reduced pandemic-related revenue expectations.
            - **China Easing Zero-COVID Policy (January 2023)**: The downward trend continued as the pandemic's impact on the stock market faded.

        - **Overall Trend:**
            - The stock price was **stable before the pandemic**, **volatile at the start**, **rallied after vaccine approvals**, and **declined post-pandemic** as COVID-related revenues decreased.
    """

def covid_cases_analysis_text():
    return """
        - This graph presents **global new COVID-19 cases**, with **red bars representing raw values** and a **blue line for smoothed trends**.
        - **Observations:**
        - **Early waves in 2020**: The number of cases increased significantly after the pandemic declaration.
        - **Major peaks in late 2021 and early 2022**: These align with **Delta and Omicron variant surges**.
        - **Case decline after mid-2022**: Due to **mass vaccinations, natural immunity, and reduced testing**.

        - **Connection to Pfizer Stock Prices:**
        - **Early pandemic surges did not significantly increase Pfizer's stock price**.
        - **The biggest stock price rise happened after vaccine approvals**, not during case surges.
        - **After case peaks and easing of restrictions, Pfizer’s stock declined**, suggesting revenue expectations shifted.
    """

def filtering_text():
    return """
        The purpose of filtering data around key events is to **analyze patterns or trends before and after these events** to see their effects, such as **stock price movements, changes in COVID-19 cases, etc.** 

        ### Why Filter Data?
        - **It isolates the data** to focus on the periods **directly before and after key events** to study their impact.
        - For example, if analyzing the event **"First Vaccine (2020-12-08)"** with a **1-month window**:
        - The filtered dataset will include data from **2020-11-08 to 2021-01-08**.
        - **Applying this method to all events** creates a dataset segmented into **smaller windows**, allowing a **detailed analysis** of each event's impact.
    """

def autocorrelation_text():
    return """
            With this analysis, we aim to investigate the autocorrelation between variables in order to help us building models that predict new trends. 
            It is worth noticing that most variables namely total_cases, new_cases, new_cases_smoothed, total_deaths and new_deaths present high autocorrelation values. This indicate that many of our variables are highly dependent on their previous values at lag 1.
            However, one could also argue that this is a common feature of cumulative variables (which give the cumulative sums over time) and smoothed variables (which are designed to reduce short-term fluctuations).
            Nevertheless, we will focus on variables with much lower correlation, since they are the ones which might add much new information to our regressions.
     """

def coefficients_text_firstLR():
    return """   
        The coefficient of determination of 0.0001 indicates that only 0.01% of the variation in daily returns is explained by the independent variables. The model does not have strong predictive power.
                     
        Regarding the interpretation of the coefficients:
                     
            > The intercept (baseline return) suggests that even in the absence of pandemic-related factors (all independent variables being equal to zero), the Pfizer stock would still have a small expected daily return of 0.05%
                     
            > new_cases_dummy_interaction and new_deaths_dummy_interaction: No meaningful effect on stock returns. This could be due to multicollinearity with other pandemic-related variables
             
            > reproduction_rate_vaccinations_lag_1: A small positive effect which suggests that under certain conditions—such as lockdowns or significant events—rising case reproduction rates, combined with vaccinations, may slightly boost stock returns. However, the effect size remains minimal.
                     
            > vaccination_signal_lag_180: a negative coefficient suggesting that vaccination trends six months prior might have had a weak adverse effect on stock performance. This could indicate that markets reacted to early vaccination signals by adjusting stock prices in advance.

            > deaths_to_cases_ratio: A higher ratio of deaths to cases is negatively correlated with stock returns, which is expected as worsening pandemic conditions may negatively affect the market. However, the magnitude is small.
                     
            > new_deaths_dummy_interaction: no meaningful effect suggests that new deaths, when interacting with specific events (e.g., lockdowns or policy announcements), had a negligible effect on Pfizer's stock.
                     
            > Dummy_Variable_lag_180: A negative impact of the event dummy variable on stock returns when lagged 180 days. Suggests that certain events had a slightly negative influence over the long term.                          
    """

def coefficients_text_secondLR():
    return """
        The coefficient of determination (R²) of **0.7508** indicates that **75.08%** of the variation in the dependent variable (new deaths smoothed) is explained by the independent variables. This suggests a strong explanatory model.

        Regarding the interpretation of the coefficients:

        > **new_cases_smoothed:**  
        A strong **positive correlation** between new cases and new deaths, which aligns with expectations—higher infection rates typically lead to more fatalities.

        > **female_smokers_rate:**  
        A **positive relationship** between female smoking rates and new deaths, suggesting that smoking may worsen COVID-19 outcomes, potentially due to respiratory vulnerabilities.

        > **stringency_index:**  
        A **small positive effect**, implying that stricter lockdown measures are slightly correlated with an increase in reported deaths. This could result from delayed reporting during strict lockdowns or high case surges prompting stricter measures.

        > **total_vaccination_rate:**  
        A **strong negative correlation**, indicating that higher vaccination rates significantly reduce new deaths. This supports the expected protective effect of widespread immunization.

        > **Dummy_Variable:**  
        A **negative coefficient**, suggesting that during specific event periods (e.g., lockdowns or vaccine rollouts), the number of deaths **decreased**, likely due to intervention measures.

        > **new_cases_dummy_interaction:**  
        No meaningful interaction effect between new cases and the dummy variable, indicating that during event periods, new cases did not significantly alter the expected deaths trend.

    """

def residual_text():
    return """
        For the first regression, there is a noticeable clustering of residuals at certain ranges of fitted values. This indicates potential issues with heteroscedasticity (non-constant variance) or omitted variable bias.
        For the second regression, the residuals display a strong pattern,  which could be understood as a sign of heteroscedasticity.The variance of residuals increases with higher fitted values, indicating that the model struggles to capture variability at those levels.
        In summary, through graphical visualization, the residuals appear to increase, indicating that they are not constant over time.
                    
        Regarding the Distribution of Residuals, the residuals of both regressions are approximately normal but have visible tails or are slightly skewed. This indicates a reasonable fit but with room for improvement in capturing the underlying patterns.
        Given this, and to improve our regressions, we should apply the necessary corrections in order to restore the classical assumption of constant variance. It is also important to highlight that these procedures, while not fully resolving the problem of heteroscedasticity, restore the validity of statistical inference in large samples.
    """

def heteroscedasticity_text():
    return """
        The test statistic and a p-value of 0.0021 and 0.0000 (for the first and second regressions respectively), indicate heteroscedasticity is present in the residuals of the third regression. Heteroscedasticity implies that the variance of the residuals is not constant, violating a key assumption of ordinary least squares (OLS). Weighted Least Squares (WLS) was applied to address heteroscedasticity, and a corrected model was obtained.

        Heteroscedasticity implies that the variance of the residuals is not constant, violating a key assumption of OLS. To address this, Weighted Least Squares (WLS) was applied, yielding the following insights:

        - **High R² (uncentered):** Indicates the model explains almost all the variation in the dependent variable. However, caution is needed due to potential multicollinearity or numerical issues.
        - **Significant coefficients:** All predictors have very low p-values (<0.05), suggesting they are statistically significant.
    """

def logReg_text():
    return """
        > The model’s accuracy is slightly above random guessing (50%). This suggests that it has limited predictive power and may not effectively capture the relationship between the independent variables and stock movement.)
                
        > The recall for class 1 (increase in stock price) is very low (0.03%), indicating that the model is almost always predicting no increase (class 0).
                
        > Precision for class 1 (56.52%) suggests that when the model does predict an increase, it is correct more than half the time. However, since recall is extremely low, it is rarely making this prediction.
                
        > F1-score for class 1 (0.07%) indicates that the model performs poorly in identifying stock price increases.
                
        > The ROC AUC score of 0.51 is nearly equivalent to random guessing (0.50). This further confirms that the model does not provide a meaningful distinction between increases and decreases in stock price.
    """

def logReg_coef_text():
    return """      
        > new_cases_smoothed: A small negative effect, suggesting that an increase in new cases slightly reduces the probability of stock price increase. This contradicts the idea that higher cases could lead to increased vaccine demand benefiting Pfizer.

        > new_deaths_smoothed:A negative effect, meaning that more deaths slightly decrease the probability of stock price increasing. 
                 
        > new_vaccinations_smoothed: no meaningful effect, implying that the rate of new vaccinations does not significantly influence short-term stock movement.
                 
        > Dummy_Variable: no significant effect, suggesting that pandemic-related events (as captured by the dummy variable) do not directly influence stock movement.

        > deaths_to_cases_ratio: a negative effect, indicating that a higher ratio of deaths to cases is correlated with a slightly lower probability of stock price increasing.

        > interaction_term (new cases × event dummy): A small positive effect, suggesting that during certain pandemic-related events, an increase in new cases is slightly associated with an increased probability of stock price rising. However, the effect is minimal.
    """

def ext_logRed_text():
    return """       
        > The accuracy of the model is 51.73%, meaning the model is slightly better than random guessing.
        > Precision and recall metrics indicate that the model is heavily biased toward predicting class 0 (no stock increase).
        > The ROC AUC score of 0.51 suggests the model has very weak predictive power, nearly random in distinguishing stock movements.
    """

def ext_logReg_coef_text():
    return """
        > new_cases_smoothed_diff_lag_1: A negative effect, meaning an increase in new cases (with a 1-day lag and differencing applied) is associated with a slightly lower probability of stock increase. This could indicate that the market reacts negatively to short-term spikes in new cases
                 
        > new_deaths_smoothed_diff_lag_1: A negative coefficient suggests that an increase in new deaths (1-day lagged and differenced) reduces the probability of stock price rising.

        > new_vaccinations_smoothed_diff_lag_180: The coefficient is essentially zero, indicating that the long-term trend of vaccinations (lagged 180 days) does not significantly impact stock movement. This suggests that vaccination trends are already priced into the market, and short-term fluctuations in vaccinations do not affect Pfizer's stock significantly.

        > Dummy_Variable_diff_lag_180: A positive effect, meaning that pandemic-related event periods (lagged 180 days) are linked to a slightly higher probability of stock price increasing.

        > deaths_to_cases_ratio: A strong negative effect, meaning that when the ratio of deaths to cases increases, the probability of a stock increase decreases.

        > interaction_term: A positive effect, suggesting that when new cases and the dummy variable (representing major pandemic events) interact, it slightly increases the probability of stock price rising.
    """

def limitations_text():
    return """
        > Despite the poor predictive performance and low explanatory power of our regressions, our primary objective was to gain hands-on experience with Python and develop confidence in working with real-world datasets. While we applied various preprocessing techniques—such as handling heteroscedasticity, differencing, and lagging—to improve our model, the results suggest that significant noise and omitted variables likely influenced our findings.
            
        > One potential reason for the limited explanatory power of our models is the narrow focus on Pfizer's stock. Expanding the dataset to include other pharmaceutical companies could provide a broader and more comparative analysis, potentially leading to better model performance. However, incorporating multiple companies would have required extensive data collection and cleaning, significantly increasing the complexity and scope of the project. 
            
        > Additionally, stock prices are influenced by a wide range of macroeconomic, financial, and industry-specific factors, many of which were not included in our dataset.
            
        > Despite these limitations, it provided valuable insights into the challenges of working with financial data, reinforcing the importance of careful variable selection, proper model specification, and robust validation techniques.    
    """


