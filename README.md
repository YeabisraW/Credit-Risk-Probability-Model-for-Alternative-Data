## Credit Scoring Business Understanding
**1. Basel II Accord & Model Interpretability**  
The Basel II Capital Accord emphasizes measuring credit risk to ensure banks hold adequate capital for potential losses. It requires a well-documented and interpretable credit risk model because:  
- Regulators must understand how risk is quantified.  
- Decisions impacting loans must be justifiable.  
- Transparency ensures that automated predictions can be audited and are compliant with risk policies.  

**2. Need for a Proxy Variable**  
Since the dataset lacks a direct “default” label, we must define a **proxy variable** (e.g., customers with high negative balances, delayed payments, or abnormal RFM patterns).  
- **Why necessary:** It allows us to approximate which customers are high risk.  
- **Business risks:** Predictions are only as good as the proxy. Incorrect proxies can lead to:  
  - Approving loans to risky customers → financial loss  
  - Rejecting creditworthy customers → lost revenue and poor customer experience  

**3. Model Trade-offs**  
- **Simple, interpretable models (Logistic Regression, WoE coding):**  
  - Pros: Transparent, easier to justify to regulators, robust to overfitting.  
  - Cons: May underperform on complex patterns.  
- **Complex, high-performance models (Gradient Boosting, Random Forest):**  
  - Pros: High predictive power, captures nonlinear relationships.  
  - Cons: Harder to explain, may require additional interpretation tools (SHAP, LIME).  

**Conclusion:**  
In a regulated financial context, a balance is needed — start with interpretable models, then optionally improve performance with complex models if interpretability tools are available.
