## Credit Scoring Business Understanding

### 1. Credit Risk Context
In recent times, banks, micro-finance banks, and other lending institutions in Nigeria have increasingly offered various credit facilities. While this generates revenue, it also increases exposure to credit risk. Lending organizations must carefully assess **who qualifies** for credit to minimize potential losses.

Traditionally, many banks rely on **credit scorecards** or the personal judgment of lending officers to determine creditworthiness. However, cyclical financial instabilities have made **robust, data-driven credit risk modeling critical**.

---

### 2. Basel II Accord & Regulatory Requirements
The **Basel II Capital Accord** prescribes the minimum regulatory capital a financial institution must hold to cushion against unexpected losses. Under the **Advanced Internal Ratings-Based (A-IRB) approach**, banks can develop internal models to estimate three key parameters:

- **Probability of Default (PD):** Likelihood that a borrower will default.  
- **Loss Given Default (LGD):** Expected percentage loss if a borrower defaults.  
- **Exposure at Default (EAD):** Total monetary exposure at the time of default.  

These parameters allow banks to determine the minimum capital required while complying with regulatory standards.

---

### 3. Importance of Proxy Variables
In the absence of direct default labels, proxy variables derived from **transactional and behavioral data** (e.g., Recency, Frequency, and Monetary patterns) are necessary to approximate credit risk. These proxies enable machine learning models to produce **risk probability scores**, which inform loan approval decisions.

**Business risks:** Incorrect proxies may result in:  
- Granting credit to high-risk customers → financial loss  
- Rejecting creditworthy customers → lost revenue and poor customer relations

---

### 4. Modeling Trade-offs
- **Simple, interpretable models (e.g., Logistic Regression, WoE coding):**  
  - Pros: Transparent, easy to justify to regulators, robust.  
  - Cons: May underperform on complex patterns.  

- **Complex models (e.g., Gradient Boosting, Random Forest):**  
  - Pros: High predictive power, captures nonlinear relationships.  
  - Cons: Harder to interpret; requires additional tools like SHAP or LIME.

**Conclusion:** In regulated financial contexts, a balance is essential: start with interpretable models for compliance, and optionally use high-performance models with interpretability support to improve predictive accuracy.
