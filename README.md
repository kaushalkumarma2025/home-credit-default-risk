🚀 Home Credit Default Risk — End-to-End Credit Risk Modeling System



📌 Objective

Build a production-ready machine learning system to predict probability of loan default and convert predictions into actionable lending decisions (approve/reject) under real-world business constraints.



\---



🧠 Business Context



In lending, approving more loans increases revenue but also increases default risk. This project models that trade-off using cost-sensitive decisioning:



\- False Negative (missed defaulter): ₹5,00,000 loss  

\- False Positive (wrong rejection): ₹15,000 cost  



Goal: Maximize net portfolio value, not just model accuracy.



\---



📊 Dataset



\- 307,511 loan applicants  

\- 122 features across 8 tables  

\- Default rate: 8.07%  



Includes bureau data, previous loans, installments, demographics.



\---



⚙️ Pipeline



\- Data cleaning (missing values, outliers)

\- Feature engineering (53 new features)

\- Models: Logistic Regression, LightGBM

\- Evaluation: AUC, KS, Decile Capture

\- Threshold optimization with business constraints



\---



📈 Model Performance



| Metric | Logistic | LightGBM |

|--------|----------|----------|

| ROC-AUC | 0.7664 | 0.7834 |

| KS | 40.1% | 42.7% |

| Top 20% Capture | \~30% | 56.5% |



\---



💰 Business Impact



\- Approval Rate: 70%  

\- Threshold: 0.50  



Test set results:

\- Default loss prevented: ₹94.7 crore  

\- Opportunity cost: ₹7.2 crore  

\- Net value: ₹93.9 crore  



\---



🎯 Key Insights



\- External credit scores are strongest predictors  

\- Feature engineering adds real value  

\- Threshold is a business decision  

\- Model reduces losses, not just improves accuracy  

\- Class imbalance requires AUC/KS, not accuracy  



\---



🏗️ Production



\- Modular pipeline  

\- Serialized models (.pkl)  

\- End-to-end scoring ready  



\---



🧰 Tech Stack



Python, Pandas, Scikit-learn, LightGBM



\---



📂 Structure



home-credit-default-risk/

├── Home\_credit.ipynb

├── models/

├── outputs/

└── .gitignore



\---



⚠️ Note



Dataset excluded due to size (>2GB)



\---



