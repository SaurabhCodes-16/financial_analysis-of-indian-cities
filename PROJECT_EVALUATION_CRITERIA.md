# Project Evaluation Criteria

This document outlines the standards and expectations for project completion and assessment.

## Evaluation Rubric

| Criteria | Description | Project Implementation |
|----------|-------------|----------------------|
| **Problem Definition, Dataset & Understanding** | Clear problem statement, well-defined objectives, dataset collected and properly described (features, size, source), and relevance justified. | **Financial Analysis of Indian Cities:** Analyzing financial data from Indian cities to identify patterns, vulnerabilities, and disparities. Dataset includes expense categories (Rent, Groceries, Transport, etc.), income levels, and savings potential. Goal: Classify city vulnerability levels and identify financial characteristics across different city tiers. |
| **Data Preprocessing** | Handling missing values, encoding, normalization, feature selection, solving class imbalance issues, and data cleaning completed. | **Phase 1 - EDA & Feature Engineering:** Missing values handled using median imputation, feature engineering applied on expense columns and savings potential, data standardized for clustering analysis, categorical encoding applied, outlier treatment performed. |
| **Implementation of Technique** | Appropriate algorithms selected and implemented correctly as per the suggestion given earlier | **Phase 2 - Clustering:** K-Means clustering to identify city groups (clusters 0-3). **Phase 3 - Classification:** XGBoost classifier trained to predict city vulnerability levels. **Random Forest Model:** Alternative classification approach for comparison. |
| **Results, Analysis & Visualization** | Results generated with evaluation metrics, proper interpretation, and meaningful visualizations (graphs/charts) | **Phase 4 - SHAP Analysis:** Feature importance analysis and prediction explanations. **Phase 5 - Disparity Analysis:** Vulnerability disparities across city tiers. Model evaluation metrics (accuracy, precision, recall, F1-score), cluster profiling visualizations, prediction scatter plots, SHAP summary plots. |
| **Progress Towards Completion** | Around 70% completion achieved | All 5 phases implemented with outputs generated. EDA completed, clustering analysis done, classification models trained, explainability analysis done, disparity analysis completed. Model predictions and visualizations saved. |
| **Individual Contribution (Viva + Explanation)** | Each member explains their part, answers questions, and demonstrates understanding | Each team member owns specific phases: Phase 1 (EDA), Phase 2 (Clustering), Phase 3 (Classification), Phase 4 (SHAP), Phase 5 (Disparity). Members can explain their implementation, results interpretation, and project impact. |

## Standard for Success

✅ **Project work meets the expected standard when all criteria are satisfactorily addressed.**

## Key Deliverables

- [ ] Clear problem statement with well-defined objectives
- [ ] Dataset description (features, size, source) with relevance justification
- [ ] Completed data preprocessing pipeline
- [ ] Implemented algorithms with documentation
- [ ] Generated results with evaluation metrics
- [ ] Meaningful visualizations and interpretations
- [ ] Progress documentation (70% completion target)
- [ ] Team member contributions documented
- [ ] Prepared explanations for viva/presentation

## Notes

Ensure to maintain clear documentation throughout the project phases for easy tracking and demonstration of progress.
