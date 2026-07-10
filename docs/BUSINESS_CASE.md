# 💼 Business Case

## The problem

A used-car marketplace needs an **instant, trustworthy price estimate** for every
car a seller lists, plus a **budget band** (Low / Medium / High) to power search
and filtering. Manual appraisal is slow, inconsistent, and doesn't scale.

## Why it matters (value hypothesis)

| Lever | Mechanism | Signal it moves |
| :---- | :-------- | :-------------- |
| Faster listing | Auto-suggest a price at listing time | Listing completion rate |
| Pricing confidence | A defensible, data-backed number | Seller trust, fewer relistings |
| Better discovery | Consistent budget bands | Buyer search relevance |
| Low serving cost | A ~1 MB model, millisecond inference | Infra cost per prediction |

## The ML framing

- **Primary task — regression:** predict `selling_price` in ₹ Lakhs.
- **Secondary output — band:** **derived** from the predicted price at the dataset
  terciles, so the band and the number can never disagree. (Training a separate
  classifier, as an earlier iteration did, added a second failure mode and a
  lower-accuracy signal for no benefit.)

## KPIs & the ship/no-ship gate

The model is defined in code (`car_pricing.config.KPI`) and enforced automatically
by the training pipeline — a model that misses **any** threshold is refused:

| KPI | Threshold | Rationale | Achieved |
| :-- | :-------- | :-------- | :------- |
| **MAE** | ≤ **₹1.0 Lakh** | "Typical miss under ₹1,00,000" is intuitive to a seller | **₹0.66 L** ✅ |
| **R²** | ≥ **0.85** | Explain most of the price variance | **0.957** ✅ |
| **Band accuracy** | ≥ **70 %** | Correct budget bucket most of the time | **85.8 %** ✅ |

> **MAE over RMSE for the headline:** because prices are right-skewed, MAE ("average
> error in rupees") is the number a business stakeholder actually understands; RMSE
> is reported alongside for the modelling audience.

## Scope & assumptions

- **In scope:** cars represented in the Cars24 dataset (41 brands, mainstream
  Indian market), point-estimate pricing.
- **Out of scope:** brand-new cars, exotic/vintage vehicles, condition/accident
  history (not in the data), and real-time market shocks.
- **Assumption:** historical listing prices are a reasonable proxy for fair resale
  value. Predictions are decision *support*, not a guaranteed sale price.

## How success is monitored in production

Track live **MAE** against the ₹1 L gate and **input drift** (e.g. new models,
shifting age/km distributions). Either crossing a threshold triggers a retrain via
the same KPI-gated pipeline (see [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md)).
