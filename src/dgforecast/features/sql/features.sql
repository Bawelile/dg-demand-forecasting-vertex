-- Lesson 4 (BigQuery version). The same leak-free features as features/build.py, in SQL,
-- so they run where the data lives. Replace `PROJECT.DATASET` (cloud/bq_features.py does it).
--
-- Rule: every sales feature looks back at least 14 days (we forecast 14 days ahead).
-- Stockout days are blanked (NULL) before computing lags and averages.

CREATE OR REPLACE TABLE `PROJECT.DATASET.features` AS
WITH clean AS (
  SELECT
    s.*,
    IF(s.stockout_flag = 0, s.units_sold, NULL) AS clean_units
  FROM `PROJECT.DATASET.sales_history` AS s
),
lagged AS (
  SELECT
    c.*,
    LAG(clean_units, 14)  OVER w AS lag_14,
    LAG(clean_units, 21)  OVER w AS lag_21,
    LAG(clean_units, 28)  OVER w AS lag_28,
    LAG(clean_units, 364) OVER w AS lag_364,
    -- windows end 14 rows back: "ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING" = the 7 days ending 14 days ago
    AVG(clean_units)    OVER (w ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING) AS rmean_7,
    AVG(clean_units)    OVER (w ROWS BETWEEN 41 PRECEDING AND 14 PRECEDING) AS rmean_28,
    AVG(clean_units)    OVER (w ROWS BETWEEN 69 PRECEDING AND 14 PRECEDING) AS rmean_56,
    STDDEV(clean_units) OVER (w ROWS BETWEEN 41 PRECEDING AND 14 PRECEDING) AS rstd_28
  FROM clean AS c
  WINDOW w AS (PARTITION BY store_id, item_id ORDER BY date)
)
SELECT
  l.date, l.store_id, l.item_id,
  l.lag_14, l.lag_21, l.lag_28, l.lag_364, l.rmean_7, l.rmean_28, l.rmean_56, l.rstd_28,
  AVG(l.rmean_28) OVER (PARTITION BY l.date, l.item_id)  AS item_chain_rmean_28,
  SUM(l.rmean_28) OVER (PARTITION BY l.date, l.store_id) AS store_total_rmean_28,
  l.price, l.price / i.base_price AS price_index, l.on_promo,
  cal.dow, cal.dom, cal.month, cal.is_snap_window, cal.is_paycheck_friday, cal.is_holiday, cal.days_to_holiday,
  st.format AS store_format, st.region, i.category,
  DATE_DIFF(DATE(l.date), DATE(st.open_date), DAY) AS store_age_days,
  l.units_sold, l.stockout_flag
FROM lagged AS l
JOIN `PROJECT.DATASET.calendar` AS cal USING (date)
JOIN `PROJECT.DATASET.stores`   AS st  USING (store_id)
JOIN `PROJECT.DATASET.items`    AS i   USING (item_id)
WHERE l.rmean_28 IS NOT NULL;
-- Note: store_format, region, and category stay as text here. Map them to the same integer
-- codes as build.py (FORMATS, REGIONS, CATEGORIES order) before training, or let the
-- training step encode them.
