CREATE DATABASE Hotel
--
USE Hotel
GO
-- Get all data from 2018, 2019, 2020 table into one table
WITH Hotels AS (
	SELECT * FROM [2018]
	UNION
	SELECT * FROM [2019]
	UNION
	SELECT * FROM [2020])

SELECT * FROM HOTELS

-- Calculate hotel's revenue
WITH Hotels AS (
	SELECT * FROM [2018]
	UNION
	SELECT * FROM [2019]
	UNION
	SELECT * FROM [2020])

SELECT 
	(stays_in_week_nights + stays_in_weekend_nights)*adr as Revenue
FROM Hotels

-- Check whether the revenue increasing by year
WITH Hotels AS (
	SELECT * FROM [2018]
	UNION
	SELECT * FROM [2019]
	UNION
	SELECT * FROM [2020])

SELECT 
	hotel,
	arrival_date_year,
	ROUND(SUM((stays_in_week_nights + stays_in_weekend_nights)*adr),2) as revenue
FROM Hotels
GROUP BY arrival_date_year, hotel

-- Check other tables
SELECT *
FROM market_segment

SELECT *
FROM meal_cost

-- Join tables to import to Power BI
WITH Hotels AS (
	SELECT * FROM [2018]
	UNION
	SELECT * FROM [2019]
	UNION
	SELECT * FROM [2020])

SELECT * FROM Hotels
LEFT JOIN market_segment ON HOTELS.market_segment = market_segment.market_segment
LEFT JOIN meal_cost ON HOTELS.meal = meal_cost.meal

SELECT * FROM meal_cost