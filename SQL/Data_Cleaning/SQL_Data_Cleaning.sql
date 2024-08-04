/* ETL USING SQL QUERIES */
USE Housing
GO

-- CHECK ALL DATA
SELECT *
FROM Nashville

-- 1. STANDARDIZE DATE FORMAT
SELECT SaleDate, CONVERT(DATE, SaleDate) AS ConvertedDate
FROM Nashville

ALTER TABLE Nashville
ADD ConvertedSaleDate DATE

UPDATE Nashville
SET ConvertedSaleDate = CONVERT(DATE,SaleDate)

SELECT SaleDate,ConvertedSaleDate
FROM Nashville

-- 2. FILL IN PROPERTY ADDRESS data

SELECT *
FROM Nashville
WHERE PropertyAddress IS NULL
ORDER BY ParcelID

SELECT A.ParcelID, A.PropertyAddress, 
	B.ParcelID, B.PropertyAddress,
	ISNULL(A.PropertyAddress,B.PropertyAddress)
FROM Nashville A
JOIN Nashville B
	ON A.ParcelID = B.ParcelID AND A.[UniqueID ] <> B.[UniqueID ]
WHERE A.PropertyAddress IS NULL

UPDATE A
SET PropertyAddress = ISNULL(A.PropertyAddress,B.PropertyAddress)
FROM Nashville A
JOIN Nashville B
	ON A.ParcelID = B.ParcelID AND A.[UniqueID ] <> B.[UniqueID ]
WHERE A.PropertyAddress IS NULL

SELECT PropertyAddress
FROM Nashville
WHERE PropertyAddress IS NULL

-- 3. SPLIT ADDRESS INTO COLUMNS (ADDRESS, CITY, STATE)

SELECT PropertyAddress
FROM Nashville

SELECT
	SUBSTRING(PropertyAddress, 1, CHARINDEX(',',PropertyAddress)-1) AS Address,
	SUBSTRING(PropertyAddress, CHARINDEX(',',PropertyAddress)+1, LEN(PropertyAddress)) AS City
	--CHARINDEX(',',PropertyAddress)
FROM Nashville

ALTER TABLE Nashville
ADD PropertySplitAddress nvarchar(255)

UPDATE Nashville
SET PropertySplitAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX(',',PropertyAddress)-1)

ALTER TABLE Nashville
ADD PropertySplitCity nvarchar(255)

UPDATE Nashville
SET PropertySplitCity = SUBSTRING(PropertyAddress, CHARINDEX(',',PropertyAddress)+1, LEN(PropertyAddress))

SELECT PropertySplitAddress, PropertySplitCity
FROM Nashville

SELECT OwnerAddress
FROM Nashville

SELECT 
	PARSENAME(REPLACE(OwnerAddress,',','.'),3),
	PARSENAME(REPLACE(OwnerAddress,',','.'),2),
	PARSENAME(REPLACE(OwnerAddress,',','.'),1)
FROM Nashville

ALTER TABLE Nashville
ADD OwnerSplitAddress nvarchar(255)

UPDATE Nashville
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress,',','.'),3)

ALTER TABLE Nashville
ADD OwnerSplitCity nvarchar(255)

UPDATE Nashville
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress,',','.'),2)

ALTER TABLE Nashville
ADD OwnerSplitState nvarchar(255)

UPDATE Nashville
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress,',','.'),1)

SELECT OwnerSplitAddress,OwnerSplitCity,OwnerSplitState
FROM Nashville

-- 4. CHANGE ('Y','N') TO ('YES','NO') IN [SOLD AS VACANT] COLUMN

SELECT DISTINCT SoldAsVacant,
	COUNT(SoldAsVacant) AS Counting
FROM Nashville
GROUP BY SoldAsVacant
ORDER BY 2

SELECT SoldAsVacant,
	CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
		WHEN SoldAsVacant = 'N' THEN 'No'
		ELSE SoldAsVacant
		END
FROM Nashville

UPDATE Nashville
SET SoldAsVacant = CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
		WHEN SoldAsVacant = 'N' THEN 'No'
		ELSE SoldAsVacant
		END

-- 5. REMOVE DUPLICATES

WITH ROW_NUM_CTE AS (
SELECT *,
	ROW_NUMBER() OVER (
		PARTITION BY ParcelID,
		PropertyAddress,
		SalePrice,
		SaleDate,
		LegalReference
		ORDER BY UniqueID) row_num
FROM Nashville
)

DELETE 
FROM ROW_NUM_CTE
WHERE row_num > 1
--ORDER BY PropertyAddress

WITH ROW_NUM_CTE AS (
SELECT *,
	ROW_NUMBER() OVER (
		PARTITION BY ParcelID,
		PropertyAddress,
		SalePrice,
		SaleDate,
		LegalReference
		ORDER BY UniqueID) row_num
FROM Nashville
)
SELECT *
FROM ROW_NUM_CTE
WHERE row_num > 1

-- 6. DROP UNNECESSARY COLUMMS

SELECT *
FROM Nashville

ALTER TABLE Nashville
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress

ALTER TABLE Nashville
DROP COLUMN SaleDate

