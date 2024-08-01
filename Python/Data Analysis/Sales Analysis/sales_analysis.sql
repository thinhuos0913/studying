SELECT *
from products

SELECT * 
FROM customers C
CROSS JOIN sales S ON C.customer_id = S.customer_id 
CROSS JOIN products P ON P.product_id = S.product_id
--WHERE S.quantity > 9 

SELECT count(*)
FROM (SELECT * 
		FROM customers C
		CROSS JOIN sales S ON C.customer_id = S.customer_id 
		CROSS JOIN products P ON P.product_id = S.product_id
	)
WHERE product_name = 'Product A'
