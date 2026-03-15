#load csv or data source 

import csv 
from typing import List,Dict

def load_products(csv_path:str)->List[Dict]:
    """load product data from a csv file"""
    products=[]

    with open(csv_path,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)

        for row in reader:
            product={
                "product_name":row["product_name"].strip(),
                "about_product":row["about_product"].strip(),
                "actual_price":float(row["actual_price"]),
                "category":row["category"].strip()
            }

            products.append(product)

    return products        