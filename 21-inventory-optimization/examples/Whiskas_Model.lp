\* The_Whiskas_Problem *\
Minimize
Total_cost_of_ingredients_per_can: 0.008 Beef_Percent + 0.013 Chicken_Percent
Subject To
Fat_Req.: 0.1 Beef_Percent + 0.08 Chicken_Percent >= 6
Fibre_Req.: 0.005 Beef_Percent + 0.001 Chicken_Percent <= 2
Percentages_Sum: Beef_Percent + Chicken_Percent = 100
Protein_Req.: 0.2 Beef_Percent + 0.1 Chicken_Percent >= 8
Salt_Req.: 0.005 Beef_Percent + 0.002 Chicken_Percent <= 0.4
Bounds
 0 <= Chicken_Percent
Generals
Chicken_Percent
End
