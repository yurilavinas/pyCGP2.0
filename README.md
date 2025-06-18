# pyCGP
Python implementation of Cartesian Genetic Programming in the context of classification and regression tasks
## Installation
make sure to install the dependencies with
`pip install -r requirements.txt`
## Tutorial: Evolving a CGP for Regression using PMLB Dataset
This tutorial show a simple use of pycgp for a regression problem using evolution strategy (1 + λ)

Check out the full tutorial in [Tutorial.ipynb](./Tutorial.ipynb).

You can find the list of datasets used to test this algorithm on the [Github of PMLB](https://github.com/EpistasisLab/pmlb/blob/master/pmlb/all_summary_stats.tsv)

### Interpretation of results

By running [Tutorial.ipynb](./Tutorial.ipynb) You can see:

- The detail of the best genome and the R2 score/fitness of it:
```
Best fitness achieved: 0.4417
n4 = sqrtxy(x1, x3)
n5 = acos(n4)
n6 = gt(n5, x2)
n7 = max(x2, x3)
n8 = min(x2, n6)
n9 = aminus(x3, x0)
n10 = one()
n11 = acos(n5)
n12 = asin(n6)
n13 = sqrt(n4)
n14 = abs(n12)
n15 = acos(n5)
n16 = aminus(n6, x2)
n17 = pow(n6, x3)
n18 = asin(x0)
n19 = abs(n14)
n20 = mult(n12, x1)
n21 = gt(n17, x1)
n22 = ceil(n20)
n23 = round(n7)
n24 = acos(x2)
n25 = asin(n13)
n26 = floor(n13)
n27 = min(x0, n13)
n28 = aminus(x0, n15)
n29 = sin(n17)
n30 = aminus(x0, n13)
n31 = squared(n16)
n32 = sqrtxy(x2, n17)
n33 = mult(x0, n5)
Outputs: n29, n31, n30, n20, n33, n6, n21, n23, n16

Unrolled output expression 0:
sin(pow(gt(acos(sqrtxy(x1, x3)), x2), x3))
Unrolled output expression 1:
squared(aminus(gt(acos(sqrtxy(x1, x3)), x2), x2))
Unrolled output expression 2:
aminus(x0, sqrt(sqrtxy(x1, x3)))
Unrolled output expression 3:
mult(asin(gt(acos(sqrtxy(x1, x3)), x2)), x1)
Unrolled output expression 4:
mult(x0, acos(sqrtxy(x1, x3)))
Unrolled output expression 5:
gt(acos(sqrtxy(x1, x3)), x2)
Unrolled output expression 6:
gt(pow(gt(acos(sqrtxy(x1, x3)), x2), x3), x1)
Unrolled output expression 7:
round(max(x2, x3))
Unrolled output expression 8:
aminus(gt(acos(sqrtxy(x1, x3)), x2), x2)
```

- The fitness/R2 score over the course of the evaluations

![image](https://github.com/user-attachments/assets/68a1339f-a68c-45b5-b570-364b393fc2f7)

- The best genome graph visualization

![image](https://github.com/user-attachments/assets/3ec5ae0a-fe9c-4727-865c-0b28a4b20888)

