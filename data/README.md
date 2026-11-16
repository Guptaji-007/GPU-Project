# Data

Sparse matrices used to test and benchmark our SpMV and SpMM kernels.
They cover different sparsity patterns (regular FEM meshes, uniform road networks, irregular web graphs) so we can see which format and kernel works best for which kind of matrix.

| File | Source | Type | Rows | Non-zeros |
|---|---|---|---|---|
| bcsstk01.mtx | NIST Matrix Market | small structural matrix, for testing | 48 | 400 |
| nasasrb.mtx | Network Repository | FEM (shuttle rocket booster) | 54,870 | 2,622,454 |
| shipsec1.mtx | Network Repository | FEM (ship section) | 140,385 | 3,415,518 |
| pwtk.mtx | Network Repository | FEM (wind tunnel) | 217,891 | 11,306,442 |
| roadNet-CA.mtx | SNAP | California road network | 1,971,281 | 5,533,214 |
| web-Google.mtx | SNAP | Google web graph | 916,428 | 5,105,039 |
| wiki-Talk.mtx | SNAP | Wikipedia talk network | 2,394,385 | 5,021,410 |



## Citation

- R. F. Boisvert, R. Pozo, K. Remington, R. Barrett, J. Dongarra. "Matrix Market: a web resource for test matrix collections." Quality of Numerical Software, 1997.
- R. A. Rossi, N. K. Ahmed. "The Network Data Repository with Interactive Graph Analytics and Visualization." AAAI, 2015. https://networkrepository.com
- T. A. Davis, Y. Hu. "The University of Florida Sparse Matrix Collection." ACM Transactions on Mathematical Software 38(1), 2011. (original source of nasasrb, shipsec1, pwtk)
- J. Leskovec, A. Krevl. "SNAP Datasets: Stanford Large Network Dataset Collection." http://snap.stanford.edu/data, 2014.
