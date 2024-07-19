# IAQF_PairsTrading2023

Background on the Problem Statement for the project:
Pairs trading or statistical arbitrage is a common trading strategy employed by hedge funds. Thecore of the strategy depends on how the prices of two assets diverge and converge over time.Pairs trading algorithms aim to profit from betting on the assumption that deviations in prices(or returns) converge to their mean. As a result, pairs trading strategies are usually based on concepts like mean reversion and stationary stochastic processes. Typically, the assets are chosen by considering either their correlation or co-integration. The investor then attempts to develop a stationary relation that can produce an alpha generating trading strategy.Relying on correlation or co-integration imposes an underlying assumption of linearity and, as we have discovered, much of the interaction of asset prices is actually non-linear. Therefore, in recent years research has begun to focus on techniques like copulas or machine learning thatallow for non-linear relationships. In addition to identifying the pairs for the strategy, it can also be important to define market events that may make departures from equilibrium pricing between the two assets permanent rather than temporary (think about LTCM).Using daily prices, take two equity indices (e.g., the S&P 500 and the Russell 2000) and develop pairs trading strategy that allows for a non-linear relationship and generates alpha.

Paper - [PaperLink](https://github.com/kohsheen1234/IAQF_PairsTrading2023/blob/main/Application%20of%20Time-Varying%20Optimal%20Copula%20and%20Mixed%20Copula%20in%20Pairs%20Trading.pdf)

Choosing pairs [Clustering](https://github.com/kohsheen1234/IAQF_PairsTrading2023/blob/main/Clustering%20and%20SVR/Clustering_new_2010.ipynb)

Test for cointegration and baseline method [Linear Approach](https://github.com/kohsheen1234/IAQF_PairsTrading2023/blob/main/Cointegration%20Test%20and%20Linear%20Approach/pairs-trading-sim.ipynb)

Non Linear approach [Coplua](https://github.com/kohsheen1234/IAQF_PairsTrading2023/blob/main/Experiment/Main%20Experiment.ipynb)

<img width="489" alt="copula" src="https://github.com/user-attachments/assets/3f642ebf-896f-4a45-a90d-491e8ae2bde1">
