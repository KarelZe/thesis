We only retain stocks with at least 10 call (put) option contracts per day and exclude options in the highest effective spread percentile to avoid biases by illiquid option trading

## Setup
Albeit the classification accuracy is a reasonable measure for comparing classifiers, one cannot immediately infer how changes in accuracy e. g., an improvement by $1{}\%$, affect the application domains. In an attempt to make our results tangible, we apply all algorithms to estimate trading cost, a problem we previously identified to be reliant on correct trade classification (cp. [[👶Introduction]]) and a common testing ground for trade classification rules (cp. [[@ellisAccuracyTradeClassification2000]]541) and ([[@finucaneDirectTestMethods2000]]569)  and ([[@petersonEvaluationBiasesExecution2003]]271--278) and ([[@savickasInferringDirectionOption2003]]896--897).

One of the most widely adopted measures for trading costs is the effective spread ([[@Piwowar_2006]]112). It is defined as the difference between the trade price and the fundamental value of the asset ([[@bessembinderIssuesAssessingTrade2003]]238--239).  Following ([[@bessembinderIssuesAssessingTrade2003]]238--239), we define the *nominal, effective spread* as 
$$
S_{i,t} = 2 (P_{i,t} - V_{i,t}) D_{i,t}.
$$
Like before, $i$ indexes the security and $t$ denotes the trade. Here, $D_{i,t}$ is the trade direction, which is either $1$ for customer buy orders and $-1$ for customer sell orders. If the trade initiator is known, we set $D_{i,t} = y_{i,t}$ and $D_{i,t}=\hat{y}_{it}$, if inferred from a rule or classifier. As the fundamental value $V_{i,t}$ is unobserved at the time of the trade, we follow a common track in research and use the midpoint of the prevailing quotes as an observable proxy. footnote-(For an alternative treatment for options (cp.[[@muravyevOptionsTradingCosts2020]]4975--4976). Our focus is on the midspread, as it is the most common proxy for the value.) This is also a natural choice, assuming that, on average, the spread is symmetrical and centred around the true fundamental value ([[@leeMarketIntegrationPrice1993]]1018). ~~~([[@hagstromerBiasEffectiveBidask2021]]317) reasons that the appeal of using the midpoint lies in the high data availability, simplicity, and applicability in an online setting.~~ We multiply the so-obtained half-spread by $2$ to obtain the effective spread, which represents the cost for a round trip trade involving a buy and sell ex commissions.

From cref-eq-effective-spread its easy to see, that a classifier correctly classifying every trade, achieves an effective spread estimate equal to the true spread. For a random classifier, the effective spread is around zero, as missclassification estimates the spread with opposite sign, which offsets with correct, random estimates for other trades.

10 The accuracy of trade direction estimation is not important for estimating effective spreads when trades are executed at quote midpoints at which points effective spreads are zero. (found in [[@chakrabartyTradeClassificationAlgorithms2007]])

Readily apparent from (cref-eq), poor estimates for the predicted trade direction, lead to an under or over-estimated effective spread, and hence to a skewed trade cost estimate. By comparing the true effective spread from the estimated, we can derive the economical significance. For convenience, we also calculate the *relative effective spread* as 
$$
{PS}_{i,t} = S_{i,t} / V_{i,t}.
$$
The subsequent section estimates both the nominal and relative effective spread for our test sets.

## Results
The actual and the estimated effective spreads, as well as the quoted spread, are shown in the (cref tab) aggregated by mean.  ([[@savickasInferringDirectionOption2003]] 896--897) previously estimated the effective spreads on a subset of rules for option trades at the gls-CBOE, which can be compared against.

Following ([[@theissenTestAccuracyLee2000]] 12) a Wilcoxon-test is used, to test if the medians of the estimated, effective spread and the true effective spread are equal. The null hypothesis of equal medians is rejected for $p\leq0.01$. Alternatively, formulate with confidence level of 1 %.

In summary, quote-based algorithms like the quote rule and the gls-LR algorithm severely overestimate the effective spread. The overestimate is less severe for the gls-clnv algorithm due to stronger dependency on the tick rule. The tick rule itself, achieves estimates closest to the true effective spread, which is num-() and num-() for the gls-ise and gls-cboe sample respectively. As primarily tick-based algorithms, like the tick rule or emo, perform like a random classifier in our samples, we conclude that the close estimate are an artefact to randomness, not due to superior predictive power. This observation is in line with ([[@savickasInferringDirectionOption2003]]897), who make a similar point for the gls-emo rule on gls-cboe trades. For rule-based algorithms $\operatorname{gsu}_{\mathrm{large}}$ provides reasonable estimates of the effective spread, while achieving high classification accuracy. From our machine learning-based classifiers the FT-Transformer or gls-GBRT trained on FS3 provides close estimates of the true effective spread, in particular on the gsl-CBOE sample. The null hypothesis of equal medians is rejected.

Based on these results, we conclude, that  $\operatorname{gsu}_{\mathrm{large}}$ provides the best estimate of the effective spread, if the true label is absent. For labelled data, Transformer or gradient boosting-based approaches can provide even better estimates. In turn, the de facto standard, gls-LR algorithm, might bias research.

**Other literature:**
Similarily in [[@chakrabartyTradeClassificationAlgorithms2007]]. Alternatively compare correlations $\rho$ and medians using the Wilcoxon test with null hypothesis of the equal medians with $p=0.01$ (cp.[[@theissenTestAccuracyLee2000]]12).

However, the requirements e. g., independence of samples are much higher for the $t$-test. Thus, I chose the Wilcoxon test instead. See e. g., [here.](https://www.methodenberatung.uzh.ch/de/datenanalyse_spss/unterschiede/zentral/wilkoxon.html#:~:text=Stichproben%20verschieden%20sind.-,Der%20Wilcoxon%2DTest%20wird%20verwendet%2C%20wenn%20die%20Voraussetzungen%20f%C3%BCr%20einen,anderen%20Stichprobe%20sich%20gegenseitig%20beeinflussen.)

The null hypothesis is that the location of medians in two independent samples are same.
(🔥What can we see? How do the results compare?)


- “During our sample period of 2004–2015, quoted half-spreads of options on stocks in the S&P 500 index averaged 13 cents per share and 8.6% of the option price. Dollar (percentage) spreads were considerably wider for well in-the-money (out-of-the-money) options.” (Muravyev and Pearson, 2020, p. 4973)
- “Although the costs of options market making can help explain why options spreads should be higher than the spreads of their underlying stocks (Battalio and Schultz 2011), a second puzzle is that existing theories are unable to explain the observed patterns of spreads. For example, the high dollar spreads of inthe-money (ITM) options and the relation between spreads and moneyness cannot be explained by hedge rebalancing costs incurred by options market makers, because hedges of well ITM options rarely need to be rebalanced. Similarly, the pattern cannot be explained by difficult to hedge gamma and vega risks that options market makers bear when they hold inventories of options, because well ITM options are not exposed to these risks.” (Muravyev and Pearson, 2020, p. 4974)

## Inside / Outside / At the Quote
- “Options traders exploit this predictability in timing their executions. Executions at the ask price tend to occur when the estimate of the fair value (the expected future midpoint) is close to but less than the quoted ask price, and executions at the bid price tend to occur when it is close to but greater than the quoted bid price. Traders who exploit this predictability are able to take liquidity at low costs, as we explain next.” (Muravyev and Pearson, 2020, p. 4975)
- “Why do option market makers not update quotes frequently? Even if liquidity providers are faster than most liquidity takers, if they are slower than only one they are at risk to get picked off.4 To protect against this risk, market-makers post wider spreads that do not have to be changed with every change in the option fair value.5 Foucault, Roell, and Sandas (2003) model the trade-off that dealers face between the cost of frequent quote revisions and the benefits of being picked off less frequently.6 It is also costly for option market makers to update quotes frequently because the options exchanges place caps on the number of quote updates and fine exchange members whose ratios of messages to executions is large. In addition, market frictions, such as minimum tick sizes, prevent market makers from continuously centering their quotes on the fair value. Finally, trades by execution timers incur a half-spread of about three cents, which exceeds market-makers’ marginal costs of executing trades. Thus, nontimers’ trades are highly profitable for market makers, while the spreads on timers’ trades appear to at least cover market makers’ marginal costs of trading. Thus, market makers can facilitate trading by cost sensitive investors by changing their quotes infrequently.” (Muravyev and Pearson, 2020, p. 4977)
- “During our sample the overwhelming bulk of option trading was electronic, with market makers generally using auto-quoting algorithms and quotes and trades disseminated almost instantly to participants in both the option and equity markets. In contrast to the previous option market structure in which trading occurred on exchange floors, in the current market structure an option market maker on the exchange where trade occurs does not have any informational advantage relative to other market participants, including market makers on the equity exchanges. This helps explain our findings that option quotes do not contain information not already reflected in stock quotes.” (Muravyev et al., 2013, p. 261)


“We repeated this analysis with our dataset from the Frankfurt Stock Exchange. The results are presented in columns 2 and 3 of Table 5. The bias is even more dramatic. The traditional spread estimate is, on average, about twice as large as the “true” spread.8 A Wilcoxon test rejects the null hypothesis of equal medians (p < 0.01). Despite the large differences, the correlation between the two spread estimates is very high (ρ= 0.96). The magnitude of the relative bias (i.e., the traditional spread estimate divided by the “true” spread) is strongly negatively related to the classification accuracy. The correlation is –0.84.” ([[@theissenTestAccuracyLee2000]], p. 12)

"Table 6 Panel A shows that our algorithm provides the best estimate of effective spread. We conduct a t-test for difference in means to assess whether the effective spread of each algorithm is statistically significantly different from actual effective spread. Results indicate that the effective spread provided by our algorithm is a statistically significant unbiased estimate of the actual effective spread while the LR (EMO) rule provides upwardly (downwardly) biased estimates. Table 6 Panel B shows that the other algorithms provide biased estimates while our alternative algorithm provides statistically insignificant difference from the actual price impact. The results show that errors in trade side classification can result in substantial biased price impacts. The underestimations of price impacts are 5.26%, 29.47%, and 44.21% for the LR, EMO, and tick rule, respectively." ([[@chakrabartyTradeClassificationAlgorithms2007]] 3820)


Thus, our results show, that . If accurate

Our results match theirs in magnitude.

Results indicate that the effective spread is best estimated $\operatorname{gsu}_{\mathrm{large}}$ or the 

If every trade is misclassified, the effective spread is similar in magnitude but with opposite sign.
\todo{write what is problematic acout the tick rule (random guess)}

\todo{Similar magnitude to \textcite{savickasInferringDirectionOption2003}}

\todo{mean! of dollar spread and relative spread, no filters / implicitly non-negative spread}

Table 6 Panel A shows that our algorithm provides the best estimate of effective spread. We conduct a t-test for difference in means to assess whether the effective spread of each algorithm is statistically significantly different from actual effective spread. Results indicate that the effective spread provided by our algorithm is a statistically significant unbiased estimate of the actual effective spread while the LR (EMO) rule provides upwardly (downwardly) biased estimates. Table 6 Panel B shows that the other algorithms provide biased estimates while our alternative algorithm provides statistically insignificant difference from the actual price impact. The results show that errors in trade side classification can result in substantial biased price impacts. The underestimations of price impacts are 5.26%, 29.47%, and 44.21% for the LR, EMO, and tick rule, respectively.

The results are in Table 5. All four methods perform poorly at estimating effective bid-ask spread for options. The quote rule overestimates effective spread: the estimate is close to the quoted spread. This is a direct consequence of the fact that the quote method fails to recognize the existence of RQ trades. The tick rule severely underestimates effective spread. This is a consequence of the method’s classifying correctly just slightly more than half of all trades. h t t p s : / / d o i . o r g / 1 0 . 2 3 0 7 / 4 1 2 6 7 4 7 P u b l i s h e d o n l i n e b y C a m b r i d g e U n i v e r s i t y P r e s

Savickas and Wilson 897 TABLE 5 Estimated Effective Spreads Average Sprd. Quote Rule LR (1991) EMO (2000) Tick Rule Actual Sprd. Quoted Sprd. Dollar 0.2339 0.2163 0.1797 0.0637 0.1448 0.2444 Relative 0.1182 0.1094 0.1001 0.0290 0.0785 0.1393 Dollar effective spreads are calculated as Si  2Pt i  Pm iI, where Pm i is the midspread from quotes outstanding at the time of trade i; Pt i is the trade i option price; and I  1iftradei is a buy and I  1 if it is a sell. The relative spread is computed as PSi  SiPm i. Specifically, if a rule misclassified 100% of all trades, the estimated effective bidask spread would be negative but equal in absolute value to the actual effective spread. If a rule misclassified 50% of all trades, its estimated effective spread would be close to zero. Because the tick rule classifies correctly about 60% of all trades, its estimated spread is slightly greater than zero. As mentioned previously, the poor performance of the tick rule is a consequence of the fact that only 59.7% (58.7%) of all option buys (sells) occur on an uptick (downtick). The quote and the tick rules are the two extremes. The LR and EMO methods take their respective places on the continuum between the quote and the tick rules. The EMO approach uses the quote rule to a lesser extent than does the LR algorithm; therefore, the EMO method exhibits a lower degree of spread overestimation than does the LR method.

% TODO: Make explanation more detailled? See e. g., https://s3.eu-central-1.amazonaws.com/up.raindrop.io/raindrop/files/526/059/341/MAS_Thesis_Mate_Nemes_final_Jan13.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAZWICFKR63DOESPJN%2F20230227%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20230227T054628Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIGh8t4%2BwrZPS53a0v4pmuMrfdSK0byfd6kOZymIodKHYAiEAmLEgZaZOGQnHox0E%2FKUKmzaSoLDsg%2FDOp3wOwqivzasq%2BAMIUxAAGgw2NjYyNjEzNDU0MDUiDER%2BrooXF33LN6%2FviirVA7XUNhcQEGET%2BR%2FFF2TxkIwpnMbB3sUSVe%2B37iVeKehJPZI4PlXs%2BvOrAyydlqpoERgBIRSbjH3otqvC8sSikw%2FOX5bg9RIQNWV988GpA4FiEwUYkck9d46o9X3GZ2ZHHHis7h5ADmBMsEUpRx5P3DkLXxOig7a6tb5%2BNDNExpPcTKaJOLL1CeM4g6dg9czBkZ1mHC6SCpTjQRBym0jndXXAFRpxrnLOG7lRzC1til7wdX9yVe7m8YgAixkTtrN0oZ81%2BunfpqVTs9dQ%2FeaaDkGwUMpdh8PKoG3V8aIKBclaaix%2BXCwkHA%2Fcq%2BbHRY7HDm4eAiZ1leUxvJfX1rx6GR8uP978qrSs3nZvep5aTi3CjeLX1fna%2FX2sE3VZr4xT8cy9vkq%2FpvQIPJ%2BhnYi1v%2BAq6y7g4RjTJmHHEm0nPTvuV0Lk1%2BSIFzYEsh1my8BKHKrr0WEHWYVcPsBNaP37a6fxzxwqJLSEMtEM7Zb%2BDNewrYEdavtiVSQGLob9LfFF3Bobc%2BhCs8xrunkNbJfMsCVBnDGPKFDRKHznBuuZu6SqXaj8bvxu1Q0YMxkImt%2Bi72zRGqUSSSLlKC9MCbbGlBt5b%2FkfCQrzFbL6PHiuZes10hwNoz4wwJTwnwY6pQGavPb2vsjGrOUbaRKmC9iQY5uvJZpHfQMgXMoTWVx58m6eU%2FotzkeDnwSVmZDIA4yu4%2B%2BKoScCFTXEDkULo8NJBITW0kX1zMG7U0sOdC%2B7TfT8VK7%2FsqDC7MjrNCCDvUxcpmCcddA2eR%2BKEn114AG9ZhNewTdGfIu4zV2w%2Bpa1lwalqqQkM5E9zKeI9mENGGtVMEcptjXJcl30O2%2BnkFkuJAxaGa8%3D&X-Amz-Signature=4f206295f4f0b090df3d32382188400b305c79ab9a45a18bbbc79e7824236dc7&X-Amz-SignedHeaders=host

% TODO: read: Glosten, L. and Harris, L. (1988). Estimating the components of the bid/ask spread.Journal of financial Economics, 21(1):123–142.

% TODO: read: Ho, T. and Stoll, H. (1981). Optimal dealer pricing under transactions and return uncertainty. Journal of Financial economics, 9(1):47–73.

% TODO: read: Huang, R. and Stoll, H. (1997). The components of the bid-ask spread: a general approach. Review of Financial Studies, 10(4):995–1034.

% TODO: read: Roll, R. (2012). A simple implicit measure of the effective bid-ask spread in an efficient market. The Journal of Finance, 39(4):1127–1139.

% TODO: read: Petrella, G. (2006). Option bid-ask spread and scalping risk: Evidence from a covered warrants market. Journal of Futures Markets, 26(9):843–867.

% TODO: read: Pinder, S. (2003). An empirical examination of the impact of market microstructure changes on the determinants of option bid–ask spreads. International Review of Financial Analysis, 12(5):563–577.


“In addition, my results offer little help in answering why option bid-ask spreads are so large. This is one of the biggest puzzles in the options literature—existing theories of the option spread fail to explain its magnitude and shape (Muravyev and Pearson (2014)).” (Muravyev, 2016, p. 696)

- [[@rosenthalModelingTradeDirection2012]] lists fields where trade classification is used and what the impact of wrongly classified trades is.
- The extent to which inaccurate trade classification biases empirical research depends on whether misclassifications occur randomly or systematically [[@theissenTestAccuracyLee2000]].

**Notes:**
[[🍕Application study notes]]