*title:* New Classical and Bayesian Estimators for Classifying Trade Direction in the Absence of Quotes
*authors:* Michael Bowe, Sungjun Cho, Stuart Hyde, Iljin Sung
*year:* 2018
*tags:* #option-trade-classification #trade-classification #probabilistic-classification #bvc #markov-chain #bayesian 
*status:* #📦 
*related:*
- [[@easleyDiscerningInformationTrade2016]] (authors adapt their Bayesian view of trade classification)

## Notes📍
- Do not cite paper. It's unpublished and of doubtable quality with regards to evaluation. Also, paper is very reptitive.
- I haven't fully understood what their model is, but doesn't really matter. -> Most important are the conclusions from the BVC paper.
- They use a state space approach that uses both classical (smoothing and filtering) and Bayesian estimators (Bayesian Markov Chain) to classify trades. 
- As their data set does not include true labels, they study the correlation with the tick rule. Baed on an conclusion made in [[@easleyDiscerningInformationTrade2016]] that the tick rule is problematic in periods of high volatility with imbalances in order flow, the authors conclude that their approach is superior, due to a greater divergence from the tick rule. The authors are aware of the fact that the conclusion has to be taken with care. During normal periods their method and the tick rule are highly correlated.
- The approach is similar to the BVC method of [[@easleyDiscerningInformationTrade2016]] 
- They test their proposed trade classification method using gold futures at the CME from May - June 2016 during the time of the brexit referendum.

## Annotations 📖
“We propose new methods for estimating the effective bid-ask spread and classifying trading intentions without access to quotes.” ([Bowe et al., 2018, p. 1](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=1&annotation=8RJZJHED))

“Our state space approach utilizes both classical and Bayesian estimators” ([Bowe et al., 2018, p. 1](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=1&annotation=T9EQQZGL))

“For illustrative purposes, we apply our approach to an analysis of the trading patterns in the CME’s gold futures contract during a period incorporating uncertainty in financial markets as a result of the UK’s 2016 Brexit referendum.” ([Bowe et al., 2018, p. 1](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=1&annotation=VMKUPQMV))

“The second major contribution of this paper is to provide trade direction classification mechanism without recourse to quotes. These classification systems utilise both Bayesian MCMC methods and classical filtering and smoothing algorithms for latent trade direction indicators.” ([Bowe et al., 2018, p. 4](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=4&annotation=XL256BA7))

“Recently, Easley, Lopez de Prado, O’Hara (2016) propose a new conceptual framework for classifying trades, taking the perspective of a Bayesian statistician with priors on the unobservable information (buy or sell indicator), who is trying to extract trading intentions from observable trade data. They compare the strengths and weakness of several rules against an ideal Bayesian rule.” ([Bowe et al., 2018, p. 4](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=4&annotation=G4NXK23W))

“We propose that certain familiar structural empirical market microstructure models, such as those we employ in this analysis, provide plausible approximations to their ideal Bayesian trade classification approach. In particular, these models employ a Markov switching process as the underlying process governing the dynamics of the unobservable buy-sell indicator, and treat the measurement equations as a plausible data generating process for the observed data relating to the indicator.” ([Bowe et al., 2018, p. 4](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=4&annotation=BPB65SJL))

“For purposes of illustration, we apply our proposed approach to analyse trading behaviour in the gold futures contract trading on the CME over the two month period from May 2016 to June 2016, a timeframe incorporating the UK Brexit referendum.” ([Bowe et al., 2018, p. 4](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=4&annotation=93IMAVFX))

“However, in the presence of greater uncertainty when trading potentially generates a greater price impact (relating from to order flow imbalances), our trade classification indicator often diverges significantly from those we obtain using the Tick rule.” ([Bowe et al., 2018, p. 5](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=5&annotation=WGR8ZNWE))

“As Easley, Lopez de Prado, and O’Hara (2016) maintain that Tick rule classifications appear particularly problematic in periods of high volatility exhibiting imbalances in order flow, we believe the approach to trade classification we propose shows some promise.” ([Bowe et al., 2018, p. 5](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=5&annotation=BWHS7KBN))

“As Easley, Lopez de Prado, and O’Hara (2016) note, each trade classification rule may demonstrate both strengths and weakness, depending on the underlying market characteristics.” ([Bowe et al., 2018, p. 5](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=5&annotation=M6BNM6PI))

“They adopt the perspective of Bayesian statisticians with priors on the unobservable information (here t q ), who are trying to extract trading intentions from observable trading data. Ideally, we would like to specify the data generating processes for both the underlying unobservable variables and subsequently for the observed data, conditional on the realizations of the underlying unobservable data.” ([Bowe et al., 2018, p. 14](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=14&annotation=7P4ANNPN))

“They claim that every trade classification algorithm can be regarded as an approximation to this Bayesian approach, and that their bulk volume classification (BVC) methodology is conceptually closer to this ideal than traditional approaches such as the Tick rule, since BVC assigns a probability to a given trade being either a buy or sell.” ([Bowe et al., 2018, p. 14](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=14&annotation=P4B8ZSP2))

“We conduct the empirical implementation of our proposed trade classification methods using data from gold futures trading on the Chicago Mercantile Exchange (CME) during May and June 2016.” ([Bowe et al., 2018, p. 15](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=15&annotation=XFZS59RF))

“Specifically, we select our sample data from the gold futures contract trading on CME’s Globex electronic trading platform during the period from May 1, 2016 to June 30, 2016.” ([Bowe et al., 2018, p. 15](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=15&annotation=TMMIH3QY))

“In order to provide appropriate benchmarks with which to compare our results on the classified trades, we proceed to classify trades using the standard Tick rule19 and generate daily correlation estimates of classified trades using the Tick rule and our model consistent rules” ([Bowe et al., 2018, p. 25](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=25&annotation=SXNKCXBB))

“We conjecture a potential explanation for the second finding is as follows. Easley, Lopez de Prado, O’Hara (2016) maintain that when the underlying data is less noisy, Tick rule classifications can be superior to other rules. However, they also show that in situations where underlying data noise is substantial or order flow is imbalanced, such as when private information motivates trading, trade classifications using the Tick rule may be unreliable.” ([Bowe et al., 2018, p. 26](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=26&annotation=UKRUT7MD))

“In summary, the model consistent trade direction classification algorithm based on the extended GH formulation generates very similar results to the Tick rule during normal trading periods, but in periods characterised by higher uncertainty and the existence of a potentially larger price impact of trades (closely related to order imbalances), the classifications obtained from the two methods diverge significantly. As these are precisely the circumstances under which Easley, Lopez de Prado, and O’Hara (2016) argue that the Tick rule appears mos” ([Bowe et al., 2018, p. 26](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=26&annotation=NFHERS2A))

“27 problematic in classifying trades, this suggest our proposed extended GH methods may be useful in such an environment.” ([Bowe et al., 2018, p. 27](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=27&annotation=7U748S7F))

“However, in the presence of greater uncertainty when trading potentially generates a greater price impact (resulting from order flow imbalances), our trade classification indicator often diverges significantly from those using the Tick rule. Easley, Lopez de Prado, and O’Hara (2016) maintain that Tick rule classifications appear particularly problematic in periods of high volatility exhibiting imbalances in order flow.” ([Bowe et al., 2018, p. 30](zotero://select/library/items/74N2TUYU)) ([pdf](zotero://open-pdf/library/items/UBVF223Y?page=30&annotation=GNQL7AZW))