# Debater

## Datasets
To obtain data simpy run get_data.sh or download manually from the following resources. 

**Warning! This should take a while!**

[IBM DEBATER](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml)

Debater - looks through datasets of text, finds relevant text segments, selects strongest claims, aranges them by themes, creates narrative.
Idebater datasets.
https://www.aclweb.org/anthology/C14-1141
Linear regression - factorized in sentence being a claim and sentence being connected to our topic.

A couple of scores here
MatchAtSubject
ExpandedCosineSimilarity
ESGFeatures
SubjectivityScore
Sentiment

Next - detecting boundaries inside sentence.
Language model estimates probability of it being a correct sentence.

Ranking.

https://aclweb.org/anthology/D15-1050
2 types of evidence: Expert and Anecodal.

https://www.research.ibm.com/artificial-intelligence/project-debater/how-it-works/

## Datasets detailed

https://www.aclweb.org/anthology/C18-1176.pdf
Topic- a short phrase that frames the discussion andContext Dependent Claim- a gen-eral, concise statement that directly supports or contests the given Topic (we henceforth use the termclaim instead of Context Dependent Claim).

MC - main concept, usually the wikipidea article name.

3.1    Setup and pre-processing
We follow the setup and pre-processing described in (Levy et al., 2017) – see appendix for details.  Weconsider the same train3and test sets, consisting of100and50topics respectively.  Next, we prepareda sentence–level index from the Wikipedia May 2017 dump, and used a simple Wikification tool (to bedescribed in a separate publication)4to focus our attention on sentences that mention the MC. Filteringout sentences that mention a location/person named entity using Stanford NER (Finkel et al., 2005), afterthe MC, results in an average of≈10Ksentences per MC

Очень интересно, из википедии получают автоматические вырывания клеймов из-за that предложений.
В принципе больше методология, потому что статей нет, но есть ссылки на них и клеймы.


https://www.aclweb.org/anthology/P18-2095.pdf
Опять аналогичный датасет. Тут уже есть выкачанные статьи и в экселе в них размечены предложения с клеймами.

Следующая такая же

Argument quality
Один датасет - на каждый топик (опять по вики) по 2 клейма, размечено какой лучше.
Stance Detection
Sentiment analysis

Thematic similarity
