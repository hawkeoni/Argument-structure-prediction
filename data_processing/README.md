# Datasets
Table of datasets :heavy_check_mark: :heavy_multiplication_x: 
| Dataset       | Topic         | Claim Sent    | Claim         | Evidence Sent | Evidence      |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Content Cell  | Content Cell  |               |               |               |               |
| Content Cell  | Content Cell  |               |               |               |               |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
## Processed
# Old
## IBM
Подстрока -AD- означает название задачи argument detection. 

### Корпус IBM-AD-ACL2014
http://acl2014.org/acl2014/W14-21/pdf/W14-2109.pdf
Первая работа по argument detection.

Терминология следующая:
Topic – a short, usually controversial statement
that defines the subject of interest. 

Context Dependent Claim (CDC) – a general concise
statement that directly supports or contests the
Topic.

Context Dependent Evidence (CDE) – a
text segment that directly supports a CDC in the
context of a given Topic

Весь CDE разделен на 3 типа:

Study: Results of a quantitative
analysis of data given as numbers or as conclusions. 

Expert: Testimony by a person / group /
committee / organization with some known expertise in or authority on the topic. 

Anecdotal: a
description of specific event(s)/instance(s) or
concrete example(s).

По числам заявляется следующее: 33 debate motions (motion == topic только в данной статье), 1,392 CDC (на самом деле 1387), 1,291 CDE с классами 3х типов (тут все честно). Помимо этого есть и сами тексты статей, с пустыми предложениями (без cdc или cde.)


Ровно на этом учился Margot в бинарной классификации предложений для поиска claim, и на этом я проводил эксперимент.


Пример для понимания:
```
Topic:
the sale of violent video games to minors 

Claim:
exposure to violent video games causes at least a temporary increase in aggression and that this exposure correlates with aggression in the real world

CDE
The most recent large scale meta-anlysis-- examining 130 studies with over 130,000 subjects worldwide-- concluded that exposure to violent video games causes both short term and long term aggression in players

evidence type: study (у некоторых evidence 2 типа)
```

Описаны критерии для разметчиков:
In practice, the labelers were asked to label
a text fragment as a CDC if and only if it complies with all the following five criteria:
* Strength – Strong content that directly supports/contests the Topic.
* Generality – General content that deals with a relatively broad idea.
* Phrasing – The labeled fragment should make a grammatically correct and semantically coherent
statement.
* Keeping text spirit – Keeps the spirit of the original text.
* Topic unity – Deals with one topic, or at most two related topics.

https://www.aclweb.org/anthology/C14-1141.pdf - в этой статье описывается бейзлайн система по поиску клеймов, но интереса кроме этого не представляет.


В итоге получается предложен небольшой датасет, на котором есть 4 задачи: 
1) По топику искать предложение с клеймом
2) По результатам 1 уточнять границы клейма (клейм = непрервный кусок предложения)
3) По (топику и) клейму искать evidence
4) Аналогично 2 искать границы evidence (можно еще определять его класс)


### Корпус IBM-AD-EMNLP2015
https://www.research.ibm.com/haifa/dept/vst/papers/Evidence2015.pdf

Вкратце - расширение предыдущего датасета, что совсем говоря неочевидно из текста статьи. Все остается прежним, кроме размеров:
Топиков стало 59, клеймов стало 2294, evidence стало 4691 (у них остались классы). Эти числа с учетом старых, т.е. датасет увеличился более чем в 2 раза, в основном по evidence. Главная его проблема - текст в финальных claim & evidence зачем-то отформатирован, убраны некоторые речевые обороты из-за чего около половины claims & evidence сложно найти в исходных текстах статей, но это решаемо.


### не корпус, а методика levy2017
https://www.aclweb.org/anthology/W17-5110.pdf
Решили получить claims. Пошли на сайт idebate.com, взяли 431 topic, в каждом топике выделили главный объект - main concept. 
Стоит обратить внимание на этот термин, потому что дальше он будет использоваться везде. К примеру для темы "запрет видеоигр" main concept = "видеоигры".

Искали предложения вида: слово маркер "that" -> Main concept -> оставшееся предложение.
Пример из статьи: topic - End affirmative action, main concept - affirmative action.
Предложение: Opponents claim that [affirmative action has undesirable side-effects and that it fails to achieve its goals.]
В квадратных скобках claim. 
Сам полученный корпус не выложен, но это и неважно, потому что дальше будет похожий корпус размером 1.5 миллиона предложений.


Вообще интересная методика для слаборазмеченных данных, о которых речь далее.

### корпус webis-debate-16
https://www.aclweb.org/anthology/N16-1165.pdf
Не IBM, но без него непонятна следующая работа. 
16.400 предложений, 66% положительных. Взяты из сайта idebate.
Идея следующая: даны рассуждения на всякие темы, каждый высказывающийся вначале немного описывает тему, 
а потом начиает рассуждения, поэтому пусть начало каждого текста
является non argumentative, а оставшаяся часть argumentative.

По сути в корпусе ни claim, ни evidence, а какое-то среднее понятие "аргументативного предложения", 
что и отмечается в следующей статье.



### Корпус IBM-AD-ACL2018
https://www.aclweb.org/anthology/P18-2095.pdf
Исследуется проблема смешения weak labeled data (wld) и strong labeled data (SLD).

Данные с debatepedia (SLD, получены разметчиками, доступны нам): 5700 пар topic, evidence (NB, об этом дальше), из которых 2200 являются положительными примерами.
WLD - корпус, состоящих из двух предыдущих работ. Из работы levy2017 получилось 250.000 предложений из википедии (25% положительных).
Предлагается обучаться одновременно на SLD и WLD постепенно уменьшая долю WLD.

Нетрудно заметить то, что в этой работе ломаются базовые определения: если классифицировать claim по Topic укладывается в парадигму IBM, то классифицировать evidence по topic раньше было нельзя. Да теперь это не context dependent evidence, а просто evidence, но все равно теряется униформность данных.

Что еще странно, работа levy2017 выделяет не evidence, а claim. По итогу есть отсюда SlD корпус, но обстоятельства его появления крайне смутные.
В итоге получилось, что тренировали бинарную классификацию evidence по топик + бинарную классификацию всякого мусора. Да, при очень аккуратном использовании WLD результаты на 1-2% accuracy лучше, но вообще сама работа лично у меня вызывает недоумение.


### Корпус IBM-AD-COLING2018
https://www.aclweb.org/anthology/C18-1176.pdf
Если предыдущая работа показалась Вам странной, то эту лучше пропустить. На самом деле нет, потому что несмотря на крайне странную задачу, результаты на других корпусах вышли ненулевые.

Берется методика из работы levy2017 и создаются 2 датасета на основе википедии 2017 года.

Первый датасет: берутся предложения в которых содержится main concept. Все они делятся на 2 типа: в одних перед main concept есть слово "that", в других его нет. Считается, что та часть, в которой слово "that" есть более богата на предложения с claim (claim sentence). Хочется бинарно уметь различать эти 2 класса предложений. Задача звучит странно - предложения отлично разделяются по наличию слова "that", но исследователи из IBM не растерялись: каждое предложение имеет структуру [prefix -> (that) Main concept -> suffix], поэтому обучение решено было проводить только на суффиксах, т.е. на части следующей непосредственно за main concept.

Второй датасет тоже очень интересный: опять берутся предложения с main concept; создается claim lexicon (его можно восстановить из корпуса) слов, которые должны идти после main concept, чтобы предложение предположительно содержало claim. Примеры этих слов: should, tuned, could, unconstitutional, violate, might, violated .... и т.д.
Второй корпус разделяется аналогично: если есть паттерн вида main concept + слово из claim lexicon - это положительный класс (claim sentence), иначе отрицательный. Т.к. эти множества опять легко разделимы обучаться решили на prefix, т.е. части предложения, идущей до main concept.
В сумме таких предложений из 2 датасетов 1.5 миллиона, они доступны.


После этого для 50 топиков из WLD 50 топ предложений предсказанных нейросетью было размечено людьми. Получилось 50*50=2500 предложений, из которых в итоге разметчиками 733 считаются по-настоящему положительным классом. Еще зачем-то даны выходы нейросети. Этот корпус тоже доступен.

Замеры полученных моделей провели на корпусах UKP, вышло число 0.66 f1.

Эту методику можно попробовать и для русского языка.


### Корпус IBM-COLING-AAAI2020
Новые определения:
We define a Motion as a high-level claim implying some clearly positive or negative stance towards a (debate’s) 
topic, and, optionally some
policy or action that should be taken as a result. For example, a motion can be We should ban the sale of violent video
games or Capitalism brings more harm than good. In the
first case, the topic is violent video games, and the proposed
action is ban. In the second case, the topic is Capitalism, and
no action is proposed. In this work, the topic of motions will
always be a reference to a Wikipedia article.

In the context of a motion, we define Evidence as a single
sentence that clearly supports or contests the motion, yet is
not merely a belief or a claim. Rather, it provides an indication whether a belief or a claim is true.

Ищут evidence по motion (motion == claim == topic, все смешалось).
Берутся все работы, указанные выше. 
На них обучается вначале логрегрессия, потом нейросети. 
После каждого этапа постепенно моделями предсказываются классы предложений на Very Large Corpus 
(VLC, корпус новостей LexisNexis из 400 миллионов статей и новостей). Наиболее вероятные по меткам нейросети предложения
доразмечаются людьми (active learning). В итоге получается Very Large Dataset из 198.457 предложений, размеченных людьми. 
Из них 33% положительных. Этого корпуса нет, но можно попробовать запросить, если нужно.
Доступен однако аналогичный корпус, собранный ими по википедии - 29.429 примеров, 23% положительных.


### Корпуса речей
Есть еще 4 корпуса речей с дебатами. Есть как звук, так и тексты, полученные машиной, тексты, полученные человеком, + 
какая-то разметка по аргументам.


### Honourable mentions
У IBM есть еще корпуса по тематической кластеризации предложений, по Concept Controversiality, по Mention Detection 
(как некую упомянутую сущность сопоставить чем-то в knowledge base) и по Semantic Relatedness между текстами.

Еще есть expressive text to speech - как лучше произнести текст, чтобы он был убедительным.

Есть датасет топиков для бинарной классификации - является ли один топик хорошим расширением другого. 

Еще есть несколько датасетов по argument quality: в одном каждому evidence присвоен скор от 0 до 1, в другом даны пары
evidence, нужно определить, какой лучше.

Есть датасеты (3 штуки) для claim sentiment (claim polarity/claim stance), т.е. claim за или против topic.

Еще есть датасет со мнениями каких-то социальных групп за/против определенных topics, но выглядит очень обскурно,
если потребуется изучу.




### toulumin model

## Используемые датасеты
Датасеты, используемые для transfer learning. 

### Scitdb
Article - https://www.aclweb.org/anthology/W19-4505.pdf
Dataset link - http://scientmin.taln.upf.edu/argmin/scidtb_argmin_annotations.tgz.
Description: 327 sentences, 8012 tokens, 862 discourse units and 352 argumentative
units linked by 292 argumentative relations taken from 60 article abstracts. 


60 введений из научных статей классифицрованы на следующие классы:
* proposal (problem or approach)
* assertion (conclusion or known fact)
* result (interpretation of data)
* observation (data)
* means (implementation)
* description (definitions/other information)
Все эти компоненты соединены в различные отношения (detail, support, sequence), но самое важное из них - support.
По итогу: proposal != claim, но все, что имеет тэг support очень похоже на evidence, поэтому в таком виде датасет и 
конвертирован. Предложения с support считаются  классом Evidence, остальные - просто предожениями.
Валидация проводится на 10 фолдах.


### DrInventor
Link - http://taln.upf.edu/drinventor/resources.php
Разобрано 40 статей по компьютерному зрению. К каждой статье содержится несколько файлов: 
3 пересказа (абстрактивная саммаризация), файл для экстрактивной саммаризации (дан скор каждого предложения), 
файл с цитатами (для каждой цитаты написана ее цель CRITICISM, NEUTRAL, COMPARISON, etc.), 
файл с аннотацией класса каждого предложения в scientific discourse (APPROACH, BACKGROUND, CHALLENGE), 
файл с subjective statements (Advantage, disadvantage, novelty, common_practice).
К датасету приложена pdf с пояснениями о разметке и о всех файлах к каждой статье.

Было решено взять за положительный класс предложения с тэгами Advantage, Disadvantage и не нейтральные цитаты. 
Все остальные предложения считаются не содержащами аргументативных компонент. В итоге получилось 10401 sentences, 
1709 evidence sentences




### Internet Argument Corpus v2
Link - https://nlds.soe.ucsc.edu/iac2
Article link - http://www.lrec-conf.org/proceedings/lrec2016/pdf/1126_Paper.pdf 
A collection of corpora for research in political debate on internet forums. 
It consists of three datasets: 4forums (414K posts), ConvinceMe (65K posts), and a sample from CreateDebate 
(3K posts). It includes topic annotations, response characterizations (4forums), and stance.

Дан вопрос и ответ, для ответа сказано, насколько он disagree/agree, attacking/respectful, emotional/fact, sarcasm;
Для каждой пары вопрос ответ известно, кто "за", а кто "против".

### Araucariadb
Link - http://corpora.aifdb.org.
662 примера из разных источников. Куски текстов соединены в графы отношений, где одно опровергает другое, третье подтверждает второе и т.п. 662 примера, в среднем 637 символа на текст (119 слов), 8.85 компонент аргументации и 7.88 ребер, соединяющих компоненты.
Скорее мертвый, чем живой проект. Данные удалось скачать только AraucariaDB и только в формате zip, иначе получается битый архив. Содержит предложения, связанные некими связями. По сути предложения выстроены в направленый граф, где каждое ребро означает либо атаку, либо поддержку. В данных отсутствуют отрицательные примеры, по сути все там является и claim и evidence. Очень непонятная разметка, все ребра имеют пустой класс, а предложения имеют неговорящие классы 'I', 'CA', 'YA', 'RA', 'L'.





# Датасеты, которые не подошли
Датасеты отсюда нельзя напрямую использовать для классификации аргументативных предложений, однако они представляют
интерес как смежные задачи и, возможно, будут использоваться в будущем или для multitask learning.

### T-REX
Link - https://hadyelsahar.github.io/t-rex/downloads/
Triplets for relation extraction (predicate, object, subject) from wikipedia.
6.2M sentences, 11M triplets, 642 unique predicates.

Нельзя использовать для прямого извлечения агрументативных предложений.

### ChangeMyView
Article - https://chenhaot.com/pubs/winning-arguments.pdf
Link - https://chenhaot.com/pages/changemyview.html
20k+ discussion trees.

Автор пишет утверждение и просит изменить его мнение. Пользователи, которым это удалось, присуждаются очки. 
Задача классификации текстов. Еще есть допзадача на этом же корпусе: классификация лучшего текста из пары.

Нельзя использовать для прямого извлечения агрументативных предложений.