# Datasets

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









## Остальные датасеты
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

### Araucariadb
Link - http://corpora.aifdb.org.
662 примера из разных источников. Куски текстов соединены в графы отношений, где одно опровергает другое, третье подтверждает второе и т.п. 662 примера, в среднем 637 символа на текст (119 слов), 8.85 компонент аргументации и 7.88 ребер, соединяющих компоненты.
Скорее мертвый, чем живой проект. Данные удалось скачать только AraucariaDB и только в формате zip, иначе получается битый архив. Содержит предложения, связанные некими связями. По сути предложения выстроены в направленый граф, где каждое ребро означает либо атаку, либо поддержку. В данных отсутствуют отрицательные примеры, по сути все там является и claim и evidence. Очень непонятная разметка, все ребра имеют пустой класс, а предложения имеют неговорящие классы 'I', 'CA', 'YA', 'RA', 'L'.

Т.к. нам хочется искать независимые claims & evidence датасет совсем не подходит.

### Internet Argument Corpus v2
Link - https://nlds.soe.ucsc.edu/iac2
Article link - http://www.lrec-conf.org/proceedings/lrec2016/pdf/1126_Paper.pdf 
A collection of corpora for research in political debate on internet forums. 
It consists of three datasets: 4forums (414K posts), ConvinceMe (65K posts), and a sample from CreateDebate 
(3K posts). It includes topic annotations, response characterizations (4forums), and stance.

Дан вопрос и ответ, для ответа сказано, насколько он disagree/agree, attacking/respectful, emotional/fact, sarcasm;
Для каждой пары вопрос ответ известно, кто "за", а кто "против".