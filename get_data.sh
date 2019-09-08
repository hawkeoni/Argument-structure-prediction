WGET="wget -v --no-check-certificate"
FILES="IBM*zip"
mkdir datasets

# Argument Detection
FOLDER="datasets/Argument_Detection"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_claim_sentences_search.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_EvidenceSentences.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CE-EMNLP-2015.v3.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CE-ACL-2014.v0.zip"
mv $FILES $FOLDER

# Argument Quality
FOLDER="datasets/Argument_Quality"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_EviConv-ACL-2019.v1.zip"
mv $FILES $FOLDER

# Argument Stance Classification and Sentiment Analysis
FOLDER="datasets/Argument_Stance"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CS_EACL-2017.v1.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_SLIDE_LREC_2018.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_SC_COLING_2018.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_SC_COLING_2018.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_WC-ACL-2016.v2.zip"
mv $FILES $FOLDER

# Debate Speech Analysis
# Possible duplicates, need to look further into this
# part of the dataset
FOLDER="datasets/Debate_Speech"
mkdir -p $FOLDER
$WGET "https://ibm.box.com/shared/static/3cen7li36e6w6e8cjdknnayntd3o2o79.zip"
$WGET "https://ibm.box.com/shared/static/xsmlp61rlf2pc02r81y751ihcnqfpwzb.zip"
$WGET "https://ibm.box.com/shared/static/rcfv83vpttjumd51ecq9sz86fqfh6dmg.zip"
$WGET "https://ibm.box.com/shared/static/fk1furl3kzuao808yx42lnuus9q4em7l.zip"
$WGET "https://ibm.box.com/shared/static/c6mkf6pewzzu59ht9n77fhiq318xg7st.zip"
$WGET "https://ibm.box.com/shared/static/slk5zznv0dxvux5yfqzmwffzhkbo6w8q.zip"
$WGET "https://ibm.box.com/shared/static/wljkrr488pzuvzln1cw8ti9yp0puc2y5.zip"
$WGET "https://ibm.box.com/shared/static/tlmdcf9j21hzwlikpvc47t81vegbqi0c.zip"
$WGET "https://ibm.box.com/shared/static/4kg3arwa0cwi7bgaltnr9va1jqtzfxnx.zip"
mv $FILES $FOLDER

# Expressive Text to Speech
FOLDER="datasets/Expressive_TTS"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_EW-Interspeech-2018.zip"
mv $FILES $FOLDER

# Basic NLP Tasks
FOLDER="datasets/Basic_NLP"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_WORD-LREC-2018.v0.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_TR9856.v2.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_MD-arXiv-2018.v0.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_TCS-ACL-2018.v0.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_Concept-Abstractness.zip"
mv $FILES $FOLDER

# Classes of Principled Arguments
FOLDER="datasets/Principled_Arguments"
mkdir -p $FOLDER
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CoPA-Motion-ACL-2019.v0.zip"
$WGET "https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_CoPA-Speech-ACL-2019.v0.zip"
mv $FILES $FOLDER
