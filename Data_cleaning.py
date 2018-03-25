## Disclaimer: some words in the dataset may be considered profane, vulgar, or offensive.

import numpy as np
import pandas as pd
import re
import string
import json
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from datetime import datetime



def Total_clean(train,test):
    
    def timer(start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod(
                (datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' %
                  (thour, tmin, round(tsec, 2)))

    # Data processing incorporated the following two kernels:
    # https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
    # https://www.kaggle.com/prashantkikani/pooled-gru-with-preprocessing/code
    
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    
    traintime = timer(None)
    train_time = timer(None)
    
    train['comment_text']=train['comment_text'].fillna("NNN")
    
    test['comment_text']=test['comment_text'].fillna("NNN")
    tr_ids = train[['id']]
    train[class_names] = train[class_names].astype(np.int8)
    target = train[class_names]

    print(' Cleaning ...')

    Toxiclist={}

    Toxiclist['faggot']=["fagetass", "jewfaggotmongo","faggot",
                "fagt", "fags", "fagg", "faga", "fagometer",
                "faggle", "fagzrz", "fagging", "youfaggot",
                "profaggot", "faggoting", "loverfaggot", 
                "gayfagonaplane", "fagti", "faghri", 
                "faggots", "faggotu", "faggott", "faggoty", 
                "fagernes", "massivefaggothater", 
                "newfag", "faggett", "bongfag",
                "faggotloving", "allfaggotsmustburn", "faggieness", "gayfag",
                "cockfag", "faggggggggggggit", "handfagtwhy", "helpedfaggots", 
                "comingofage", "swedefag", "faggot", "quebecfaggots", 
                "faggotmongo", "wikifag", "faggotts", "faggotjiggaboomongo", 
                "fagplease", "douchefaggot", "furfag", "wikifags", 
                "jewfagtwhy", "ranmafag", "faganthony",
                "useracceptefag",  
                "fagood", "fagnes", "faggotttttttt", "fagopedia",
                "fayssalfag", "fagan", "faggggg", "fagex", 
                "fages", "faget", "handfaggotmongo", 
                "faggotgot", "cruiserfaggot", "afaghaneh", "aquafag",
                "selfaggrandize", "lolfagstar", "fagcircle", "fagit", 
                "fagin", "fagreterion", "fagots", "fagoty", "fagetyou", 
                "fagtwhy", "faggs", "faggt", "faggy", "faggg", "fag", 
                "categoryfaghags", "faggit", "fagbag", "faithlessfaggotboy", 
                "fagggot", "superfag", "faggotry", 
                "faggotness", "fagget", "fagged", "fagot", "fagmom", 
                "fagtard", "faggotttttttttttt", "wikifagia", "faggity",
                "fjrgugfaggot", "fagat", 
                "profaggots", "fagboy", "selfaggrandizing", "furfags", "selfaggrandizement"]

    Toxiclist['jerk']=["masturbationjerk", "bjerknes", "jerkwad",
              "englishjerk", "jerks", "jerky", "jerkwater", "jerkso",
              "jerkoff", "jerkish", "jerkoffs", "tearjerking", "jerkov",
              "jerkremove", "radiojerk", "jerkwads", "kneejerk", "dreamjerk", 
              "jerking", "kneejerkly", "jerkofoz", "jerknazi", "jerkass", 
              "jerkins", "jerked", "jerkweasel", "circlejerk", "wikijerks","jerk"]

    Toxiclist['noob']=["noobs", "noobish", "noobies", "noobuntu", "noobenger", 
              "sanoobar", "nipplenoob", "noobie", "acanoobic", "janoob",
              "noobsgo", "noobness", "noob", "noobsauce", "noobarino", "nooblet", 
              "noobieness"]
    Toxiclist['rape']=["raped", "rapes", "raper", "violencerapes", "rapesexual",
              "talkrape",  "rapeunited", "childrape",  "rapeist", "rapeing",
              "rapers","raperaped",  "wikirape", "rape"]

    Toxiclist['pig']=["pigment", "pigmented",   "pigyea","pigporky", "pigsnort", 
             "hyperpigmented",  "piglike",  "pigments", 
              "panpig","pigman", "pigofabully", "pigtails",
             "pigtest", "pighead", "pigheadedness", "pigsick", 
              "saltypig", "piggery", "ipigott",  "pigdogs",  "piggy", 
             "pigmentation", "talkpigmanman", 
              "pignose",  "pigi", "pigs",  "pighumping", "pigheaded"]
    Toxiclist['slut']=["slutloadcom", "slutwalk", "reslut", "slutter", "sluthouse", 
              "slutbag", "sluthing", "manslut", "slutty",   "sluts","slut"]
    Toxiclist['fuck']=["fukyou", "fukushima","fuking",  
              "starfuk",  "fukhead", "fukcing", "halfukrainian", 
              "fuked", "fukka", "fukkk", "fukng", "fukkkk", 
              "fuker", "scumfuk", "mothafuka", 
              "asfuking", "fukface", "kfukimoto", 
              "muvvafukka", "fukker", "textfukin",
              "fukking", "fukersikh", "fukkers", 
              "ufuk", "motherfuker", "motherfukkin",
              "fukkatsu", "fukuhara", "mathrfuker",
              "catastrofuk", "fuku", "fuks", "fukn", 
              "fuko", "fukwit", "fuk", "fukin",'fuck','fucck','motherfuking']

    Toxiclist['cunt']=["douch","douc","douche"]
    Toxiclist['moron']=["moron","morons","douche"]
    Toxiclist['insult']=['insluting','inslut']

    # PREPROCESSING PART
    repl = {
        "yay!": " good ",
        "yay": " good ",
        "yaay": " good ",
        "yaaay": " good ",
        "yaaaay": " good ",
        "yaaaaay": " good ",
        ":/": " bad ",
        ":&gt;": " sad ",
        ":')": " sad ",
        ":-(": " frown ",
        ":(": " frown ",
        ":s": " frown ",
        ":-s": " frown ",
        "&lt;3": " heart ",
        ":d": " smile ",
        ":p": " smile ",
        ":dd": " smile ",
        "8)": " smile ",
        ":-)": " smile ",
        ":)": " smile ",
        ";)": " smile ",
        "(-:": " smile ",
        "(:": " smile ",
        ":/": " worry ",
        ":&gt;": " angry ",
        ":')": " sad ",
        ":-(": " sad ",
        ":(": " sad ",
        ":s": " sad ",
        ":-s": " sad ",
        r"\br\b": "are",
        r"\bu\b": "you",
        r"\bhaha\b": "ha",
        r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bcan't\b": "can not",
        r"\bcannot\b": "can not",
        r"\bi'm\b": "i am",
        "m": "am",
        "r": "are",
        "u": "you",
        "haha": "ha",
        "hahaha": "ha",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "i'm": "i am",
        "m": "am",
        "i'll" : "i will",
        "its" : "it is",
        "it's" : "it is",
        "'s" : " is",
        "that's" : "that is",
        "weren't" : "were not",
        "lets": "let us",
        "theres":"there is",
        "pls":"please",
        "fukng":"fuck",
        "cannotbut":"can not but",
        "fileimgpjpg":"",
        "pcprotection":"protection",
        "runoffround":"run off round",
        "japanchannel":"japan channel",
        "youhahahahi":"you ha",
        "wahahhahaahah":"ha",
        'fckers':"fuck",
        "backgroundfff":"background",
        "broncofreak":"freak",
        "fkc":"fuck",
        "wtf":"fuck",
        "youre":"you are",
        "wtf":"fuck",
        "fuckiing":"fucking",
        "usertalkfuckyou":"fuck",
        "dumbfuck":"fuck",
        "fucked":"fuck",
        "motherfucking":"fucking",
        "cumbag":"bitch",
        "antiislam":"antislam",
        "selfimportant":"arrogant",
        "backstabb":"backstab",
        "btard":"bastard",
        "nonesens":"nonsense",
        "pathetical":"miserable",
        "cencor":"censor",
        "jibb":"jiz",
        "fyo":"faggot",
        "bigdunc":"dick",
        "rvert":"revert",
        "diots":'idiot',
        "hebag":"bitch",
        "rverted":"revert",
        "wikipolice":"arrogant",
        "omite":"omit",
        "dmack":"dick",
        "sycop":"arrogant",
        "trouted":"drunk",
        "fnot":"bitch",
        "fukkin":"fucking",
        "fucc":"fuck",
        "fcking":"fucking",
        "fckin":"fucking",
        "fcuk":"fuck",
        "fcukin":"fucking",
        "fck":"fuck",
        'hcore':"whore",
        "wobbs":"whore",
        "selfabsorbed":"arrogant",
        "ecock":"cock",
        "fggt":"faggot",
        "biatch":"bitch",
        "niggas":"nigger",
        'cck':'cock',
        'ckhead':'cock',
        "faaa":"fuck",
        "ngros":"nigger",
        "folloin":"stalker",
        "ashol":"asshole",
        "assaultin":"assault",
        "diots":"idiot",
        "idiotical":"idiot",
        "ngros":"nigger",
        "njgw":"nigger",
        "fagget":"faggot",
        "fukkin":"fuck",
        "fcuk":"fuck",
        "fagget":"faggot",
        "motherf":"motherfucker",
        "fuking":"fucking",
        "fckin":"fucking",
        "fcking":"fucking",
        "fagot":"faggot",
        "niggas":"nigger",
        "fuc":"fuck",
        "fagg":"faggot",
        "ggot":"faggot",
        "btch":"bitch",
        "nigga":"nigger",
        "fukin":"fucking",
        "faggots":"faggot",
        "fag":"faggot",
        "fking":"fucking",
        "arsehole":"asshole",
        "fukk":"fuck",
        "scumbags":"bitch",
        "nigg":"nigger",
        "fkin":"fucking",
        "asswipe":"ass",
        "shyt":"shit",
        "ussy":"pussy",
        "cun":"cunt",
        "bich":"bitch",
        "gaye":'gay',
        "gga":"gay",
        "igger":"nigger",
        "gger":"nigger",
        "fukkin":"fucking",
        "iggers":"nigger",
        "diem":"die",
        "slutt":"slut",
        "jdelanoy":"bad",
        "fuckiing":"fucking",
        "usertalkfuckyou":"fuck",
        "fucck":"fuck",
        "wustenfuchs":"fuck",
        "wpfuc":"fuck",
        "fuccin":"fucking",
        "dumbfuck":"fuck",
        "fucin":"fucking",
        "fucing":"fucking",
        "fucccccckkkkk":"fuck",
        "fuchs":"fuck",
        "wstenfuchs":"fuck",
        "fuca":"fuck",
        "fuch":"fuck",
        "fucxxxk":"fuck",
        "fucers":"fuck",
        "fucyour":"fuck",
        "silberfuchs":"fuck",
        "motherfucking":"fucking",
        "motherfuccker":"motherfucker",
        "dihck":"dick",
        "nihgga":"nigger",
        "uck":"fuck",
        "stup":"stupid",
        "idio":"idiot",
        "enis":"penis",
        "iot":"idiot",
        "idioti":"idiot",
        "oron":"moron",
        "tink":"stink",
        "coc":"cock",
        "moro":"moron",
        "oron":"moron",
        "idiotic":"idiot",
        "igger":"nigger",
        "homos":"homosexual",
        "mothe":"mother",
        "fk":"fuck",
        "idi":"idiot",
        "tard":"idiot",
        "astar":"bitch",
        "negro":"nigger",
        "retarde":"idiot",
        "retard":"idiot",
        "retarded":"idiot",
        "byatch":"bitch"

    }

    replace_gram={" i ll ":" i will ",
                 " ki ll ":" kill ",
                  " im ":" i am ",
                 " ill be ":" i will be ",
                  "ball licking":"obscene",
                    "lemonparty":"obscene",
                    "lemon party":"obscene",
                    "ballicks":"obscene",
                    "dlickheads":"idiot",
                    "wet dream":"obscene",
                    " eatballs":" obscene",
                    "doggy style":"obscene",
                    "brea5t":"obscene",
                    " tea bagging":" obscene",
                    "deep throat":"obscene",
                    "fudge packer":"faggot",
                    "doggie style":"obscene",
                    "dog style":"obscene",
                    "ball sucking":"obscene"}


    new_train_data = []
    new_test_data = []

    from copy import deepcopy
    def Pre_process_sentence(sen):
        sens=deepcopy(sen)
        
        for k in replace_gram:
            sens=sens.replace(k,replace_gram[k])

        arr = str(sens).split(' ')
        xx = ""
        for WW in arr:
            j=deepcopy(str(WW).lower())
            if j=="?" or j=="!":
                xx += j+" "
                continue

            if j.endswith('?') or j.endswith('!'):
                Puc=j[-1]
            else:
                Puc=""

            j="".join([char for char in j if char not in string.punctuation])


            if j in repl:
                j = repl[j]
                
                
                
            for key_word in ["shit","fool","stupid","rubbish",'bitch',\
                                'whore','damn','bloody','dick','shag','piss','suck',\
                                'nigger','monkey','dumb','trash','garbage',\
                               'bastard','bumpkin','silly','queer','piss','freak','sissy',\
                               'nigger','coward','chicken','jiz',"lesbian","nerd","geek",'junk','laugh','fuck',\
                            'rape','faggot','asshole','idiot','retard','wank','stupid']:
                if key_word in j:
                    j =key_word
                    break

            for key_word in Toxiclist:
                for word in Toxiclist[key_word]:
                    if word == j:
                        j=key_word
                        break

            if ('http' in j) or ('www' in j) or ('jpg' in j) or\
               ('png' in j ):
                continue
            
            
            elif ('cock' in j) and ("cocktail" not in j) and ("peacock" not in j):
                j= "cock"

            elif ('nigg' in j) and ("snigg" not in j):
                j= "nigger"

            elif ('fuck' in j) or ('fck' in j) or (j.startswith('fuck') or (j=='fuc')):
                j='fuck'

            elif "asshol" in j:
                j="asshole"

            elif j.startswith('cunt'):
                j="cunt"
                
            elif j.startswith('stink'):
                j="stink"

            elif "puss" in j:
                j="pussy"

            elif ("dung" in j) and ("dungeon" not in j):
                j="dung"

            elif (j.startswith('punk') ) or (j.endswith('punk')):
                j="punk"


            elif (j.startswith('junk') ):
                j="junk"


            elif ("wierdo" in j) or ("weirdo" in j):
                j="weirdo"
            elif ("butt" in j) and ("butter" not in j) \
                and ("button" not in j) and ("buttle" not in j):
                j="butt"
            
            elif ("hahaha") in j:
                j= "laugh"


            if len(Puc)==1:
                xx += j +" "+Puc+" "
            else:
                xx += j+" "
        return xx



    train["new_comment_text"] = train['comment_text'].map(Pre_process_sentence)
    test["new_comment_text"] = test['comment_text'].map(Pre_process_sentence)

    trate = train["new_comment_text"].tolist()
    tete = test["new_comment_text"].tolist()
    for i, c in enumerate(trate):
        trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
    for i, c in enumerate(tete):
        tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
    train["comment_text"] = trate
    test["comment_text"] = tete
    del trate, tete
    train.drop(["new_comment_text"], axis=1, inplace=True)
    test.drop(["new_comment_text"], axis=1, inplace=True)
    train['comment_text'].fillna('NAN',inplace=True)
    test['comment_text'].fillna('NAN',inplace=True)


    timer(train_time)
    
    return train,test

if __name__=="__main__":

        
    with open('config.json') as json_data_file:
        config_dic = json.load(json_data_file)
        print(config_dic)

    data_dic=config_dic['data_dic']

    data_path=data_dic["data_path"]

    TRAIN_DATA_FILE=data_path+data_dic['raw_train_file']
    TEST_DATA_FILE=data_path+ data_dic['raw_test_file']

    train_raw = pd.read_csv(TRAIN_DATA_FILE)
    test_raw = pd.read_csv(TEST_DATA_FILE)

    train_clean, test_clean =Total_clean(train_raw,test_raw)
    train_clean.to_csv(data_path+'train_clean.csv',index=False)
    test_clean.to_csv(data_path+'test_clean.csv',index=False)
