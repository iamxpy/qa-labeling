# !/usr/bin/python
# -*- coding: utf-8 -*-
import re
import html
import string
import psutil
from multiprocessing import Pool
import multiprocessing as mp
from unicodedata import category, name, normalize
import pandas as pd
import numpy as np
from toolz.itertoolz import partition_all, concatv
from joblib import Parallel, delayed


symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}£·©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√«»´º¾¡§£₤·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'


# 里面可能重复的key，因为是从多份代码copy过来的
contraction_mapping = {
    "'cause": 'because', ',cause': 'because', ';cause': 'because', "ain't": 'am not',
    'ain,t': 'am not',
    'ain;t': 'am not', 'ain´t': 'am not', 'ain’t': 'am not', "aren't": 'are not',
    'aren,t': 'are not', 'aren;t': 'are not', 'aren´t': 'are not', 'aren’t': 'are not', "can't": 'cannot',
    "can't've": 'cannot have', 'can,t': 'cannot', 'can,t,ve': 'cannot have',
    'can;t': 'cannot', 'can;t;ve': 'cannot have',
    'can´t': 'cannot', 'can´t´ve': 'cannot have', 'can’t': 'cannot', 'can’t’ve': 'cannot have',
    "could've": 'could have', 'could,ve': 'could have', 'could;ve': 'could have', "couldn't": 'could not',
    "couldn't've": 'could not have', 'couldn,t': 'could not', 'couldn,t,ve': 'could not have', 'couldn;t': 'could not',
    'couldn;t;ve': 'could not have', 'couldn´t': 'could not',
    'couldn´t´ve': 'could not have', 'couldn’t': 'could not', 'couldn’t’ve': 'could not have', 'could´ve': 'could have',
    'could’ve': 'could have', "didn't": 'did not', 'didn,t': 'did not', 'didn;t': 'did not', 'didn´t': 'did not',
    'didn’t': 'did not', "doesn't": 'does not', 'doesn,t': 'does not', 'doesn;t': 'does not', 'doesn´t': 'does not',
    'doesn’t': 'does not', "don't": 'do not', 'don,t': 'do not', 'don;t': 'do not', 'don´t': 'do not',
    'don’t': 'do not', ' dont ': ' do not ',
    "hadn't": 'had not', "hadn't've": 'had not have', 'hadn,t': 'had not', 'hadn,t,ve': 'had not have',
    'hadn;t': 'had not',
    'hadn;t;ve': 'had not have', 'hadn´t': 'had not', 'hadn´t´ve': 'had not have', 'hadn’t': 'had not',
    'hadn’t’ve': 'had not have', "hasn't": 'has not', 'hasn,t': 'has not', 'hasn;t': 'has not', 'hasn´t': 'has not',
    'hasn’t': 'has not',
    "haven't": 'have not', 'haven,t': 'have not', 'haven;t': 'have not', 'haven´t': 'have not', 'haven’t': 'have not',
    "he'd": 'he would',
    "he'd've": 'he would have', "he'll": 'he will',
    "he's": 'he is', 'he,d': 'he would', 'he,d,ve': 'he would have', 'he,ll': 'he will', 'he,s': 'he is',
    'he;d': 'he would',
    'he;d;ve': 'he would have', 'he;ll': 'he will', 'he;s': 'he is', 'he´d': 'he would', 'he´d´ve': 'he would have',
    'he´ll': 'he will',
    'he´s': 'he is', 'he’d': 'he would', 'he’d’ve': 'he would have', 'he’ll': 'he will', 'he’s': 'he is',
    "how'd": 'how did', "how'll": 'how will',
    "how's": 'how is', 'how,d': 'how did', 'how,ll': 'how will', 'how,s': 'how is', 'how;d': 'how did',
    'how;ll': 'how will',
    'how;s': 'how is', 'how´d': 'how did', 'how´ll': 'how will', 'how´s': 'how is', 'how’d': 'how did',
    'how’ll': 'how will',
    'how’s': 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', 'i,d': 'i would',
    'i,ll': 'i will',
    'i,m': 'i am', 'i,ve': 'i have', 'i;d': 'i would', 'i;ll': 'i will', 'i;m': 'i am', 'i;ve': 'i have',
    "isn't": 'is not',
    'isn,t': 'is not', 'isn;t': 'is not', 'isn´t': 'is not', 'isn’t': 'is not', "it'd": 'it would', "it'll": 'it will',
    "It's": 'it is',
    "it's": 'it is', 'it,d': 'it would', 'it,ll': 'it will', 'it,s': 'it is', 'it;d': 'it would', 'it;ll': 'it will',
    'it;s': 'it is', 'it´d': 'it would', 'it´ll': 'it will', 'it´s': 'it is',
    'it’d': 'it would', 'it’ll': 'it will', 'it’s': 'it is',
    'i´d': 'i would', 'i´ll': 'i will', 'i´m': 'i am', 'i´ve': 'i have', 'i’d': 'i would', 'i’ll': 'i will',
    'i’m': 'i am',' Im ': ' I am ', 
    'i’ve': 'i have', "let's": 'let us', 'let,s': 'let us', 'let;s': 'let us', 'let´s': 'let us',
    'let’s': 'let us', "ma'am": 'madam', 'ma,am': 'madam', 'ma;am': 'madam', "mayn't": 'may not', 'mayn,t': 'may not',
    'mayn;t': 'may not',
    'mayn´t': 'may not', 'mayn’t': 'may not', 'ma´am': 'madam', 'ma’am': 'madam', "might've": 'might have',
    'might,ve': 'might have', 'might;ve': 'might have', "mightn't": 'might not', 'mightn,t': 'might not',
    'mightn;t': 'might not', 'mightn´t': 'might not',
    'mightn’t': 'might not', 'might´ve': 'might have', 'might’ve': 'might have', "must've": 'must have',
    'must,ve': 'must have', 'must;ve': 'must have',
    "mustn't": 'must not', 'mustn,t': 'must not', 'mustn;t': 'must not', 'mustn´t': 'must not', 'mustn’t': 'must not',
    'must´ve': 'must have',
    'must’ve': 'must have', "needn't": 'need not', 'needn,t': 'need not', 'needn;t': 'need not', 'needn´t': 'need not',
    'needn’t': 'need not', "oughtn't": 'ought not', 'oughtn,t': 'ought not', 'oughtn;t': 'ought not',
    'oughtn´t': 'ought not', 'oughtn’t': 'ought not', "sha'n't": 'shall not', 'sha,n,t': 'shall not',
    'sha;n;t': 'shall not', "shan't": 'shall not',
    'shan,t': 'shall not', 'shan;t': 'shall not', 'shan´t': 'shall not', 'shan’t': 'shall not', 'sha´n´t': 'shall not',
    'sha’n’t': 'shall not',
    "she'd": 'she would', "she'll": 'she will', "she's": 'she is', 'she,d': 'she would', 'she,ll': 'she will',
    'she,s': 'she is', 'she;d': 'she would', 'she;ll': 'she will', 'she;s': 'she is', 'she´d': 'she would',
    'she´ll': 'she will',
    'she´s': 'she is', 'she’d': 'she would', 'she’ll': 'she will', 'she’s': 'she is', "should've": 'should have',
    'should,ve': 'should have', 'should;ve': 'should have',
    "shouldn't": 'should not', 'shouldn,t': 'should not', 'shouldn;t': 'should not', 'shouldn´t': 'should not',
    'shouldn’t': 'should not', 'should´ve': 'should have',
    'should’ve': 'should have', "that'd": 'that would', "that's": 'that is', 'that,d': 'that would',
    'that,s': 'that is', 'that;d': 'that would',
    'that;s': 'that is', 'that´d': 'that would', 'that´s': 'that is', 'that’d': 'that would', 'that’s': 'that is',
    "there'd": 'there had',
    "there's": 'there is', 'there,d': 'there had', 'there,s': 'there is', 'there;d': 'there had', 'there;s': 'there is',
    'there´d': 'there had', 'there´s': 'there is', 'there’d': 'there had', 'there’s': 'there is',
    "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have',
    'they,d': 'they would', 'they,ll': 'they will', 'they,re': 'they are', 'they,ve': 'they have',
    'they;d': 'they would', 'they;ll': 'they will', 'they;re': 'they are',
    'they;ve': 'they have', 'they´d': 'they would', 'they´ll': 'they will', 'they´re': 'they are',
    'they´ve': 'they have', 'they’d': 'they would', 'they’ll': 'they will',
    'they’re': 'they are', 'they’ve': 'they have', "wasn't": 'was not', 'wasn,t': 'was not', 'wasn;t': 'was not',
    'wasn´t': 'was not',
    'wasn’t': 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have',
    'we,d': 'we would', 'we,ll': 'we will',
    'we,re': 'we are', 'we,ve': 'we have', 'we;d': 'we would', 'we;ll': 'we will', 'we;re': 'we are',
    'we;ve': 'we have',
    "weren't": 'were not', 'weren,t': 'were not', 'weren;t': 'were not', 'weren´t': 'were not', 'weren’t': 'were not',
    'we´d': 'we would', 'we´ll': 'we will',
    'we´re': 'we are', 'we´ve': 'we have', 'we’d': 'we would', 'we’ll': 'we will', 'we’re': 'we are',
    'we’ve': 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is',
    "what've": 'what have', 'what,ll': 'what will', 'what,re': 'what are', 'what,s': 'what is', 'what,ve': 'what have',
    'what;ll': 'what will', 'what;re': 'what are',
    'what;s': 'what is', 'what;ve': 'what have', 'what´ll': 'what will',
    'what´re': 'what are', 'what´s': 'what is', 'what´ve': 'what have', 'what’ll': 'what will', 'what’re': 'what are',
    'what’s': 'what is',
    'what’ve': 'what have', "where'd": 'where did', "where's": 'where is', 'where,d': 'where did',
    'where,s': 'where is', 'where;d': 'where did',
    'where;s': 'where is', 'where´d': 'where did', 'where´s': 'where is', 'where’d': 'where did', 'where’s': 'where is',
    "who'll": 'who will', "who's": 'who is', 'who,ll': 'who will', 'who,s': 'who is', 'who;ll': 'who will',
    'who;s': 'who is',
    'who´ll': 'who will', 'who´s': 'who is', 'who’ll': 'who will', 'who’s': 'who is', "won't": 'will not',
    'won,t': 'will not', 'won;t': 'will not',
    'won´t': 'will not', 'won’t': 'will not', "wouldn't": 'would not', 'wouldn,t': 'would not', 'wouldn;t': 'would not',
    'wouldn´t': 'would not',
    'wouldn’t': 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', 'you,d': 'you would',
    'you,ll': 'you will',
    'you,re': 'you are', 'you;d': 'you would', 'you;ll': 'you will',
    'you;re': 'you are', 'you´d': 'you would', 'you´ll': 'you will', 'you´re': 'you are', 'you’d': 'you would',
    'you’ll': 'you will', 'you’re': 'you are',
    '´cause': 'because', '’cause': 'because', "you've": "you have", "could'nt": 'could not',
    "havn't": 'have not', "here’s": "here is", 'i""m': 'i am', "i'am": 'i am', "i'l": "i will", "i'v": 'i have',
    "wan't": 'want', "was'nt": "was not", "who'd": "who would",
    "who're": "who are", "who've": "who have", "why'd": "why would", "would've": "would have", "y'all": "you all",
    "y'know": "you know", "you.i": "you i",
    "your'e": "you are", "arn't": "are not", "agains't": "against", "c'mon": "common", "doens't": "does not",
    'don""t': "do not", "dosen't": "does not",
    "dosn't": "does not", "shoudn't": "should not", "that'll": "that will", "there'll": "there will",
    "there're": "there are",
    "this'll": "this all", "u're": "you are", "ya'll": "you all", "you'r": "you are", "you’ve": "you have",
    "d'int": "did not", "did'nt": "did not", "din't": "did not", "dont't": "do not", "gov't": "government",
    "i'ma": "i am", "is'nt": "is not", "‘I": 'I',
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've":
    "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's":"this is","that'd": "that would",
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
    "today's": "today is", ' yr old ': ' years old '，' u r ': ' you are ', ' u ': ' you ','e.g.': 'for example', 'i.e.': 'in other words', '...': '.', 
    'et.al': 'elsewhere',
}

# 需要针对数据集来设计, 例如可以包括mis-spell的词, 被敏感处理需要恢复的词等等
other_mapping = {'whattsup': 'WhatsApp', 'whatasapp': 'WhatsApp', 'whatsupp': 'WhatsApp',
                     'whatcus': 'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat': 'what',
                     'Whwhat': 'What', 'whatshapp': 'WhatsApp', 'howhat': 'how that',
                     'Whybis': 'Why is', 'laowhy86': 'Foreigners who do not respect China',
                     'Whyco-education': 'Why co-education',
                     "Howddo": "How do", 'Howeber': 'However', 'Showh': 'Show',
                     "Willowmagic": 'Willow magic', 'WillsEye': 'Will Eye', 'Williby': 'will by',
                     'pretextt': 'pre text', 'aɴᴅ': 'and', 'amette': 'annette', 'aᴛ': 'at', 'Tridentinus': 'mushroom',
                     'dailycaller': 'daily caller', "™": 'trade mark','f***': 'fuck', 'f**': 'fuc', 'F***': 'fuck', 'F**': 'fuc', 'a****': 'assho', 'a**': 'ass',
                     'h***': 'hole', 'A****': 'assho', 'A**': 'ass', 'H***': 'hole',
                     's***': 'shit', 's**': 'shi', 'S***': 'shit', 'S**': 'shi', 'Sh**': 'shit',
                     'p****': 'pussy', 'p*ssy': 'pussy', 'P****': 'pussy', 'p***': 'porn', 'p*rn': 'porn',
                     'P***': 'porn',
                     'st*up*id': 'stupid', 'd***': 'dick', 'di**': 'dick', 'h*ck': 'hack',
                     'b*tch': 'bitch', 'bi*ch': 'bitch', 'bit*h': 'bitch', 'bitc*': 'bitch', 'b****': 'bitch',
                     'b***': 'bitc', 'b**': 'bit', 'b*ll': 'bull','f**k': 'fuck', 'F**k': 'fuck', 'F**K': 'fuck'}


small_caps_mapping = {
"ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ғ": "f", "ɢ": "g", "ʜ": "h", "ɪ": "i",
"ᴊ": "j", "ᴋ": "k", "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ǫ": "q", "ʀ": "r",
"s": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "x": "x", "ʏ": "y", "ᴢ": "z"}


mapping_dict=dict(small_caps_mapping.items()+contraction_mapping.items()+mis_spell_mapping.items())


isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}


def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x


def replace_words(x, dic):
    for word in dic.keys():
        if word in x:
            x = x.replace(word, dic[word])
    return x

def clean_white_space(x): 
    """
    有些Tokenizer（例如Roberta）不会去除空格与回车等字符，因此需要自己去除
    """
    return " ".join(x.split())

def preprocess(x):
    x = clean_white_spece(x)
    x = handle_punctuation(x)
    x = replace_words(x,mapping_dict) 
    return x


def nn_preprocess(df, columns):
    """
    df是含有文本列的DataFrame, columns为其中的文本列
    """
    # question_title, question_body, answer
    parallel = Parallel(48, backend="multiprocessing", verbose=0)
    
    print('preprocessing ...')
    for c in columns:
        df[c] = parallel(delayed(preprocess)(x) for x in df[c].tolist())
        # 或：
        # df[c] = df[c].progress_apply(lambda x:preprocess(x))

        # replace &gt; with > and &lt; with < and &amp; with & and &quot; with "
        df[c] = df[c].progress_apply(html.unescape)
        print(f'column {c} done')
    return df