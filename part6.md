# 6. Transformer #

Az eddigi részekben bemutattam a neurális hálózatok kutatásának korai időszakát, majd a backpropagation betanítási algoritmust és a visszacsatolásos hálózatokat. Láttuk, ahogy a word embedding trükkje elvezetett az első használható fordításokig, majd az enkóder-dekóder architektúra és a Bahdanau-attention megoldotta, hogy változó hosszúságú szövegeket is kezelni lehessen. Nézzük, mi hiányzott még, hogy megszülessen a Transformer architektúra! 

Az előző részben tulajdonképpen már minden fontosat összeszedtem. Láttuk, a közönséges visszacsatolásos hálózat nagy hátrányát, hogy az újabb és újabb menetekkel a korábbi bemenetek hatása szép lassan kimosódik. Emiatt használt mindenki önmagában is memóriával rendelkező neurális hálózatot, konvolúciósat vagy LSTM-et, GRU-t. Emiatt trükközött Sutskever csapata azzal, hogy visszafelé adta be a szavakat. És ez ösztönözte a Bahdanau attention mechanizmus megszületését is. A Transformer architektúra létrehozásakor tulajdonképpen azt ismerték fel, hogy nem kell a memóriával rendelkező hálózat, nem kell a visszafelé beadogatás trükkje, önmagában az attention mechanizmus mindent meg tud oldani. Ezért is adták a tanulmányuk címének azt, hogy "Attention Is All You Need", vagyis "Figyelemre van csupán szükség".

A fő motivációt ehhez a felismeréshez az adta, hogy a jobb eredmény elérése érdekében a korábbiaknál nagyobb hálózatokat szerettek volna építeni. A visszacsatolásos hálózatok, mint a konvolúciós és az LSTM azonban nagyon számításigényesek, mert az egyes lépések kevéssé párhuzamosíthatók. A sima előrecsatolásos hálózat viszont jóval egyszerűbb eset. Egyszerű vektor-mátrix szorzásból áll a számítás, amely jól szétválasztható sok vektor-vektor szorzásra. A többmagos processzorok, illetve a sok adaton ugyanolyan műveletet egyidőben elvégezni képes grafikus processzorok (GPU) jól kihasználhatók volnának itt.

Mindezt a Google Brain csapata valósította meg, akik a tanulmányt 2017-ben publikálták.

| <img src="images/JakobUszkoreit.png" height="300" /> | <img src="images/AshishVaswani.png" height="300" /> | <img src="images/IlliaPolosukhin.png" height="300" /> | <img src="images/NoamShazeer.png" height="300" /> |
|:-----------------------------------------------------:|:---------------------------------------------------:|:-----------------------------------------------------:|:-------------------------------------------------:|
|                    Jakob Uszkoreit                    |                   Ashish Vaswani                    |                   Illia Polosukhin                    |                   Noam Shazeer                    |

Itt is enkóder-dekóder architektúrát alkalmaztak, és az attention mechanizmus lépései is megegyeztek a Bahdanau-attention lépéseivel. Annyi fejlesztés történt ebben a tekintetben, hogy bevezettek egy self-attention lépést is.

Az eredeti Bahnadau-attention a dekóderben működött, ahol az volt a feladata, hogy az enkóder által eltárolt sok vektor közül kiválassza a szerinte legfontosabbakat, és azt a néhányat használja csak, amire érdemes figyelmet fordítani. (Mindet nem használhatja, mert nem fér bele az egész egyetlen vektorba.) Ez tehát a dekóderben működő figyelem volt, ami az enkóder adataira figyelt. Nem önmagára, hanem egy másik részegységre.

A Transformerben is van egy pontosan ugyanilyen lépés. De emellett pluszban az enkóderben és a dekóderben is van egy self-attention lépés, vagyis önmagára való figyelés. Ilyenkor az enkóder azon adatokra figyel (azokat használta), melyeket saját maga tárolt el a többi tokenről. És a dekóder is ugyanezt csinálja, a saját maga által eltárolt adatok alapján.

Így aztán nincs szükség arra, hogy az enkóder kimenetét a következő lépésben újra beadjuk a bemeneten. Ez a visszacsatolás ki lett véve a rendszerből. És a dekóder oldalon sincs rá szükség.

| <img src="images/TransformerEncoderDecoder.png" height="400" /> |
|:---------------------------------------------------------------:|
|                           Transformer                           |


Hasonló maradt a feldolgozás sorrendje. Tehát a bemenő szavakat egyesével beadtuk az enkódernek. De az újabb és újabb menetek nem mosták ki a korábbiakat, minden el lett tárolva. És minden menet figyelni tudott az addig beadott szavakra. De csak azokra fókuszált, amik szerinte ebből fontosak voltak.

A dekódert is hasonló iterációval használtuk újra is újra. Beadva neki az előző kimenetet bemenetként. De megintcsak megvolt minden korábbi feldolgozásról minden szükséges információ hagyományos módon eltárolva, és azok közül szemezgetve, súlyozva fel lehetett használni bármelyiket.

Az attention mechanizmusban szerepelt egy további fejlesztés is, a multi-head megoldás. De ennek oka csak a jobb párhuzamosíthatóság volt.

A Google Brain részleg két hónapon belül előrukkolt egy módosított Transformerrel, amelyet már nem fordításra fejlesztettek, hanem egyszerűen csak a szöveg folytatására. Itt tehát nem volt szükség enkóderre és dekóderre is, elég volt egyetlen komponens. (Ezt dekódernek hívják, de tulajdonképpen ellátja az enkóder funkciót is: feldolgozza a bemenetet. Sőt, igazából az enkóder szerkezetével egyezik meg, mert csak self-attention van benne, hisz nincs is másik komponens, amire irányulhatna a figyelem.)

| <img src="images/Decoder-only_transformer.png" height="500" /> |
|:--------------------------------------------------------------:|
|            Csak dekódereket tartalmazó transformer             |

A csak dekóderes változatot természetesen nem egy nyelvpárra tanították be, hanem a következő szót (tokent) kellett mindig megtippelnie. 

A megoldás azonnal mindenkinek felkeltette az érdeklődését, így az OpenAI is azonnal nekilátott, hogy lemásolja azt. (Az Attention All You Need publikáció 2017. decemberében jelent meg, a csak dekóderes változat 2018. január végén, 2018. júniusában pedig kész lett a GPT-1, kompletten letesztelve, tanulmányozva, publikációstul. Persze nem olyan nagy csoda, mert a csapatban volt a már tapasztalt Sutskever is.)   

Ilyenkor szokás azt mondani, hogy "a többi történelem". Nem tudom miért, mert a korábbiak is azok voltak, a többi meg ahhoz képest inkább a jelen. :-) De valahogy mégis ide passzol. Szóval a többi történelem. 2019-ben kijött a GPT-2. 2020-ban a GPT-3, majd a következő évben annak jobban betanított, instrukciókra jobban reagáló változata. 2022. vége felé a ChatGPT. (Ez is csak betanításban fejlődött.) 2023. márciusában pedig megjelent a GPT-4.  

<img src="images/OpenAI-GPT.png" width="900" height="200"/>

Néhány fejlesztést nem említettem meg, melyek olyan apróbb problémákat orvosoltak, melyek hozzájárultak a nyelvi modellek sikeréhez. Ezt most bepótolom.

Az egyik probléma a fix méretű szótár korlátja volt. Vagy ha úgy vesszük, az ismeretlen szavak kezelése, melyek nem fértek bele a szótárba. Ha a bemenet mindig egy szó, akkor ezzel a problémával szembesülünk. A betűnként beadott módszer meg azért nem előnyös, mert így a neuronhálózat alig tudja kikövetkeztetni a szövegben rejlő tartalmi rendszert.

Részmegoldás, ha irgalmatlanul nagy, több százezer szót tartalmazó szótárral dolgozunk. Ezt alkalmazták is, de még így is maradnak ismeretlen szavak. Főleg, ha soknyelvű a rendszer. Az eddigi legjobb megoldás a byte-pair encoding, ami a szavak és a betűk közt félúton áll, vagyis szavak és szó-részletek, valamint betűk vegyesen reprezentáltak. (2016, Sennrich, Haddow, Birch.) (A byte-pair encoding (BPE) eredetileg tömörítési eljárás volt.)

A másik terület az aktivációs függvények kérdése, vagyis hogy a neuron a bemeneteire érkező jelek összegéből hogyan állapítsa meg, mi kerüljön a kimenetére. Sokféle függvényt kipróbáltak, mindegyik alakja nagyjából hasonló: a kimenet egy bizonyos küszöbértékig nulla, vagy ahhoz közeli érték. Afölött pedig emelkedni kezd. Érdemes differenciálható függvényt választani, tehát amelyik folyamatosan változik, nincsenek benne ugrások vagy szakadások, mert akkor nagyon nehéz matematikailag levezetni a betanításkor használt műveleteket. A mostanában leggyakrabban használt függvényt, a "Gaussian Error Linear Unit" nevűt 2018-ban javasolták.

A GELU számítási módja a következő: <img src="images/GELU.png" height="40"/>

| <img src="images/activationFunctions.png" height="200" /> |
|:---------------------------------------------------------:|
|        Három aktivációs függvény összehasonlítása         |


- GPT- Más jellegű transformer alkalmazások (fordítás, keresés, képfeldolgozás, javítás, önvezető autókig)

Foglaljuk össze, milyen elemek voltak szükségesek a Transformer architektúrához:
- Neurális hálózat (1943, 1949, 1958)
- Jobb aktivációs függvények (1969, ???)
- Backpropagation betanítás (1974, 1985-86)
- Szekvenciális bemenet kezelése iterációval (1986)
- Kellő méretű számítási kapacitás, nagy szövegkorpusz (90-es, 2000-es évek)
- Embedding vektor használata (2001, 2013)
- Enkóder-dekóder architektúra (2013) - Opcionális
- Attention mechanizmus (2014)
- Encoder-decoder transformer (2017)
- Decoder only transformer (2018)

Mi az, amivel azóta kísérleteznek?
- Nagyobb méret
- Több, tisztítottabb adat betanításkor, korpusz méretének pontos eltalálása
- Finomhangolás, human feedback használata
- Különféle position embedding. Hosszabb, vagy akár kötetlen hosszúságú szövegek feldolgozására
- Jobb tokenizáció
- Normalizáció helye (numerikus stabilitás)
- Kisebb számítást igénylő finomítások: 16-bites modellek, sparse transformer

- Multimodalitás

Konklúzió:

A fejlődés folyamatos volt, apránként haladt. Egy-egy lépés megtétele benne volt a levegőben, csak az volt a kérdés, hogy ki éri el hamarabb. Ahogy jöttek a komolyabb sikerek, egyre többen kezdtek vele foglalkozni, egyre komolyabb finanszírozással. A megvalósult rendszerek egyre jobban, immár elég látványosan közelítik az emberi képességeket. Az eredmények nem feltétlenül ezt a lineáris fejlődést követik: egy darabig szerények, de aztán beérik a gyümölcs, és gyorsul a fejlődés. Ez talán egy exponenciális görbe, melynek hosszú ideig lapos az íve, majd egy ponton meredeken kilő. (Ez Kurzweil tapasztalata és becslése.) De ha nem is gondoljuk törvényszerűnek, hogy a fejlődés exponenciális, és akár lassabb fejlődési szakaszokat is elképzelhetőnek tartunk, nagy valószínűséggel ezalatt is gyűlnek majd az apró haladások, melyeknek eredményeként később újra kilő a görbe.

Talán szükség van még néhány rafinált trükkre, melyről ma még fogalmunk sincs. De ezek vagy viszonylag gyorsan, az evolúció fokozatos haladásával valósultak meg az emberré válás során, tehát kicsi az esélye, hogy annyira összetett dolgokról legyen szó, amit képtelenség lenne leutánoznunk. Vagy ha evolúciósan jóval hosszabb idő alatt alakult ki, tehát esetleg mégis rettenetesen komplex dologról, dolgokról van szó, akkor a viszonylag egyszerű állatok is képesek lehetnek ezekre. Úgyhogy akkor meg emiatt tűnik valószínűnek, hogy nem vagyunk túlságosan távol az emberi szint elérésétől.

Az elérhető számítási kapacitás egyre nagyobb. Az agy méret-tartományának elérése is a belátható jövőbe került. Agyunk különféle képességek kombinációját valósítja meg, tehát tulajdonképpen sokféle funkció egybegyúrása. Valószínűleg különböző módon betanított hálózatok hibridje. Nem szükséges pontosan ugyanolyat csinálni, de ennek megfelelő szerkezetét kell a jövőben megtalálni, és megvalósulhat az általános mesterséges intelligencia.  Ami a mennyiségi, sebességbeli fejlesztés miatt azonnal a meghaladását is jelenti. Nagy esély van rá, hogy mindez még sokunk életében megvalósul.
