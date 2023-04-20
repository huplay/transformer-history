# 2. A tanítás algoritmusa: backpropagation #

Az első részben bemutattam a neuronhálózatok kutatásának kezdeti időszakát, melynek végén a kutatás tulajdonképpen zátonyra futott, mert nem jöttek a remélt eredmények. Vajon miért nem?

Utólag már látszik, hogy a problémát elsősorban az okozta, hogy két rétegnél mélyebb hálózatot a gyakorlatban nem sikerült betanítaniuk. (Hebb módszere többre nem volt képes.) És ez csupán lineárisan elválasztható kategorizálást tud megvalósítani, vagyis olyanokat, ahol a bemenet/kimenet értékeket koordináta-rendszerben ábrázolva egyenes vonalakkal el lehetett választani az eseteket. (Szóval bizonyos eseteket meg lehetett vele oldani, másokat meg sehogysem.)

Ezen egy újabb fontos fejlesztés sem segített, melyet 1969-ben Kunihiko Fukishima alkalmazott először, hogy az addigi küszöb-aktiváció helyett megjelent a RELU (Rectified Linear Unit) aktivációs függvény. Itt a korábbihoz hasonlóan egy bizonyos küszöbértékig 0 maradt a kimenet, de a küszöböt elérve ez az adott érték jelent meg a kimeneten, tehát nem hirtelen ugrással egy 1-es.
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

1974-ben Paul Werbos amerikai tudós doktori disszertációjában bemutatta, hogy egy más területeken alkalmazott algoritmus alkalmas többrétegű neuronhálózatok betanítására is. A területtől azonban annyira elfordultak a kutatók, hogy ez gyakorlatilag visszhang nélkül maradt. Werbos eredményét nem ismerve 1985-ben Yann Le Cun és David Parker egymástól függetlenül újra felismerte ugyanezt. Ennek hatására született meg 1986-ban Rumelhart, Hinton és Williams tanulmánya, mely lényegében széleskörűen ismertté tette az általuk back-propagation-nek elnevezett módszert.

Hinton és Yann Le Cun a mai napig a terület legnagyobb szaktekitélyei. Le Cun a Facebook vezető kutatója a mesterséges intelligencia területén, Hinton pedig a Torontói Egyetem professzora, de rengeteg más eredménnyel is hozzájárultak a terület fejlődéséhez. Önállóan, illetve tanítványaikat segítve is. Yoshua Bengio-val együtt 2018-ban ők kapták a Turing-díjat, amelyet a számítástechnika Nobel-díjaként is aposztrofálnak. (1 millió dollár a jutalom.) (Hármukat szokás a Deep Learning keresztapjaiként, vagy a három muskétásként nevezni.)

Yann Le Cun 1986 (az eredeti 1985-ös francia publikáció angol változata):
https://link.springer.com/chapter/10.1007/978-94-011-0770-9_2
http://yann.lecun.com/exdb/publis/pdf/lecun-86.pdf

Rumelhart, Hinton, Williams 1986: http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf

TODO: Vázoljuk picit a módszert!

A 80-as évek számítógépei azonban nem rendelkeztek kellően nagy számítási teljesítménnyel, és a betanításhoz szükséges hatalmas adatmennyiség sem állt rendelkezésre, így gyakorlati eredményre egészen 1993-ig kellett várni, mikor egy mintafelismerési versenyen nyert egy neurális hálózatot használó program. Ne feledjük, hogy a web is csak ekkoriban jött létre (1991-ben), és használata csak az 1990-es évek közepére terjedt el. (Én 1996-ban még nem láttam élőben internetet, pedig ebben az évben végeztem a főiskolán informatika szakirányon. Tanultunk ugyan róla, meg talán volt is az iskolában néhány szerveren elérés, de a gyakorlatban a számítógépes hálózatot csak arra használtam, hogy párhuzamos porton összekötve két gépet egy osztálytársammal Doom-ozhattunk.) Tehát ekkoriban nem léteztek nagy szöveges szövegkorpuszok, nem volt fenn minden könyv vagy publikáció a neten. Alig voltak hírportálok (a Yahoo 1994-ben alakult, a magyar Internetto 1995-ben, az Index 1997-es). Szociális hálók pedig még később lettek. (A Facebook-ot 2004-ben alapították, bár mi magyarok már 2002-től használhattuk a WiW-et, majd iWiW-et.)

Tehát nagyjából a 90-es évek közepétől kezdett gyűlni a digitális korpusz, de inkább csak a 2000-es évtized hozta el azt, hogy ilyenekhez a kutatók is hozzáfértek.

Ebben a részben tehát bemutattam a backpropagation algoritmust, a következő részben további eredményekről számolok be.