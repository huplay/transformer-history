# 6. Transformer #

- A fix méretű szótár problémájának megoldása - Bpe tokenizáció- Jobb aktivációs függvények (nemlinearitás. Mikor?????????? Hozzuk ezt át az első részből)- Next word prediction (??????????? Hol is jelent az meg?)- Decoder only transformer
- GPT- Más jellegű transformer alkalmazások (fordítás, keresés, képfeldolgozás, javítás, önvezető autókig)
- GAN ????https://matthewdwhite.medium.com/a-brief-history-of-generative-ai-cb1837e67106

Mondjuk el, hogy az enkóder-dekóder megoldás nem is szükséges a generatív esetre!
A Transformer architektúrához szükséges elemek:- Neurális hálózat (1943, 1949, 1958)- Jobb aktivációs függvények (1969, ???)- Backpropagation betanítás (1974, 1985-86)- Szekvenciális bemenet kezelése iterációval (1986)- Kellő méretű számítási kapacitás, nagy szövegkorpusz (90-es, 2000-es évek)- Embedding vektor használata (2001, 2013)- (Opcionális: encoder-decoder architektúra (2013))
- Attention mechanizmus (2014)- Encoder-decoder transformer (2017)- Decoder only transformer (2018)
  Apró trükkök:- Multihead attention- Tanult embedding- normalizáció ???
- recurrent connection ????- position encoding ????
- több fokozat ???- még jobb tokenizerek
  Amivel azóta játszanak:- Nagyobb méret- Több, tisztítottabb adat betanításkor, korpusz méretének pontos eltalálása- Finomhangolás, human feedback használata- Különféle position embedding. Hosszabb, vagy akár kötetlen hosszúságú szövegek feldolgozására- Jobb tokenizáció- Normalizáció helye (numerikus stabilitás)- Kisebb számítást igénylő finomítások: 16-bites modellek, sparse transformer
- Multimodalitás



Konklúzió: a fejlődés folyamatos volt, apránként haladt. Egy-egy lépés megtétele benne volt a levegőben, csak az volt a kérdés, hogy ki éri el hamarabb. Ahogy jöttek a komolyabb sikerek, egyre többen kezdtek vele foglalkozni, egyre komolyabb finanszírozással. A megvalósult rendszerek egyre jobban, immár elég látványosan közelítik az emberi képességeket. Az eredmények nem feltétlenül ezt a lineáris fejlődést követik: egy darabig szerények, de aztán beérik a gyümölcs, és gyorsul a fejlődés. Ez talán egy exponenciális görbe, melynek hosszú ideig lapos az íve, majd egy ponton meredeken kilő. (Ez Kurzweil tapasztalata és becslése.) De ha nem is gondoljuk törvényszerűnek, hogy a fejlődés exponenciális, és akár lassabb fejlődési szakaszokat is elképzelhetőnek tartunk, nagy valószínűséggel ezalatt is gyűlnek majd az apró haladások, melyeknek eredményeként később újra kilő a görbe.

Talán szükség van még néhány rafinált trükkre, melyről ma még fogalmunk sincs. De ezek vagy viszonylag gyorsan, az evolúció fokozatos haladásával valósultak meg az emberré válás során, tehát kicsi az esélye, hogy annyira összetett dolgokról legyen szó, amit képtelenség lenne leutánoznunk. Vagy ha evolúciósan jóval hosszabb idő alatt alakult ki, tehát esetleg mégis rettenetesen komplex dologról, dolgokról van szó, akkor a viszonylag egyszerű állatok is képesek lehetnek ezekre. Úgyhogy akkor meg emiatt tűnik valószínűnek, hogy nem vagyunk túlságosan távol az emberi szint elérésétől.

Az elérhető számítási kapacitás egyre nagyobb. Az agy méret-tartományának elérése is a belátható jövőbe került. Agyunk különféle képességek kombinációját valósítja meg, tehát tulajdonképpen sokféle funkció egybegyúrása. Valószínűleg különböző módon betanított hálózatok hibridje. Nem szükséges pontosan ugyanolyat csinálni, de ennek megfelelő szerkezetét kell a jövőben megtalálni, és megvalósulhat az általános mesterséges intelligencia.  Ami a mennyiségi, sebességbeli fejlesztés miatt azonnal a meghaladását is jelenti. Nagy esély van rá, hogy mindez még sokunk életében megvalósul.
