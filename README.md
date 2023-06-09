# Számítógépes grafika 2022/23/2

## Első házi feladat: UFO hami

A hiperbolikus síkon két azonos méretű UFO hami garázdálkodik. A zöld egy körpályán kering, a pirosat a felhasználó irányítja ('e': menj az orrod irányába egyenesen állandó sebességgel; 's' fordulj jobbra, 'f': fordulj balra állandó szögsebességgel). A cél, hogy a piros UFO hami bekapja a zöld UFO hamit. A hamik a csigához hasonlóan a nyálukat a síkon hagyják, azaz a meglátogatott pontokon fehér színű görbék rajzolódnak ki. A hamik teste kör, szájuk kör alakú és ciklikusan nyílik, illetve záródik, két szemük, bennek szemgolyók vannak, amelyek ugyancsak körök. A hamik mindig egymásra néznek. Mi a jelenet Beltrami-Poincaré vetületét élvezhetjük a képernyőnkön.

A megoldás során a következő függvényeket valósítsa meg és használja fel:

1. Egy irányra merőleges irány állítása.

2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.

3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.

4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.

5. Egy pontban egy vektor elforgatása adott szöggel.

6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.

## Második házi feladat: Lehallgatástervező

Készítsen CPU sugárkövetés felhasználásával lehallgatástervező programot.

1. A megjelenített szoba téglatest alakú benne két további típusú Platon-i szabályos test található (Pl. OBJ formátumú definíciójuk: https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html). A szoba falai kívülről befelé átlátszóak, így belelátunk a szobába még kívülről is.

2. A szobában 3 darab kúp alakú lehallgató található. Az lehallgató csak a kúp szögében képes érzékelni, az érzékenység a távolsággal csökken (a csökkenés sebessége megválasztható, cél az esztétikus megjelenés).

3. A szobát és eszközeit alapesetben szürke színnel spekuláris-ambiens modellel jelenítük meg, amely a felületi normális és a nézeti irány közötti szög koszinuszának lineáris függvénye, amelynek értékkészlete a [0.2, 0.4] tartomány (L = 0.2 * (1 + dot(N, V)), ahol L az észlelt sugársűrűség, N a felületi normális, V pedig a nézeti irány, mindketten egységvektorok).

4. A lehallgatás érzékenységét a szürke alapszínhez hozzáadjuk, az első lehallgatóra piros árnyalatokkal, a másodikra zölddel, a harmadikra kékkel. Az érzékenység a takart pontokban zérus.

5. A lehallgatók interaktívan áthelyezhetők. A bal egérgomb lenyomására, a kurzor alatt látható 3D ponthoz megkeressük a legközelebbi lehallgatót, annak pozícióját a megtalált pontra állítjuk, az irányát pedig a felület normálvektorára.

## Harmadik házi feladat: Bungee jumping szimulátor

Készítsen Inkrementális képszintézissel bungee jumping szimulátort. A 600x600-as képernyő két nézetre (viewport) van osztva, a bal oldaliban az ugró szempontjából, a jobb oldaliban egy keringő drón szempontjából követjük az ugrást. Az ugró egy téglatest, amely mindig a fejének megfelelő lap irányába néz. Az ugró véletlen kezdősebességgel lép le a láthatatlan platformról, amikor a felhasználó bármelyik billentyűt megnyomja. A terep 1/f zaj, amely diffúz/spekuláris, a diffúz visszaverődési együttható a terep magasság függvénye a térképekhez hasonlatosan. A kötél csak a nyugalmi hosszát túllépve fejt ki erőt, illetve forgatónyomatékot, mégpedig a test szimmetriasíkjában, ezért feltételezhető, hogy az ugró forgástengelye állandó. A kötél kirajzolása nem kötelező.  Az ugróra a haladó és forgó mozgás sebességével arányos közegellenállás érvényesül. A téglatest oldalhosszai (a, b, c), a tömeg m, a nehézségi gyorsulás g, a gumi rugóállandója D és nyugalmi hossza l0 úgy választandó meg, hogy a mozgás ízléses és realisztikus legyen.  A tehetetlenségi nyomaték a b éllel párhuzamos, középponton átmenő tengelyre m * (a * a + c * c) / 12. A virtuális világban a Newtoni dinamika szabályai érvényesülnek mind a haladó, mind pedig a forgó mozgásra.
