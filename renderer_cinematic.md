# Plan: `plot.renderer = 'cinematic'` — path tracing w K3D

Stan: **etapy 0–7 zaliczone i zacommitowane (2026-08-19)**. Pełna suita:
**358 passed, 0 failed, 45:34** (trzy tryby; wzrost ~20 min wobec budżetu
+60 min). 161 scen cinematic ma referencje w 640×360 przy 32 samplach.
Poza planem zostają pozycje z §6a.
Fakty o bibliotece zweryfikowane 2026-08-18 (issues, README, package.json, npm).
Konwencja jak przy planie advanced: etapy z bramkami, miny wypisane jawnie,
decyzje autora oznaczone **[DECYZJA]**.

### Zrealizowane w etapie 1 (kontrakt + szkielet)

- Traity `cinematic_samples` (64, [1,4096]) i `cinematic_bounces` (6, [1,32])
  przez wszystkie punkty synchronizacji (plot_base + walidatory, factory,
  anywidget PLOT_HANDLERS + konstruktor, Core.parameters + settery,
  headless.html, plot_snapshot); kontrolki GUI widoczne tylko przy
  `renderer == 'cinematic'` (mechanika environmentControls).
- `setRenderer('cinematic')`: preflight `unsupportedReason()` → odmowa =
  overlay `Cinematic Error` z konkretnym powodem + trait wraca do wartości
  sprzed przełączenia (dropdown GUI wysyła prawdę PO setRenderer). Snapshot
  otwierany od razu w cinematic: bez trybu powrotu — render() melduje ten sam
  błąd, żadnego cichego fallbacku.
- Dispatch w `Renderer.render()`: cinematic omija peel/direct; RENDERED po
  osiągnięciu budżetu sampli (punkt synchronizacji headless); wyjście z trybu
  przerywa akumulację w locie (`abort()`) i chowa HUD.
- Pętla akumulacji przerywalna licznikiem generacji; `CAMERA_CHANGE`
  restartuje akumulację (rAF w cinematic stoi — klatka = cała akumulacja).
- HUD licznika sampli („cinematic: N / M samples", „building BVH…").
- Rozgrzewka RNG warunkowa (`needsWarmup` po setScene/zmianie bounces);
  hash determinizmu niezmieniony (`b190883f…`).
- Testy: walidatory traitów + mapa snapshotu (test_anywidget), legalność
  trybu i degradacja nieznanych nazw (test_advanced_renderer_state), smoke
  end-to-end w headless (przełączenie traitem, RENDERED, HUD, powrót).

### Zrealizowane w etapach 2–5 (commity c5b07d6, 08390af, 7e180c0)

- **Proxy sceny** (`sceneProxy.js`): wszystkie mapowania z tabeli §2 plus
  wypieki CPU dla ścieżek, których `Material.clone()` nie przenosi
  (mesh+volume, marching_cubes z atrybutem, texture/data). Cache per id,
  inwalidacja OBJECT_CHANGE/REMOVED, pełny zrzut na bezpayloadowym
  OBJECT_LOADED. Mina: `IcosahedronGeometry` jest zupą trójkątów bez indeksu
  (`mergeVertices` przed rozmnożeniem), a `matrixWorld` trzeba odświeżyć samemu
  (w cinematic nie ma pętli rasteryzującej, która to robiła).
- **Screenshoty**: akumulacja offscreen w dokładnej rozdzielczości docelowej
  (`synchronizeRenderSize=false` + własny `setSize`; `renderScale` zaokrągla).
  Tone mapping jednym blitem dla trzech trybów, `renderToCanvas=false` po
  stronie biblioteki. Miny osiadania: kamera helpera osi i overlaye DOM
  (`refreshGrid` + `BEFORE_RENDER`) nie siadają same bez pętli rasteryzującej —
  bez tego PIERWSZY screenshot różnił się od kolejnych.
- **Hybryda volume/MIP**: marsz cięty depth-passem proxy przez mechanizm
  segmentów #277 (MIP dostał ten sam clamp, uśpiony przy `uPeelSegment==0`),
  kompozycja premultiplied przed krzywą tonalną. Mina: `overrideMaterial` nie
  obejmuje kopuły środowiska — jej kolory wpadały do kanału głębi i marsz nie
  był cięty. Weryfikacja liczbowa z advanced jako oracle: czarna kula przy
  przedniej ścianie chmury 0.3 vs 80.1 dla marszu bez cięcia (oracle 2.3).
- **Koalescencja renderów** (kluczowa dla budżetu suity): `K3D.autoRendering`
  nigdy nie jest `true`, więc każdy setter woła render — przy rasteryzacji
  tanio, w cinematic to akumulacja per setter. Bez force: w przeglądarce jedna
  akumulacja na tick, w headless żadna (klatki wyłącznie na żądanie). Koszt
  sceny referencyjnej spadł z 34 s do ~18 s.

### Poprawki po testach autora (2026-08-19, commity 0e7dc27, 3c707f3)

- **Zacinanie strony i watchdog GPU**: `tiles.set(1,1)` + `gl.finish()` w pętli
  dawały jedno nieprzerwane zadanie GPU na próbkę. Tor interaktywny przepisany
  na rytm z przykładów biblioteki: **jeden `renderSample()` na klatkę rAF**,
  praca cięta kafelkami dobieranymi z liczby pikseli (~120k/kafelek), bez
  `finish()`. Pętla parkuje się po osiągnięciu budżetu; headless zostaje przy
  pętli do budżetu (determinizm + promise dla RENDERED).
- **Przerywanie**: zmiana kamery/obiektu/parametru natychmiast porzuca
  akumulację (bump generacji) i startuje od zera. `CAMERA_CHANGE` nie
  wystarcza — programowe `plot.camera` go nie emituje — więc pętla porównuje
  macierze kamery. Podczas ruchu rysowany jest **rasteryzowany podgląd**
  (odpowiednik `rasterizeScene`), inaczej ekran stoi do puszczenia myszy.
- **Porzucony plot**: pętla zatrzymuje się, gdy węzeł DOM przestaje być
  podłączony (przeliczenie komórki nie zawsze woła `disable()`).
- **Ekspozycja**: bez własnego gainu, dokładnie krzywa advanced — patrz
  osobne ustalenie w pamięci. Mediana jasności zgodna z simple/advanced
  co do <1%; jaśniejszy górny ogon to GI, sterowalne przez
  `cinematic_bounces` (2 → p90 183, 6 → 202) i `tone_mapping`.
- **`mesh_detail`** respektowany co do trójkąta (proxy buduje tę samą sferę,
  którą rasteryzator instancjonuje); budżet trójkątów podniesiony do 12M i
  tylko obniża detal, głośno.
- **Zmiany materiałowe** (roughness/metalness/opacity) nie przebudowują BVH —
  materiały proxy są resynchronizowane, tracer odświeża tylko swoją teksturę
  materiałów. Kolor zostaje przebudową (jest wypiekany w vertex colors).
- **`volume_slice` poza zakresem V1** (ostrzeżenie w konsoli + docs): maluje
  własną płaszczyznę przekroju i nie ma mechanizmu cięcia głębią.

### Wyniki bramki etapu 0 (spike, torus knot 20,8k tris + podłoga + env studio, 640×360, SwiftShader)

- **Budowanie:** pin `0.0.24` bundluje się; przyrost bundli +239 kB
  (standalone 3,30→3,54 MiB). Mina webpacka: reguła `test: /\.(glsl|txt)/`
  bez kotwicy `$` połykała `*.glsl.js` biblioteki — zakotwiczona.
- **Wydajność:** prepare (scena+BVH) ~0,2 s; pierwsza próbka ~4,9 s (kompilacja
  shaderów, raz na sesję); potem ~0,5–0,7 s/sample. 32 sample ≈ 21–25 s/test →
  pełna suita mieści się w budżecie +60 min.
- **Determinizm: bitowy**, w sesji i między świeżymi kontenerami (md5
  identyczne, diff 0 px). Wymagał trzech warstw:
  1. `stableNoise = true` (seedowany LCG jittera strat, `material.seed=0`
     przy reset);
  2. **rozgrzewka jedną próbką + `reset()` przed każdą mierzoną akumulacją**
     (pierwsza próbka po `setScene` przebudowuje `StratifiedSamplesTexture`
     i tasuje straty konsumując LCG; `reset()` przywraca porządek
     tożsamościowy bez tasowania — bez rozgrzewki pierwsza akumulacja ma
     inną sekwencję niż każda następna);
  3. **seedowana regeneracja `stratifiedOffsetTexture`** (blue noise per
     piksel losowany `Math.random` przy konstrukcji materiału, nigdy nie
     resetowany — jedyne źródło różnic między załadowaniami strony);
     deep import `BlueNoiseGenerator` + LCG o stałych GCC, seed 1,
     w `webglBackend.reseedOffsetTexture` z guardem rzucającym przy bumpie
     biblioteki.
- Wniosek do §5: `max_mismatched_pixels` dla cinematic zostaje **0**.

---

## 0. Decyzje wiążące i kontekst

- **Biblioteka:** `three-gpu-pathtracer`, **pin dokładny `0.0.24`** (nie `^`).
  Powody: (a) issue **#779** (2026-07-09, gkjohnson): *„WebGLPathTracer will be
  deprecated and removed in an upcoming release once the WebGPUPathTracer is
  ready (#713) […] WebGL will no longer be supported either via three.js'
  WebGLRenderer nor WebGPURenderer with the webgl backend"* — czyli gałąź WebGL
  jest już dziś **maintenance-frozen**, a ostatnie wydanie przed usunięciem
  będzie naszym pinem na lata; (b) pakiet jest 0.0.x z 20-miesięczną przerwą
  między wydaniami — API nie daje gwarancji.
- **Zgodność:** peer `three >= 0.180` i `three-mesh-bvh >= 0.7.4`; my mamy
  three 0.185.1 i three-mesh-bvh 0.9.14 (ich dev testuje na 0.9.5) — **pasuje
  bez podbić**.
- **WebGPU:** issue **#777** (milestone v0.0.25, commity na gałęzi do
  2026-08-17) — pierwszopartyjny `WebGPUPathTracer` (architektura wavefront
  compute) powstaje w tej samej bibliotece. Migracja k3d→WebGPU będzie więc
  najpewniej **podmianą backendu w tym samym pakiecie**, nie zmianą biblioteki.
  Braki w tierze „Initial Release" (światła punktowe/powierzchniowe, IES,
  walidacja IOR, wydajność) → nie zakładać parytetu w pierwszym wydaniu.
- **Złota zasada trybów** („renderer zmienia światło, nie to, co narysowałeś")
  w cinematic dostaje doprecyzowanie: zachowujemy **kształt**, nie
  implementację. Kula punktu zostaje kulą, linia linią — ale impostor staje
  się siatką, a wstążka ekranowa rurką o szerokości światowej. Różnice
  reprezentacji wypisane w §2 i w docs.
- **[DECYZJA autora, przyjęta]** wszystkie testy wizualne obejmują trzeci tryb;
  refy cinematic w **640×360** (ćwierć pikseli 1280×720).
- Nic z tego planu nie commitujemy do czasu akceptacji.

## 1. Architektura izolacji — kontrakt backendu (WebGPU-ready)

Nowy katalog `js/src/providers/threejs/initializers/cinematic/`:

```
cinematic/
  index.js          – orkiestracja trybu (proxy sceny, pętla, kompozycja)
  webglBackend.js   – JEDYNE miejsce importujące three-gpu-pathtracer
  sceneProxy.js     – „wszystko meshem" (§2)
  volumeHybrid.js   – kompozycja wolumenów (§4)
```

Kontrakt backendu (duck-typing, bez typów biblioteki na granicy):

```js
backend = createWebGLBackend(renderer)   // przyszłość: createWebGPUBackend(...)
backend.isSupported()                    // WebGL2 + floaty; false => fallback
backend.setScene(proxyScene, camera)     // pełny rebuild BVH (async przez workera)
backend.updateCamera() / updateEnvironment() / updateMaterials()
backend.renderSample()                   // 1 sample; zwraca {samples, texture}
backend.setSize(w, h) / reset() / dispose()
```

Twarde zakazy (checklista WebGPU-readiness, egzekwowana w review):
1. importy `three-gpu-pathtracer` wyłącznie w `webglBackend.js`;
2. na granicy kontraktu żadnych obiektów WebGL (target wychodzi jako
   `THREE.Texture` do naszego blita kompozycji);
3. proxy sceny = czyste `THREE.Mesh` + `MeshStandardMaterial`/`Physical`
   (dokładnie to, co wspiera i WebGL-owy, i przyszły WebGPU-owy tracer);
4. tone mapping, GUI, screenshoty — nasze, poza biblioteką;
5. `isSupported() === false` ⇒ **błąd, nie fallback**: `setRenderer('cinematic')`
   zgłasza overlay istniejącym mechanizmem `core/lib/Error.js` (jak „Loader
   Error") z konkretnym powodem (brak WebGL2 / floatów / padnięta inicjalizacja
   biblioteki), tryb się NIE przełącza — plot zostaje w dotychczasowym
   rendererze, a trait `renderer` wraca do poprzedniej wartości przez
   `PARAMETERS_CHANGE`, żeby stan kernela i GUI zgadzał się z rzeczywistością.

Mapowanie na API 0.0.24 (w `webglBackend.js`): `new WebGLPathTracer(renderer)`,
`setBVHWorker(new ParallelMeshBVHWorker())` + `setSceneAsync` (BVH poza głównym
wątkiem), `renderSample()`, `reset()`, `tiles` (responsywność), `renderScale`,
`filterGlossyFactor` (fireflies), `minSamples`, readonly `target`.

## 2. Proxy sceny — „wszystko meshem"

Zasada: **K3DObjects pozostaje źródłem prawdy i nie jest modyfikowane.**
Cinematic utrzymuje równoległą grupę `proxyScene`; oryginał nie jest renderowany
w tym trybie. Python/JS API bez zmian — użytkownik dalej ma `shader='3d'`.

Mapowania per typ (zweryfikowane w kodzie 2026-08-12):

| typ | reprezentacja cinematic | uwagi |
|---|---|---|
| mesh (standard), stl, surface, marching_cubes (plain), texture (image) | passthrough | klon Mesh + sanityzacja materiału (strip onBeforeCompile/NoBlending gałęzi depth-peel, przywrócenie transparent/depthWrite regułą depthPeels==0); texture/image to MeshBasic (unlit) → Standard z mapą, różnica do docs |
| voxels, voxels_group, sparse_voxels | passthrough chunków | chunki = Standard + vertexColors, klon czysty; ODFILTROWAĆ outline'y MeshLine (ShaderMaterial bez .color — filtr materiałowy je łapie) i rollOverMesh (visible=false) |
| mesh (volume), marching_cubes (attribute), texture (data) | wypiek kolorów CPU | ShaderMaterial / expando-uniformy + onBeforeCompile — **clone gubi je po cichu**; kolory per wierzchołek (próbka Data3DTexture) lub wypieczona tekstura RGBA z color_map+opacity_function w json |
| points (dot/flat/3d/3dSpecular/mesh) | ikosfery **zmergowane do jednej geometrii** | instancje NIEwspierane; '3dSpecular'→'3d' (alias w Points.js); SEMANTYKA SIZES: billboardy = średnice absolutne (sizes[i] zamiast point_size), mesh = mnożniki point_size; dot nie ma rozmiaru światowego (piksele!) — przyjmujemy średnicę = point_size, do docs; kolormapa liczona w shaderze → wypiek CPU z color_map; per-point opacities poza V1 (do docs); detal: subdiv 3 ≤1k pkt, 2 ≤20k, 1 ≤200k, 0 powyżej; budżet ~2M tris → warn + degradacja |
| line/lines shader='mesh' | passthrough | to JUŻ są rurki Standard (Streamline, width=PROMIEŃ) |
| line/lines simple | rurka Streamline, promień = `width` | Streamline natywnie tnie na separatorach NaN (line); lines: krawędzie z indices (guardIndices + dedup nieskierowany, jump 2/3 wg indices_type) — reużyć userData.edgeVertices |
| line/lines thick | rurka Streamline, promień = `width/2` | thick ekstruduje PEŁNĄ szerokość w clip-space; mesh używa width jako PROMIENIA — różnica 2× celowo zachowana per wariant |
| vectors / vector_field | trzony → rurki (promień=line_width/2), groty → klon geometrii stożków + Standard({vertexColors}) | trzony to MeshLine (ShaderMaterial bez .color); groty mają już normal+color, wystarczy podmiana MeshBasic→Standard; trzon skrócony o 0.2·head_size (VectorField: ·scalar); UWAGA: VectorField buduje model_matrix przez fromArray (column-major), Vectors przez .set (row-major) — powielić, nie „naprawiać"; grupa VF ma offset (-0.5,-0.5,-0.5\|0) |
| texture_text | kwady zwrócone do kamery w prepare() | to THREE.Sprite'y (NIE passthrough — korekta planu po rozpoznaniu 2026-08-18); canvas jako mapa + alphaTest, orientacja zamrożona w momencie prepare, do docs |
| text, text2d | poza path tracingiem | czysty DOM overlay, nigdy nie trafiają do K3DObjects (guard w Core.addOrUpdateObject) |
| label | POMINĄĆ w proxy | jest w K3DObjects jako LineSegments, ale drugi wierzchołek leadera mutowany per klatkę od kamery — nie do statycznego BVH |
| volume, mip | §4 | |

Fakty do inwalidacji per obiekt (rozpoznanie 2026-08-18): `OBJECT_CHANGE`
niesie `{id, key, value}`, `OBJECT_REMOVED` niesie samo id, ale
`OBJECT_LOADED` jest BEZ payloadu — cache proxy per String(id) + diff po
`world.ObjectsById`; tożsamość instancji niestabilna (addOrUpdateObject
podmienia obiekt pod tym samym id bez OBJECT_REMOVED); `reload` z
`visible=false` zdejmuje obiekt ze sceny też bez OBJECT_REMOVED. Dispatch
buildera po `world.ObjectsListJson[id].type` + `.shader`; transform proxy =
`matrixWorld` źródłowego obiektu/grupy (łapie initialPosition typu Surface
(-0.5,-0.5,0) i skalę voxeli).

Miny biblioteki do obsłużenia w proxy (potwierdzone w źródłach):
- `MaterialsTexture` czyta `material.color.r` bez guardu — **filtr sceny po
  `material.color !== undefined`**, nie po typie obiektu;
- `getLights()` pomija `AmbientLight` — światło ambientowe zastępuje kopuła
  środowiskowa (i tak jest naszym jedynym światłem w advanced);
- wymóg: wszystkie tekstury z tymi samymi flagami wrap/interpolation —
  normalizacja we własnych klonach tekstur;
- interleaved buffers niewspierane — sprawdzić nasze geometrie (three 0.185
  defaulty są nie-interleaved, ale np. Text3D/zewnętrzne STL zweryfikować).

Aktualizacje: subskrypcje `OBJECT_LOADED/REMOVED/CHANGE` → inwalidacja proxy
tylko dotkniętego obiektu → `setSceneAsync` (rebuild BVH w workerze; overlay
„rebuilding…" przy liczniku sampli). Strategie in-place z rasteryzacji tu nie
obowiązują — każda zmiana geometrii to rebuild BVH; zmiany materiałowe idą przez
`updateMaterials()` bez rebuildu.

## 3. Pętla renderowania i integracja z resztą K3D

- `Renderer.js render()`: trzeci dispatch — cinematic omija depthPeel/direct;
  pętla `renderSample()` w rAF przy `auto_rendering`, do budżetu przy
  screenshotach;
- kamera: `CAMERA_CHANGE` → `updateCamera()` (biblioteka sama resetuje
  akumulację); zoom/resize → `setSize` + `reset`;
- **tone mapping nasz**: `target` biblioteki blitowany naszym composite
  (uToneMapping agx/aces jak dotąd) — jedna ścieżka dla trzech trybów;
- GUI: licznik sampli + spinner BVH w rogu (jak fps meter); **[PRZYJĘTE]**
  nowe traity plota: `cinematic_samples` (budżet dla screenshot/headless,
  default 64) i `cinematic_bounces` (default 6, mapowany na `bounces`) —
  **kontrolki widoczne w panelu tylko przy `renderer == 'cinematic'`**
  (mechanika jak `environmentControls` w Core.js, które już dziś chowają się
  poza advanced: lista kontrolek pokazywana/ukrywana w setRenderer);
- screenshot/`get_screenshot`: pętla sampli do budżetu, potem odczyt —
  `screenshot_scale` przez `renderScale`; SSAA chunkowane NIE dotyczy trybu;
- headless: `RENDERED` dispatchowane po osiągnięciu budżetu sampli —
  deterministyczny punkt synchronizacji dla suity;
- snapshoty standalone: działa out of the box (biblioteka w bundlu) — zmierzyć
  przyrost bundla w etapie 0 (szacunek: +300–500 kB min.; wpływ na wheel).

## 4. Volume i MIP w cinematic (od razu, decyzja autora)

Fakty: jedyna wolumetryka biblioteki to `FogVolumeMaterial` (gęstość
**jednorodna**) — heterogeniczna siatka 3D jest poza zakresem WebGL-owego
tracera i (na dziś) poza tierem „Initial Release" WebGPU (#777 wymienia „volume
stacks" w tierze Future).

**V1 — hybryda depth-aware (wdrażana teraz):**
1. własny rasterowy depth pass proxy sceny (deterministyczny, tani — mamy całą
   maszynerię z AO);
2. nasz istniejący marsz wolumenu (identyczny shader co advanced, ze światłem
   SH + L1 z env) tnie promień na segment kamera→pierwsze trafienie geometrii;
3. kompozycja: `L = T_vol · L_pathtraced + L_vol` (marsz daje premultiplied
   radiance + transmitancję) — nakładana po każdej porcji sampli, więc podgląd
   progresywny zawiera wolumen od pierwszej klatki.

Ograniczenia V1 (do docs, sekcja renderers):
- wolumen nie występuje w odbiciach/refrakcjach ani nie rzuca cienia na
  geometrię (GI „nie widzi" gazu);
- geometria ZA wolumenem jest tłumiona transmitancją poprawnie, ale geometria
  widziana PRZEZ odbicie nie;
- to świadomy kompromis: dla powierzchni GI robi robotę, dla gazu marsz z
  advanced jest wizualnie identyczny jak dotąd.

**MIP:** z definicji nie-fizyczny (maksimum po promieniu, projekcja
diagnostyczna) — w cinematic renderowany dokładnie jak w advanced i komponowany
jak wolumen w V1. Zapisać w docs: „MIP w cinematic = warstwa diagnostyczna,
poza GI".

**V2 — natywne MC (osobny etap badawczy, PO migracji WebGPU):** delta tracking
heterogeniczny wymagałby forka zamrożonego WebGL-owego tracera — nie
inwestujemy (#779). Właściwy moment to backend WebGPU: wzorzec licencyjnie
bezpieczny istnieje (webgpu-volume-pathtracer Ushera, MIT; Grenzwert, MIT;
VPT jest GPL-3 — tylko do czytania). Wtedy wolumetryczne MC wchodzi jako nasz
moduł obok `WebGPUPathTracer`.

## 5. Testy i referencje

- `plot_compare.compare()`: `modes=('simple', 'advanced', 'cinematic')`
  domyślnie; istniejący parametr `modes` zostaje jako wyjątek per test.
- **Rozdzielczość cinematic: 640×360** — w pętli trybów compare przełącza
  rozmiar okna headless przed trybem cinematic i wraca po nim; refy w
  `k3d/test/references/cinematic/` w tej rozdzielczości.
- Determinizm: **rozstrzygnięty w etapie 0 — bitowy** (trzy warstwy, patrz
  wyniki bramki na górze). `max_mismatched_pixels` dla cinematic = 0.
- Budżet refów: `REF_SAMPLES = 32`, `bounces = 6`, `tiles = 1×1` (headless nie
  potrzebuje responsywności). Kalibracja w etapie 0: zmierzyć ms/sample na
  SwiftShader dla sceny średniej (np. mesh_advanced); budżet **[PRZYJĘTE]**:
  suita może urosnąć o **≤ 60 min**, czyli ~45 s na test cinematic. Jeśli
  pomiar wyjdzie gorzej: REF_SAMPLES w dół przed cięciem zakresu testów
  (zakres = decyzja autora, domyślnie wszystkie).
- Generacja refów: istniejący accept-loop (uruchomienie z akceptacją, wizualna
  inspekcja próby, commit refów osobnym commitem).
- pixelmatch: threshold 0.2 jak dotąd; `max_mismatched_pixels` dla cinematic
  skalibrować po pierwszej pełnej generacji (start: 0, poluzować tylko z dowodem
  niedeterminizmu).

## 6. Etapy (każdy z bramką)

0. **Spike wykonalności** — dep `==0.0.24`; przyrost bundla standalone+widget;
   minimalna scena (mesh + env `studio`) przez `webglBackend` w headless;
   pomiar ms/sample @640×360 SwiftShader; test determinizmu (dwa przebiegi,
   diff bitowy); werdykt budżetów. *Bramka: obraz zbieżny, powtarzalny,
   koszt suity policzalny.*
1. **Kontrakt + szkielet** — `cinematic/`, dispatch w Renderer.js, fallback
   detect, licznik sampli w GUI. Test dymny nie-wizualny.
2. **Proxy meshowe** — passthrough + merged-ikosfery punktów + rurki linii;
   filtr `material.color`, normalizacja tekstur. Testy wizualne: po jednej
   scenie na mapowanie (na razie refy tymczasowe, poza suitą).
3. **Pętla i integracja** — progresja, kamera, reset, screenshot/headless
   z budżetem sampli, tone mapping wspólnym blitem.
4. **Środowiska** — `scene.environment` z naszych map (proceduralne i
   fotograficzne; bez `BlurredEnvMapGenerator` w refach — wierność ponad
   zbieżność, blur ewentualnie jako opcja użytkownika), `lighting` = ekspozycja.
5. **Volume/MIP V1** — depth pass + kompozycja hybrydowa; wariant cinematic
   testu kompozycji #277.
6. **Suita** — trio trybów, 640×360, budżety; pełna generacja refów
   (accept-loop); raport czasu CI.
7. **Docs** — sekcja cinematic w renderers.rst (przez tabelą różnic reprezentacji
   i ograniczeń V1), interaktywne osadzenie (`k3d_plot` z małym budżetem
   sampli), wpis CHANGELOG. Przykład galeriowy **[PRZYJĘTE]** dopiero po
   ustabilizowaniu refów (osobny etap, poza 2.20-owym wdrożeniem trybu).
8. **Nasłuch WebGPU** — śledzić v0.0.25/#777/#713; gdy `WebGPUPathTracer`
   wyjdzie: spike `webgpuBackend.js` za tym samym kontraktem (okres podwójnej
   zależności: pin starego wydania dla WebGL + nowe dla WebGPU, wybór przez
   feature detect). Wolumetryczne MC (V2) dopiero wtedy.

## 6a. Znane, jeszcze nieadresowane (z adwersarialnego przeglądu kodu)

Przegląd 2026-08-18 (5 obiektywów, 44 zgłoszenia, 6 potwierdzonych werdyktem —
reszta weryfikacji padła na limicie). Naprawione: uScreenSize, try/finally w
warstwie wolumenu, BEFORE_RENDER w screenshotach, limit bezczynności pętli,
prolog w łańcuchu obietnic, `stale` w screenshotach, releaseFixedSize, pixelRatio
w blicie. Zostaje:

- **Brak `dispose()` w proxy** (high): przy inwalidacji cache porzucane są
  zmergowane geometrie, wypieczone DataTexture i sklonowane mapy — GPU ich nie
  zwalnia, więc edycja sceny w pętli wycieka pamięcią. Do zrobienia: jawne
  zwalnianie zasobów wpisu przed usunięciem z cache (uwaga: geometrie
  passthrough są WSPÓŁDZIELONE ze sceną rasteryzowaną — wolno zwalniać tylko to,
  co proxy samo utworzyło).
- **Wyciek tekstury środowiska** (medium): `getEnvironmentTexture` tworzy nową
  `DataTexture` przy każdym `buildScene`, stara nie jest zwalniana, a zmiana
  tożsamości tekstury zmusza bibliotekę do przeliczania CDF importance samplingu
  przy każdej przebudowie sceny. Wzorzec do naśladowania: `environmentSource`
  w `Scene.js`.
- **`TextureText` zamrożony do kamery z ostatniej zmiany sceny** (high wg
  przeglądu): proxy przebudowuje billboardy tylko przy `sceneDirty`, więc ta sama
  końcowa kamera może dać dwa różne obrazy zależnie od historii.
- **`usesColorMap` nie sprawdza długości `attribute`** (medium): za krótki
  atrybut daje NaN w kolorach wierzchołków, a jeden NaN trwale zatruwa piksele
  akumulacji.
- **Brak limitu geometrii w `buildLines`** (medium): odpowiednik budżetu
  trójkątów z punktów, dla scen z setkami tysięcy krawędzi.

## 7. Ryzyka i niewiadome

| ryzyko | mitygacja / bramka |
|---|---|
| ms/sample na SwiftShader zabija suitę | etap 0 mierzy; REF_SAMPLES↓, dopiero potem rozmowa o zakresie testów |
| niedeterminizm sampli | etap 0; patch seedu w backendzie albo budżet pikseli |
| bundle size (wheel, snapshoty) | pomiar w etapie 0; biblioteka jest ESM/sideEffects:false — tree-shaking pomoże |
| chmury punktów 10⁶+ po merge | budżet trójkątów + degradacja detalu + warn w konsoli |
| unifikacja wrap/interp zmienia wygląd tekstur | normalizacja per-klon + test wizualny texture |
| duże BVH: pamięć + czas rebuildu przy update'ach | `setSceneAsync` + worker; inwalidacja per obiekt; komunikat w GUI |
| biblioteka WebGL zamrożona (#779) | pin ==0.0.24; bugi WebGL łatamy lokalnie (patch-package) zamiast czekać na upstream |
| WebGPU tier 1 bez części świateł | nas nie dotyka — świecimy wyłącznie środowiskiem |

## 8. Decyzje autora (2026-08-18)

1. **Traity `cinematic_samples` / `cinematic_bounces`: TAK**, kontrolki
   w GUI widoczne wyłącznie przy `renderer == 'cinematic'`.
2. **Bez fallbacku**: gdy przeglądarka nie udźwignie path tracera (brak
   WebGL2/floatów, padnięta inicjalizacja, w przyszłości brak WebGPU),
   przełączenie na `cinematic` kończy się **błędem** komunikowanym overlayem
   (`core/lib/Error.js`); tryb się nie zmienia, trait `renderer` wraca do
   poprzedniej wartości. Żadnego cichego renderowania „jak advanced".
3. **Budżet CI: +60 min** na trzeci tryb w pełnej suicie.
4. **Przykład galeriowy: po stabilizacji refów** — nie wchodzi w pierwsze
   wdrożenie trybu.
