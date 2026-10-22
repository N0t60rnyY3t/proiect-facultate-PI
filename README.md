# proiect-facultate-PI
Proiect Prelucrarea Imaginilor - Multi-Organ Segmentation

Nr. 1

Autor(i)/An
S. M. Rolfe, S. M. Whikehart, A. M. Maga / 2023

Titlul articolului/proiectului
“Deep learning enabled multi-organ segmentation of mouse embryos”. MEMOS (Mouse Embryo Multi-Organ Segmentation)

Aplicație/Domeniu
imagistică biomedicală (micro-CT 3D), genetică funcțională și fenotipare la șoarece, biologia dezvoltării. 
Aplicatie: segmentarea automată a 50 de structuri anatomice la embrioni de șoarece, pentru a măsura volumul și forma organelor și a detecta anomalii în liniile knockout.

Tehnologii utilizate
scanări micro-CT cu contrast de iod (IMPC, rezoluție joasă, ~250×250×400 voxeli); ANTs/ANTsR pentru segmentarea bazată pe atlas (ABM), folosită la generarea etichetelor; arhitectura UNETR în PyTorch și MONAI; AdamW (learning rate 0,0001, 80.000 iterații); pierdere Dice + entropie încrucișată; server cu 512 GB RAM și GPU NVIDIA A6000; extensie pentru 3D Slicer (VTK, ITK), cod open source pe GitHub (SlicerMorph/SlicerMEMOS). Encoder cu 12 straturi de transformere, embedding K=768, patch-uri de 16, inferență cu fereastră glisantă 128³ și suprapunere 0,8.

Metodologie/Abordare
91 de scanări baseline înregistrate (rigid, apoi deformabil) pe atlasul KOMP2 (35 de exemplare C57BL/6, segmentat manual în ~400 h); atlasul a fost proiectat pe fiecare specimen prin transformarea inversă, rezultând „ground truth” generat prin ABM, nu manual. Împărțire: 73 antrenare, 18 validare, 5 test. Antrenare de la zero, pe volume 128³ eșantionate aleatoriu, cu augmentare (rotație, afină, intensitate, zgomot gaussian). Evaluare: Dice față de ABM, timp de rulare pe trei sisteme, sensibilitate pe linia knockout Cbx4 (n=8 vs. n=11, test t Student, prag 0,05) și comparație ABM vs. MEMOS vs. segmentare manuală pe 8 structuri (22 baseline + 8 Cbx4). Modelul e integrat în 3D Slicer, cu revizuire și editare manuală.

Rezultate
Dice mediu 0,89 pe validare și 0,91 pe test; 48 din 50 de segmente au Dice > 0,8; cele mai slabe sunt vertebrele (0,64) și ventriculul drept (0,63). Timp pe specimen: 21,9 s pe server cu GPU (×6575 față de manual ~40 h, ×986 față de ABM ~6 h); 412 s pe același server fără GPU (×350 și ×52); 1719,9 s (~28,7 min) pe desktop (ABM nerulat). Pe Cbx4, scădere semnificativă a volumului pentru suprarenala dreaptă (P=0,001) și rudimentele timice stâng și drept (P<0,001); suprarenala stângă nesemnificativă (P=0,985). Pe un mutant, Dice față de manual pentru suprarenala stângă: 0,818 la MEMOS vs. 0,591 la ABM. Față de manual, pe baseline nu există diferențe semnificative între ABM și MEMOS; la knockout singura diferență e rudimentul timic drept (P=0,021), unde ABM a fost mai apropiat.

Limitari
etichete de antrenare derivate din ABM (modelul moștenește erorile și părtinirea acestuia); părtinire spre anatomia normală, deoarece s-a antrenat doar pe baseline; memoria GPU a impus rezoluție joasă și patch-uri aleatorii; performanță scăzută pe structuri subțiri (~2 voxeli lățime); ușoară sub-segmentare (vertebre, lob accesoriu, ventricul drept) și mici regiuni de supra-segmentare deconectate; antrenat doar pe date KOMP2 la E15.5, alte stadii sau protocoale necesită reantrenare; nu e complet automat, pentru analiză cantitativă se recomandă editare manuală; eșantioane mici (test n=5, Cbx4 n=8 vs. 11); personalizarea modelului cere Python și server cu GPU.

Comentarii suplimentare
autorii îl prezintă drept primul model preantrenat public pentru segmentarea embrionilor de șoarece; accesibilitatea (fără programare, rulabil pe desktop) e punct central. Dice 0,91 măsoară concordanța cu ABM, nu cu adevărul biologic; comparația cu segmentarea manuală (doar 8 structuri) e cea care susține echivalența cu ABM. Suprarenala stângă (P=0,985) e nesemnificativă deși efectul e așteptat bilateral, iar autorii nu o explică pe deplin. Nu se menționează corecție pentru comparații multiple. Propunerile din Discussion (mai multă memorie GPU, mutanți în antrenare, fine-tuning pe organ unic) sunt netestate. Finanțare NIH (OD032627, HD104435), fără conflicte de interese; date și cod publice.


Nr. 2

Titlul articolului: AutoProSAM: Automated Prompting SAM for 3D Multi-Organ Segmentation (Li, Sultan, Khanduri, Qiang, Indrin, Zhu; Wayne State University și Cedars Sinai Medical Center; WACV)/ 2025

Aplicația/domeniul: imagistică medicală (CT 3D), modele de bază (foundation models) pentru viziune computerizată, adaptare parameter-efficient. Aplicație: segmentarea automată 3D multi-organ pe CT, fără prompturi manuale de la experți (medici), pornind de la modelul SAM (Segment Anything Model).

Tehnologii utilizate: SAM (encoder ViT, prompt encoder, mask decoder) adaptat de la 2D la 3D; Depth Adapter (proiecție down/up cu convoluție 3D depth-wise, X + Act(X·W_down)·W_up); Auto Prompt Generator (APG), encoder-decoder FCN în stil 3D UNet, antrenat de la zero; mask decoder cu convoluții 3D și mecanism MLAM (multi-layer aggregation, hărți din stagiile 1-4 ale encoderului); atenție cu fereastră glisantă în stil SwinUNETR; MONAI 1.2, PyTorch 2.0; 8 GPU-uri NVIDIA A100; AdamW, learning rate 5e-4, momentum 0,9, decay 1e-5, 200 de epoci (5 de warm-up), batch 1/GPU, patch-uri 128³; pierdere Dice + entropie încrucișată; metrici Dice și NSD (Normalized Surface Distance). Datasete: BTCV, AMOS (CT, plus MRI în extensie), CT-ORG, un set pelvian instituțional privat, MM-WHS CT.

Metodologie/abordare: poziționale: tabel de encodare pozițională 2D înghețat (h, w) plus unul nou antrenabil pe adâncime (d), inițializat cu zero. Patch embedding: convoluție 1×k×k (greutăți 2D preantrenate, înghețate) plus k×1×1 depth-wise nou (k=14). Atenția: query-uri [B, HW, C] extinse la [B, DHW, C], cu reutilizarea greutăților preantrenate. Bottleneck cu convoluții 3D antrenate de la zero. În antrenare se ajustează doar convoluțiile, adapterele de adâncime și straturile de normalizare; restul e înghețat. APG ia harta de trăsături de la ultimul bloc de atenție + adapter și o folosește în locul prompturilor manuale (puncte/casete), evitând și supra-netezirea cross-attention-ului în 3D. Preprocesare per dataset (spacing fix, intervale HU, normalizare), eșantionare 1:1 de patch-uri pozitive/negative, augmentare (flip, rotație, scalare și deplasare de intensitate), inferență cu fereastră glisantă (suprapunere 0,75). Baseline-urile antrenate de la zero cu setările implicite. Împărțiri: BTCV 24 antrenare/6 test; AMOS 200/100; CT-ORG 81/19; pelvian 225/30/45. Comparații cu SOTA (nnUNet, AttUNet, UNet++, nnFormer, UNETR, SwinUNETR) și cu metode bazate pe SAM (SAM, MedSAM P&M și Full), plus ablații (APG, MLAM), cost computațional și test pe MRI/inimă.

Rezultate: BTCV: Dice 87,15 / NSD 78,83 (cel mai bun SOTA anterior: SwinUNETR 86,58 / 75,26). AMOS: 88,65 / 79,41 (nnUNet 87,43 / 77,12). CT-ORG: Dice 85,12 (nnUNet 85,51 e ușor mai bun), NSD 76,37 cel mai mare. Pelvian: 91,30 / 84,35. Semnificativ (p < 0,01, Wilcoxon) în majoritatea cazurilor, dar nu pe BTCV Dice, nici pe CT-ORG Dice. Față de metode SAM: SAM fără fine-tuning 54,86 (BTCV) și 49,31 (AMOS); MedSAM Full 84,57/73,76 și 83,10/70,26; AutoProSAM depășește MedSAM Full pe ambele. Ablație pe BTCV: fără APG și fără MLAM 81,14 / 70,33; fără APG 85,23; fără MLAM 84,37; cu ambele 87,15. Cost: 26,53 M parametri ajustați (cei mai puțini), 7,06 s/scan (nnUNet 3,74 s, SwinUNETR 4,24 s, UNETR 58,93 s, nnFormer 11,73 s). AMOS MRI: 80,8 Dice; MM-WHS CT (inimă): 90,9.

Limitări: textul nu are o secțiune dedicată de limitări; cele de mai jos sunt deduse din conținut. Inferență mai lentă decât nnUNet și SwinUNETR. Pe CT-ORG, Dice nu e cel mai bun. Dimensiuni mici ale seturilor (BTCV 30 de subiecți). Setul pelvian e privat, deci neverificabil. Evaluare centrată pe CT abdominal/pelvian (MRI și inimă doar într-un test scurt). Numărul de parametri ajustați e marcat ca variabil în funcție de implementare. Comparația cu alte metode SAM e restrânsă (altele excluse pe motive de dimensiune, date sau 2D).

Comentarii suplimentare: autorii susțin că multe lucrări bazate pe SAM folosesc adevărul de teren (ground truth) la inferență pentru a genera prompturi, ceea ce dă un avantaj incorect; APG nu depinde de așa ceva. Tabelul 4 din text spune „Multi-Level Attention Mechanism”, în timp ce restul textului folosește „Multi-Layer Aggregation Mechanism”, o inconsecvență de denumire. Afirmația „doar 3% / 14% îmbunătățire” e formulată ca „up to”, deci reprezintă maximul față de cel mai slab competitor, nu față de cel mai bun. Valoarea 91,30 pe setul pelvian depășește net celelalte metode, dar e pe un set privat. Cod disponibil (link menționat în abstract).



Autor(i)/An
