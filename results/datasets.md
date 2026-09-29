# Datasets

Written by `make data` (`src/afe/data/registry.py`). Source: OpenML-CC18 (suite 99), OpenML-CTR23 (suite 353) and the typed extra set. `cols` excludes the target, row id and ignored columns. `md5_file` is the md5 of the file as downloaded from OpenML; `frame_hash` is `pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame (target included). `date_typed_cols`: OpenML type `date`; `date_like_names`: a name regex, informational only.

Counts by task: {'binary': 15, 'multiclass': 6, 'regression': 14}. With categoricals: 14.

| key                           |   did |   version | task       |   rows |   cols |   n_categorical | source     | md5_file                         |           frame_hash | date_typed_cols   | date_like_names    |
|:------------------------------|------:|----------:|:-----------|-------:|-------:|----------------:|:-----------|:---------------------------------|---------------------:|:------------------|:-------------------|
| kr-vs-kp                      |     3 |         1 | binary     |   3196 |     36 |              36 | CC18       | 44a8cdccb27891b8014a3bdfb5f35813 |  5846383090484319539 |                   |                    |
| mfeat-fourier                 |    14 |         1 | multiclass |   2000 |     76 |               0 | CC18       | 00e0dd8627b970dbe47146461f67829d | 10580013087150151890 |                   |                    |
| mfeat-karhunen                |    16 |         1 | multiclass |   2000 |     64 |               0 | CC18       | 58481bc46e0ac18d6ce1f3fbd11dcdd5 | 12849214296312623294 |                   |                    |
| mfeat-morphological           |    18 |         1 | multiclass |   2000 |      6 |               0 | CC18       | ceac23dd55feacdd619c04ada5acaeb0 | 15502622657185770655 |                   |                    |
| mfeat-zernike                 |    22 |         1 | multiclass |   2000 |     47 |               0 | CC18       | e76a412ac4d0099849c145a471964c0f |  6556698415957084998 |                   |                    |
| cmc                           |    23 |         1 | multiclass |   1473 |      9 |               7 | CC18       | 1fe6aaf4995211a8e40bf2f910ddb758 |  4276284797694165826 |                   |                    |
| optdigits                     |    28 |         1 | multiclass |   5620 |     64 |               0 | CC18       | 11ef32e6b81278858541248f7f212af4 | 12643541705057182407 |                   |                    |
| credit-g                      |    31 |         1 | binary     |   1000 |     20 |              13 | CC18+typed | f20f4f62ce92921ab932b78312ff68d1 |  7606122327431373292 |                   |                    |
| sick                          |    38 |         1 | binary     |   3772 |     29 |              23 | CC18       | edbbcb7bdee005d1d044a98e9e20e5b8 | 15039467398525504561 |                   |                    |
| spambase                      |    44 |         1 | binary     |   4601 |     57 |               0 | CC18       | f52519460a6022e4c5b137745862b9a9 |  4540542590134708508 |                   |                    |
| electricity                   |   151 |         1 | binary     |  45312 |      8 |               1 | CC18       | 95b8fa75b51f5d242f0c981f10b72ebc |  6613769375656497642 |                   | date,day           |
| pc4                           |  1049 |         1 | binary     |   1458 |     37 |               0 | CC18       | 333f5139db5fa84abb02ef2637e415ba |  2784722723235312916 |                   | HALSTEAD_PROG_TIME |
| pc3                           |  1050 |         1 | binary     |   1563 |     37 |               0 | CC18       | e01b96587de46374b688150a0388d0df |  8743726978496065724 |                   | HALSTEAD_PROG_TIME |
| jm1                           |  1053 |         1 | binary     |  10885 |     21 |               0 | CC18       | 0efd5c21d9fc741794d03a2cd7bcf574 | 13030368134980865993 |                   |                    |
| kc1                           |  1067 |         1 | binary     |   2109 |     21 |               0 | CC18       | 5501f6ebf2525d136ec454a32514dd2c | 13550314665590438078 |                   |                    |
| pc1                           |  1068 |         1 | binary     |   1109 |     21 |               0 | CC18       | b826a65fe631728e63fc0a1070e88d3f |  6578379081200143584 |                   |                    |
| bank-marketing                |  1461 |         1 | binary     |  45211 |     16 |               9 | CC18+typed | 874b8e0e5d38605db3190ef8534ac3d4 |  2484976918774064709 |                   |                    |
| banknote-authentication       |  1462 |         1 | binary     |   1372 |      4 |               0 | CC18       | c01f837e3d57da29c67e0a998ce12e3e |  6036520945535955541 |                   |                    |
| Moneyball                     | 41021 |         2 | regression |   1232 |     14 |               6 | CTR23      | f5d2ea0a7ca33cb17468d04620173868 | 18371239449379540741 |                   | Year,RankSeason    |
| abalone                       | 44956 |        15 | regression |   4177 |      8 |               1 | CTR23      | cf60a46da12cedd673344dd889210f9d | 18418919520324962107 |                   |                    |
| airfoil_self_noise            | 44957 |         8 | regression |   1503 |      5 |               0 | CTR23      | 120eda8ce2be821c74df18af7288a9bf | 17945803456140043343 |                   |                    |
| auction_verification          | 44958 |         7 | regression |   2043 |      7 |               2 | CTR23      | ec666317ccd2287ba615f8a4a9db45bb |  4074807302068729782 |                   |                    |
| concrete_compressive_strength | 44959 |         7 | regression |   1030 |      8 |               0 | CTR23      | 3b3035c476267e7929a179e2cc5286c8 |  3757496306289002596 |                   |                    |
| physiochemical_protein        | 44963 |         7 | regression |  45730 |      9 |               0 | CTR23      | 27ac7ad8c2caaa0cf485bf697ecaa35f | 10566618383863273934 |                   |                    |
| superconductivity             | 44964 |         7 | regression |  21263 |     81 |               0 | CTR23      | 0575925b1fab9d146d86ab99fa0b5b6b | 11444629703213451326 |                   |                    |
| solar_flare                   | 44966 |         7 | regression |   1066 |     10 |               8 | CTR23      | 120cb704bcf1a7a5852f14d1f78b1445 |  3566118545390151923 |                   |                    |
| naval_propulsion_plant        | 44969 |         7 | regression |  11934 |     14 |               0 | CTR23      | af3274aaa2e43b0b9287633d58c27efc |  8241177867036765998 |                   |                    |
| white_wine                    | 44971 |         7 | regression |   4898 |     11 |               0 | CTR23      | eeaca690506cbc84554174a490f8e960 | 10563570576793314036 |                   |                    |
| red_wine                      | 44972 |        11 | regression |   1599 |     11 |               0 | CTR23      | 6fcbadfab0d0c458766f03c0c36290c8 |  1224293810254024155 |                   |                    |
| grid_stability                | 44973 |         7 | regression |  10000 |     12 |               0 | CTR23      | 168ac43d6de3a51525bb1ff52f396b44 |  6738102959384363058 |                   |                    |
| adult                         |  1590 |         2 | binary     |  48842 |     14 |               8 | typed      | 6f9af14fb61d70198b494ed54b7cf7b6 | 11220653127075058909 |                   | hours-per-week     |
| Amazon_employee_access        |  4135 |         1 | binary     |  32769 |      9 |               9 | typed      | 4b6066938515dc50b36eeeab3be13765 | 16508201442281791848 |                   |                    |
| Amazon_employee_access_int    |  4135 |         1 | binary     |  32769 |      9 |               0 | typed      | 4b6066938515dc50b36eeeab3be13765 |   508317086095130713 |                   |                    |
| diamonds                      | 42225 |         1 | regression |  53940 |      9 |               3 | typed      | 658eaf4009d25553284aaa211f88ccb7 | 12400245448291971708 |                   |                    |
| house_sales                   | 42092 |         2 | regression |  21613 |     19 |               2 | typed      | f9cdc8443cf2b57e0cb2903d895de2be |  2454219016414863087 |                   | date               |

## Selection log

- reject 6 letter (CC18): classes=26
- reject 11 balance-scale (CC18): rows=625
- reject 12 mfeat-factors (CC18): cols=216
- reject 15 breast-w (CC18): rows=699
- reject 29 credit-approval (CC18): rows=690
- reject 37 diabetes (CC18): rows=768
- reject 50 tic-tac-toe (CC18): rows=958
- reject 54 vehicle (CC18): rows=846
- reject 188 eucalyptus (CC18): rows=736
- reject 300 isolet (CC18): cols=617
- reject 307 vowel (CC18): rows=990
- reject 458 analcatdata_authorship (CC18): rows=841
- reject 469 analcatdata_dmft (CC18): rows=797
- reject 554 mnist_784 (CC18): cols=784
- reject 1063 kc2 (CC18): rows=522
- reject 1464 blood-transfusion-service-center (CC18): rows=748
- reject 1468 cnae-9 (CC18): cols=856
- reject 1478 har (CC18): cols=561
- reject 1480 ilpd (CC18): rows=583
- reject 1485 madelon (CC18): cols=500
- reject 1486 nomao (CC18): cols=118
- reject 1501 semeion (CC18): cols=256
- reject 1510 wdbc (CC18): rows=569
- reject 4134 Bioresponse (CC18): cols=1776
- reject 6332 cylinder-bands (CC18): rows=540
- reject 23381 dresses-sales (CC18): rows=500
- reject 40670 dna (CC18): cols=180
- reject 40923 Devnagari-Script (CC18): cols=1024
- reject 40927 CIFAR_10 (CC18): cols=3072
- reject 40978 Internet-Advertisements (CC18): cols=1558
- reject 40979 mfeat-pixel (CC18): cols=240
- reject 40994 climate-model-simulation-crashes (CC18): rows=540
- reject 40996 Fashion-MNIST (CC18): cols=784
- reject 44960 energy_efficiency (CTR23): rows=768
- reject 44962 forest_fires (CTR23): rows=517
- reject 44965 geographical_origin_of_music (CTR23): cols=116
- reject 44967 student_performance_por (CTR23): rows=649
- reject 44970 QSAR_fish_toxicity (CTR23): rows=908
- reject 44994 cars (CTR23): rows=804
- typed bank-marketing: already selected as 1461
- typed credit-g: already selected as 31
