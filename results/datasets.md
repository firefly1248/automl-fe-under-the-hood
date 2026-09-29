# Datasets

Written by `make data` (`src/afe/data/registry.py`). Pool: OpenML suites 99 (CC18), 353 (CTR23), 271 (AMLB-clf), 269 (AMLB-reg), 334 (TB-cat-clf), 335 (TB-cat-reg), plus the typed extra set. `cols` excludes the target, row id and ignored columns. `primary`: in the primary aggregate (one dataset per family). `max_cat_levels`: most levels of any categorical column. `dup_frac`: share of exact duplicate rows (target included). `md5_file`: md5 of the file as downloaded from OpenML; `frame_hash`: `pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame. `date_typed_cols`: OpenML type `date`; `date_like_names`: a name regex, informational only.

Primary: 50 datasets, by task {'binary': 20, 'multiclass': 10, 'regression': 20}; with a categorical column of >= 10 levels: 17 (34%). Case studies (not primary): 2.

| key                               |   did |   version | family                            | task       |   rows |   cols |   n_categorical |   max_cat_levels |   dup_frac | primary   | source              | md5_file                         |           frame_hash | date_typed_cols   | date_like_names                                                                                                                                                                                                                 |
|:----------------------------------|------:|----------:|:----------------------------------|:-----------|-------:|-------:|----------------:|-----------------:|-----------:|:----------|:--------------------|:---------------------------------|---------------------:|:------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| kr-vs-kp                          |     3 |         1 | krvskp                            | binary     |   3196 |     36 |              36 |                3 |     0      | True      | CC18+AMLB-clf       | 44a8cdccb27891b8014a3bdfb5f35813 |  5846383090484319539 |                   |                                                                                                                                                                                                                                 |
| mfeat-fourier                     |    14 |         1 | mfeat                             | multiclass |   2000 |     76 |               0 |                0 |     0.003  | True      | CC18                | 00e0dd8627b970dbe47146461f67829d | 10580013087150151890 |                   |                                                                                                                                                                                                                                 |
| cmc                               |    23 |         1 | cmc                               | multiclass |   1473 |      9 |               7 |                4 |     0.0326 | True      | CC18+AMLB-clf       | 1fe6aaf4995211a8e40bf2f910ddb758 |  4276284797694165826 |                   |                                                                                                                                                                                                                                 |
| credit-g                          |    31 |         1 | creditg                           | binary     |   1000 |     20 |              13 |               10 |     0      | True      | CC18+AMLB-clf+typed | f20f4f62ce92921ab932b78312ff68d1 |  7606122327431373292 |                   |                                                                                                                                                                                                                                 |
| pendigits                         |    32 |         1 | pendigits                         | multiclass |  10992 |     16 |               0 |                0 |     0      | True      | CC18                | 887d950c52887d387b92fbece35185dc | 11286597853309107680 |                   |                                                                                                                                                                                                                                 |
| sick                              |    38 |         1 | sick                              | binary     |   3772 |     29 |              23 |                5 |     0.0162 | True      | CC18                | edbbcb7bdee005d1d044a98e9e20e5b8 | 15039467398525504561 |                   |                                                                                                                                                                                                                                 |
| yeast                             |   181 |         1 | yeast                             | multiclass |   1484 |      8 |               0 |                0 |     0.0209 | True      | AMLB-clf            | ccc7c17fa863094422411219f810e3fb |  7056912554020308106 |                   |                                                                                                                                                                                                                                 |
| satimage                          |   182 |         1 | satimage                          | multiclass |   6430 |     36 |               0 |                0 |     0      | True      | CC18                | 8953b6e4951f56af844637873ecd4b18 | 12922424232236546568 |                   |                                                                                                                                                                                                                                 |
| pol                               |   201 |         1 | pol                               | regression |  15000 |     48 |               0 |                0 |     0.0028 | True      | AMLB-reg            | c6882da7daef4781d938e04fefe5f76d | 10275077228533597765 |                   |                                                                                                                                                                                                                                 |
| elevators                         |   216 |         1 | elevators                         | regression |  16599 |     18 |               0 |                0 |     0      | True      | AMLB-reg            | 4cae794c87e84f2457037a7cc2ce6ba1 |  9179919818536369724 |                   | SaTime1,SaTime2,SaTime3,SaTime4,diffSaTime1,diffSaTime2,diffSaTime3,diffSaTime4                                                                                                                                                 |
| space_ga                          |   507 |         1 | spacega                           | regression |   3107 |      6 |               0 |                0 |     0      | True      | AMLB-reg            | ff323cd77564eb01a9d15ddf6de6d145 |  7552628635272816196 |                   |                                                                                                                                                                                                                                 |
| socmob                            |   541 |         1 | socmob                            | regression |   1156 |      5 |               4 |               17 |     0      | True      | AMLB-reg            | 547fb2aa23f936b93beaeb5be53ce25d | 14581229078174021808 |                   |                                                                                                                                                                                                                                 |
| house_16H                         |   574 |         1 | house16h                          | regression |  22784 |     16 |               0 |                0 |     0      | True      | AMLB-reg            | 2bfe0315bb0670447a4b94fb7fcf8671 |  9172293859343199803 |                   |                                                                                                                                                                                                                                 |
| bank-marketing                    |  1461 |         1 | bankmarketing                     | binary     |  45211 |     16 |               9 |               12 |     0      | True      | CC18+AMLB-clf+typed | 874b8e0e5d38605db3190ef8534ac3d4 |  2484976918774064709 |                   |                                                                                                                                                                                                                                 |
| banknote-authentication           |  1462 |         1 | banknoteauthentication            | binary     |   1372 |      4 |               0 |                0 |     0.0175 | True      | CC18                | c01f837e3d57da29c67e0a998ce12e3e |  6036520945535955541 |                   |                                                                                                                                                                                                                                 |
| ozone-level-8hr                   |  1487 |         1 | ozonelevel8hr                     | binary     |   2534 |     72 |               0 |                0 |     0.0032 | True      | CC18+AMLB-clf       | cc9a86abd9bc68c7d324d66fd71f08f3 |  1407284738404859728 |                   |                                                                                                                                                                                                                                 |
| phoneme                           |  1489 |         1 | phoneme                           | binary     |   5404 |      5 |               0 |                0 |     0.0017 | True      | CC18+AMLB-clf       | 962dad720372fbb130cd5227f175e73c |  6946153924460944913 |                   |                                                                                                                                                                                                                                 |
| qsar-biodeg                       |  1494 |         1 | qsarbiodeg                        | binary     |   1055 |     41 |               0 |                0 |     0.0028 | True      | CC18+AMLB-clf       | 45ad444487508919d1e01d658cab5815 |  6333269847869092554 |                   |                                                                                                                                                                                                                                 |
| wall-robot-navigation             |  1497 |         1 | wallrobotnavigation               | multiclass |   5456 |     24 |               0 |                0 |     0      | True      | CC18                | f8013bfac137366c222c0850cb152cfc | 16312164251834682395 |                   |                                                                                                                                                                                                                                 |
| adult                             |  1590 |         2 | adult                             | binary     |  48842 |     14 |               8 |               41 |     0.0011 | True      | CC18+AMLB-clf+typed | 6f9af14fb61d70198b494ed54b7cf7b6 | 11220653127075058909 |                   | hours-per-week                                                                                                                                                                                                                  |
| Amazon_employee_access            |  4135 |         1 | amazon                            | binary     |  32769 |      9 |               9 |             7518 |     0      | True      | AMLB-clf+typed      | 4b6066938515dc50b36eeeab3be13765 | 16508201442281791848 |                   |                                                                                                                                                                                                                                 |
| Amazon_employee_access_int        |  4135 |         1 | amazon                            | binary     |  32769 |      9 |               0 |             7518 |     0      | False     | AMLB-clf+typed      | 4b6066938515dc50b36eeeab3be13765 |   508317086095130713 |                   |                                                                                                                                                                                                                                 |
| GesturePhaseSegmentationProcessed |  4538 |         1 | gesturephasesegmentationprocessed | multiclass |   9873 |     32 |               0 |                0 |     0      | True      | CC18+AMLB-clf       | 98e5f72cebadb213089a35d5805431fa |  5203080199379285600 |                   |                                                                                                                                                                                                                                 |
| numerai28.6                       | 23517 |         2 | numerai286                        | binary     |  96320 |     21 |               0 |                0 |     0      | True      | CC18+AMLB-clf       | d6b6a846d786bcff9b874e376de5ec98 |  6053378713234538099 |                   |                                                                                                                                                                                                                                 |
| texture                           | 40499 |         1 | texture                           | multiclass |   5500 |     40 |               0 |                0 |     0.0049 | True      | CC18                | 8fe7c051c1a88c6c0463e5a270281ab9 |  7559918417259855066 |                   |                                                                                                                                                                                                                                 |
| connect-4                         | 40668 |         2 | connect4                          | multiclass |  67557 |     42 |              42 |                3 |     0      | True      | CC18+AMLB-clf       | 2a5fda3446a380abc733242d66c60847 |  4256675358754380921 |                   |                                                                                                                                                                                                                                 |
| churn                             | 40701 |         1 | churn                             | binary     |   5000 |     20 |               4 |               10 |     0      | True      | CC18+AMLB-clf       | 6e9a1fe79a5004616530d60980d6146f |  1948560441185624034 |                   | total_day_minutes,total_day_calls,total_day_charge                                                                                                                                                                              |
| wilt                              | 40983 |         2 | wilt                              | binary     |   4839 |      5 |               0 |                0 |     0.0041 | True      | CC18+AMLB-clf       | ebf4141d88a633fc9f6b54d4c86c8158 | 11513832925785059745 |                   |                                                                                                                                                                                                                                 |
| Moneyball                         | 41021 |         2 | moneyball                         | regression |   1232 |     14 |               6 |               39 |     0      | True      | CTR23+AMLB-reg      | f5d2ea0a7ca33cb17468d04620173868 | 18371239449379540741 |                   | Year,RankSeason                                                                                                                                                                                                                 |
| sylvine                           | 41146 |         1 | sylvine                           | binary     |   5124 |     20 |               0 |                0 |     0      | True      | AMLB-clf            | a7f8157aa73581b2bd60197713bbee3d |  2197574914309762447 |                   |                                                                                                                                                                                                                                 |
| ada                               | 41156 |         1 | ada                               | binary     |   4147 |     48 |               0 |                0 |     0.0002 | True      | AMLB-clf            | 073a57dc0e6027ebb198880b9886fecf | 11993760166359642252 |                   |                                                                                                                                                                                                                                 |
| kick                              | 41162 |         1 | kick                              | binary     |  72983 |     32 |              18 |             1063 |     0      | True      | AMLB-clf            | 9000c46c64ddfb3f32012b54e8d05e4a |  7669173070657304193 |                   | PurchDate,VehYear                                                                                                                                                                                                               |
| diamonds                          | 42225 |         1 | diamonds                          | regression |  53940 |      9 |               3 |                8 |     0.0027 | True      | AMLB-reg+typed      | 658eaf4009d25553284aaa211f88ccb7 | 12400245448291971708 |                   |                                                                                                                                                                                                                                 |
| house_prices_nominal              | 42563 |         1 | housepricesnominal                | regression |   1460 |     79 |              43 |               25 |     0      | True      | AMLB-reg            | 4a944fae729c327ad9b6058d6a7092a0 | 15861061464722162601 |                   | YearBuilt,YearRemodAdd                                                                                                                                                                                                          |
| Brazilian_houses                  | 42688 |         4 | brazilianhouses                   | regression |  10692 |     12 |               4 |               35 |     0.0335 | True      | AMLB-reg            | 6293fabe5f24101e7fbc97d517cbb78f | 15626364672743687083 |                   |                                                                                                                                                                                                                                 |
| OnlineNewsPopularity              | 42724 |         2 | onlinenewspopularity              | regression |  39644 |     59 |               0 |                0 |     0      | True      | AMLB-reg            | 40264bb20604b05927fdf463f080588d | 17147084186673794476 |                   | timedelta,weekday_is_monday,weekday_is_tuesday,weekday_is_wednesday,weekday_is_thursday,weekday_is_friday,weekday_is_saturday,weekday_is_sunday,global_sentiment_polarity,title_sentiment_polarity,abs_title_sentiment_polarity |
| abalone                           | 42726 |         5 | abalone                           | regression |   4177 |      8 |               1 |                3 |     0      | True      | AMLB-reg            | 767946787af525982285d72b3052177a | 18418919520324962107 |                   |                                                                                                                                                                                                                                 |
| colleges                          | 42727 |        14 | colleges                          | regression |   7063 |     44 |              12 |             6039 |     0      | True      | AMLB-reg            | 979583767394099838b282cb6d272830 | 17056424624515241249 |                   | percent_part_time,average_cost_academic_year,average_cost_program_year,percent_part_time_faculty,mean_earnings_6_years,median_earnings_6_years,mean_earnings_10_years,median_earnings_10_years                                  |
| house_sales                       | 42731 |         3 | housesales                        | regression |  21613 |     21 |               1 |               70 |     0      | True      | AMLB-reg            | 99bd987d3bc9554f8b3cd6db132d1f10 |  7893314387633940950 |                   | date_year,date_month,date_day                                                                                                                                                                                                   |
| Click_prediction_small            | 42733 |        10 | clickpredictionsmall              | binary     |  39948 |     11 |               6 |            30114 |     0.0006 | True      | AMLB-clf            | e1a4b2b75e96f4283b287edfe10e70b9 | 11426656764212397405 |                   |                                                                                                                                                                                                                                 |
| okcupid-stem                      | 42734 |         3 | okcupidstem                       | multiclass |  50789 |     19 |              17 |             7019 |     0.0001 | True      | AMLB-clf            | 1d2c8359db9599c1845eec22e823e3bd |  1165949438572898746 |                   |                                                                                                                                                                                                                                 |
| visualizing_soil                  | 44056 |         4 | visualizingsoil                   | regression |   8641 |      4 |               1 |                2 |     0      | True      | TB-cat-reg          | 538514200019223797ba3d835f08a42c | 11530121194783056378 |                   |                                                                                                                                                                                                                                 |
| Bike_Sharing_Demand               | 44063 |         7 | bikesharingdemand                 | regression |  17379 |     11 |               5 |                4 |     0.0003 | True      | TB-cat-reg          | 360e6f057b15d6c7d3cb8955e9b41416 |   203910454390705423 |                   | season,year,month,hour,holiday,workingday                                                                                                                                                                                       |
| eye_movements                     | 44157 |         8 | eyemovements                      | binary     |   7608 |     23 |               3 |                2 |     0      | True      | TB-cat-clf          | b175ed3a2249268cb16ad62a05e481bf | 13132955005870033920 |                   | timePrtctg                                                                                                                                                                                                                      |
| airfoil_self_noise                | 44957 |         8 | airfoilselfnoise                  | regression |   1503 |      5 |               0 |                0 |     0      | True      | CTR23               | 120eda8ce2be821c74df18af7288a9bf | 17945803456140043343 |                   |                                                                                                                                                                                                                                 |
| auction_verification              | 44958 |         7 | auctionverification               | regression |   2043 |      7 |               2 |                6 |     0      | True      | CTR23               | ec666317ccd2287ba615f8a4a9db45bb |  4074807302068729782 |                   |                                                                                                                                                                                                                                 |
| concrete_compressive_strength     | 44959 |         7 | concretecompressivestrength       | regression |   1030 |      8 |               0 |                0 |     0.0243 | True      | CTR23               | 3b3035c476267e7929a179e2cc5286c8 |  3757496306289002596 |                   |                                                                                                                                                                                                                                 |
| fps_benchmark                     | 44992 |         2 | fpsbenchmark                      | regression |  24624 |     43 |              14 |               27 |     0      | True      | CTR23               | 09256f68229687c6056337190067867d |  5376474954327924468 |                   |                                                                                                                                                                                                                                 |
| fifa                              | 45012 |        19 | fifa                              | regression |  19178 |     28 |               1 |              163 |     0      | True      | CTR23               | 61ff6bb01511ded2e9abbd42387d969a | 16967463257491026436 |                   |                                                                                                                                                                                                                                 |
| albert                            | 45035 |         2 | albert                            | binary     |  58252 |     31 |              10 |               14 |     0      | True      | TB-cat-clf          | 1e6e9b96e98255ae72fce4afb9efba75 | 17430244892716848182 |                   |                                                                                                                                                                                                                                 |
| default-of-credit-card-clients    | 45036 |         4 | defaultofcreditcardclients        | binary     |  13272 |     21 |               1 |                2 |     0.0022 | True      | TB-cat-clf          | a51c4e1b409c1fb2e329413e4e6961d3 |  4521900904229719378 |                   |                                                                                                                                                                                                                                 |
| house_sales_42092                 | 42092 |         2 | housesales                        | regression |  21613 |     19 |               2 |              372 |     0      | False     | typed               | f9cdc8443cf2b57e0cb2903d895de2be |  2454219016414863087 |                   | date                                                                                                                                                                                                                            |

## Selection log

- reject 6 letter (CC18): classes=26
- reject 11 balance-scale (CC18): rows=625
- reject 12 mfeat-factors (CC18+AMLB-clf): cols=216
- reject 15 breast-w (CC18): rows=699
- reject 28 optdigits (CC18): image
- reject 29 credit-approval (CC18): rows=690
- reject 37 diabetes (CC18): rows=768
- reject 44 spambase (CC18): duplicate rows 8.5%
- reject 46 splice (CC18): duplicate rows 5.8%
- reject 50 tic-tac-toe (CC18): rows=958
- reject 54 vehicle (CC18+AMLB-clf): rows=846
- reject 151 electricity (CC18): temporal order; random CV over-rewards date features
- reject 188 eucalyptus (CC18+AMLB-clf): rows=736
- reject 287 wine_quality (AMLB-reg): duplicate rows 18.1%
- reject 300 isolet (CC18): cols=617
- reject 307 vowel (CC18): rows=990
- reject 416 yprop_4_1 (AMLB-reg): cols=251
- reject 422 topo_2_1 (AMLB-reg): cols=266
- reject 458 analcatdata_authorship (CC18): rows=841
- reject 469 analcatdata_dmft (CC18): rows=797
- reject 505 tecator (AMLB-reg): rows=240
- reject 531 boston (AMLB-reg): rows=506
- reject 546 sensory (AMLB-reg): rows=576
- reject 550 quake (AMLB-reg): cols=3
- reject 554 mnist_784 (CC18): cols=784
- reject 1049 pc4 (CC18+AMLB-clf): duplicate rows 7.8%
- reject 1050 pc3 (CC18): duplicate rows 7.9%
- reject 1053 jm1 (CC18): duplicate rows 18.1%
- reject 1063 kc2 (CC18): rows=522
- reject 1067 kc1 (CC18+AMLB-clf): duplicate rows 42.5%
- reject 1068 pc1 (CC18): duplicate rows 14.0%
- reject 1111 KDDCup09_appetency (AMLB-clf): cols=230
- reject 1169 airlines (AMLB-clf): rows=539383
- reject 1457 amazon-commerce-reviews (AMLB-clf): cols=10000
- reject 1464 blood-transfusion-service-center (CC18+AMLB-clf): rows=748
- reject 1468 cnae-9 (CC18+AMLB-clf): cols=856
- reject 1475 first-order-theorem-proving (CC18+AMLB-clf): duplicate rows 7.9%
- reject 1478 har (CC18): cols=561
- reject 1480 ilpd (CC18): rows=583
- reject 1485 madelon (CC18): cols=500
- reject 1486 nomao (CC18+AMLB-clf): cols=118
- reject 1501 semeion (CC18): cols=256
- reject 1510 wdbc (CC18): rows=569
- reject 1515 micro-mass (AMLB-clf): rows=571
- reject 1596 covertype (AMLB-clf): rows=581012
- reject 3050 QSAR-TID-11 (AMLB-reg): cols=1024
- reject 3277 QSAR-TID-10980 (AMLB-reg): cols=1024
- reject 4134 Bioresponse (CC18+AMLB-clf): cols=1776
- reject 4534 PhishingWebsites (CC18+AMLB-clf): duplicate rows 47.1%
- reject 4541 Diabetes130US (AMLB-clf): rows=101766
- reject 4549 Buzzinsocialmedia_Twitter (AMLB-reg): rows=583250
- reject 6332 cylinder-bands (CC18): rows=540
- reject 23381 dresses-sales (CC18): rows=500
- reject 40498 wine-quality-white (AMLB-clf): duplicate rows 19.1%
- reject 40670 dna (CC18+AMLB-clf): cols=180
- reject 40923 Devnagari-Script (CC18): cols=1024
- reject 40927 CIFAR_10 (CC18): cols=3072
- reject 40978 Internet-Advertisements (CC18+AMLB-clf): cols=1558
- reject 40979 mfeat-pixel (CC18): cols=240
- reject 40981 Australian (AMLB-clf): rows=690
- reject 40984 segment (CC18+AMLB-clf): duplicate rows 11.5%
- reject 40994 climate-model-simulation-crashes (CC18): rows=540
- reject 40996 Fashion-MNIST (CC18+AMLB-clf): cols=784
- reject 41138 APSFailure (AMLB-clf): cols=170
- reject 41142 christine (AMLB-clf): cols=1636
- reject 41143 jasmine (AMLB-clf): cols=144
- reject 41144 madeline (AMLB-clf): cols=259
- reject 41145 philippine (AMLB-clf): cols=308
- reject 41147 albert (AMLB-clf): rows=425240
- reject 41150 MiniBooNE (AMLB-clf): rows=130064
- reject 41157 arcene (AMLB-clf): rows=100
- reject 41158 gina (AMLB-clf): cols=970
- reject 41159 guillermo (AMLB-clf): cols=4296
- reject 41161 riccardo (AMLB-clf): cols=4296
- reject 41163 dilbert (AMLB-clf): cols=2000
- reject 41164 fabert (AMLB-clf): cols=800
- reject 41165 robert (AMLB-clf): cols=7200
- reject 41166 volkert (AMLB-clf): cols=180
- reject 41167 dionis (AMLB-clf): rows=416188
- reject 41169 helena (AMLB-clf): classes=100
- reject 41540 black_friday (AMLB-reg): rows=166821
- reject 41980 SAT11-HAND-runtime-regression (AMLB-reg): cols=116
- reject 42570 Mercedes_Benz_Greener_Manufacturing (AMLB-reg): cols=376
- reject 42571 Allstate_Claims_Severity (AMLB-reg): rows=188318
- reject 42572 Santander_transaction_value (AMLB-reg): cols=4991
- reject 42705 Yolanda (AMLB-reg): rows=400000
- reject 42728 Airlines_DepDelay_10M (AMLB-reg): rows=10000000
- reject 42729 nyc-taxi-green-dec-2016 (AMLB-reg): rows=581835
- reject 42730 us_crime (AMLB-reg): cols=126
- reject 42732 sf-police-incidents (AMLB-clf): rows=2215023
- reject 42742 porto-seguro (AMLB-clf): rows=595212
- reject 42746 KDDCup99 (AMLB-clf): rows=4898431
- reject 42769 Higgs (AMLB-clf): rows=1000000
- reject 43071 MIP-2016-regression (AMLB-reg): cols=144
- reject 43072 KDDCup09-Upselling (AMLB-clf): cols=14891
- reject 44055 analcatdata_supreme (TB-cat-reg): duplicate rows 85.0%
- duplicate 44059 diamonds
- reject 44061 Mercedes_Benz_Greener_Manufacturing (TB-cat-reg): cols=359
- reject 44065 nyc-taxi-green-dec-2016 (TB-cat-reg): rows=581835
- reject 44068 particulate-matter-ukair-2017 (TB-cat-reg): rows=394299
- reject 44069 SGEMM_GPU_kernel_performance (TB-cat-reg): rows=241600
- reject 44156 electricity (TB-cat-clf): temporal order; random CV over-rewards date features
- reject 44159 covertype (TB-cat-clf): rows=423680
- reject 44960 energy_efficiency (CTR23): rows=768
- reject 44962 forest_fires (CTR23): rows=517
- reject 44965 geographical_origin_of_music (CTR23): cols=116
- reject 44966 solar_flare (CTR23): duplicate rows 67.2%
- reject 44967 student_performance_por (CTR23): rows=649
- reject 44970 QSAR_fish_toxicity (CTR23): rows=908
- reject 44971 white_wine (CTR23): duplicate rows 19.1%
- reject 44972 red_wine (CTR23): duplicate rows 15.0%
- reject 44976 sarcos (CTR23): duplicate rows 9.1%
- duplicate 44979 diamonds
- reject 44984 cps88wages (CTR23): duplicate rows 7.6%
- duplicate 44987 socmob
- reject 44994 cars (CTR23): rows=804
- reject 45038 road-safety (TB-cat-clf): rows=111762
- reject 45039 compas-two-years (TB-cat-clf): duplicate rows 45.4%
- reject 45041 topo_2_1 (TB-cat-reg): cols=255
- reject 45043 seattlecrime6 (TB-cat-reg): duplicate rows 87.5%
- reject 45045 delays_zurich_transport (TB-cat-reg): rows=5465575
- reject 45046 Allstate_Claims_Severity (TB-cat-reg): rows=188318
- reject 45047 Airlines_DepDelay_1M (TB-cat-reg): rows=1000000
- reject 45048 medical_charges (TB-cat-reg): rows=163065
- family housesales: drop 44989 kings_county (kept 42731)
- family brazilianhouses: drop 44990 brazilian_houses (kept 42688)
- family mfeat: drop 16 mfeat-karhunen (kept 14)
- family mfeat: drop 18 mfeat-morphological (kept 14)
- family mfeat: drop 22 mfeat-zernike (kept 14)
- family satimage: drop 40900 Satellite (kept 182)
- family brazilianhouses: drop 44062 Brazilian_houses (kept 42688)
- family housesales: drop 44066 house_sales (kept 42731)
- family abalone: drop 44956 abalone (kept 42726)
- family abalone: drop 45042 abalone (kept 42726)
- family spacega: drop 45402 space_ga (kept 507)
- share with a >= 10-level categorical is 34%, below 50%
- typed adult: already selected as 1590
- typed bank-marketing: already selected as 1461
- typed credit-g: already selected as 31
- typed Amazon_employee_access: already selected as 4135
- typed diamonds: already selected as 42225
