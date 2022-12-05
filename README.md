# TrustworthyCPI: Trustworthy Compound–Protein Interaction Prediction

This code is an implementation of our paper
" Trustworthy Compound–Protein Interaction Prediction" in PyTorch. In this repository, we provide two CPI datasets:
Human and C.elegans created by
"[Improving compound–protein interaction prediction by building up highly credible negative samples (Bioinformatics, 2015).](https://academic.oup.com/bioinformatics/article/31/12/i221/216307)"

In our problem setting of TrustworthyCPI prediction, the input of our model is the pair of a SMILES format of compound and an
amino acid sequence of protein and the outputs are not only a binary prediction result interaction but also the the confidence level of
prediction result. The overview of our **TrustworthyCPI Model** is as follows:
![Alt](Figures/The_framework%20_of%20_TrustworthyCPI.jpg)

The details of the TrustworthyCPI Model are described in our paper.
## Characteristics of the repository

- We provide the **several demo-scripts** of the experiments in the paper, so that you can quickly understand the trustworthy
  prediction process of the TrustworthyCPI model.
- This code is **easy to use**. It means that you can customize your own dataset and train your own TrustworthyCPI prediction
  model, and apply it to the new "trustworthy drug discovery" scenario.
- We provide **several pre-training models** on Human and C.elegans datasets for your further researches.

## Requirements

- Pytorch 1.10.0
- Numpy 1.19.2
- Scikit-learn 0.24.2
- RDKit 2020.09.1.0

## (a) Case Study

- Run the script **"case_study/drugs_trustworthy_predict.py"** for trustworthy drug discovery on SARS CoV2 3CLPro (
  For the detailed description of this experiment, please refer to **4.6 Case Study** in the paper):

<details>
  <summary>Click here for the results!</summary>

```
Trustworthy Drug Discovery Result for SARS-CoV2 3CL Protease
+------+----------------------+-------------+------------------------+----------------------+
| Rank |      Drug_Name       | Interaction |      Probability       |     Uncertainty      |
+------+----------------------+-------------+------------------------+----------------------+
|  1   |      Cobicistat      |     YES     |          1.0           | 0.08056621998548508  |
|  2   |      Sofosbuvir      |     YES     |          1.0           | 0.09978272765874863  |
|  3   |     Grazoprevir      |     YES     |   0.9999995231628418   | 0.12084315717220306  |
|  4   |      Lopinavir       |     YES     |   0.9999973773956299   | 0.13479585945606232  |
|  5   |      Maraviroc       |     YES     |   0.999990701675415    |  0.1472548395395279  |
|  6   |      Ritonavir       |     YES     |   0.9999873638153076   | 0.15061740577220917  |
|  7   |      Amprenavir      |     YES     |   0.9999854564666748   | 0.15222729742527008  |
|  8   |      Vicriviroc      |     YES     |   0.9999790191650391   | 0.15657827258110046  |
|  9   |    Fosamprenavir     |     YES     |   0.9999722242355347   | 0.15950043499469757  |
|  10  |      Letermovir      |     YES     |   0.9999697208404541   | 0.16121140122413635  |
|  11  |      Darunavir       |     YES     |   0.9999691247940063   | 0.16148033738136292  |
|  12  |      Remdesivir      |     YES     |   0.9995914101600647   | 0.20403088629245758  |
|  13  |      Saquinavir      |     YES     |   0.9993853569030762   | 0.21290338039398193  |
|  14  |      Foscarnet       |     YES     |   0.9985707998275757   |  0.2339385598897934  |
|  15  |      Telaprevir      |     YES     |   0.9983593821525574   |  0.2010272592306137  |
|  16  |     Daclatasvir      |     YES     |   0.9981291890144348   |  0.2063629925251007  |
|  17  |      Tipranavir      |     YES     |   0.9841045141220093   | 0.19573618471622467  |
|  18  |     Rilpivirine      |     YES     |   0.9790443778038025   | 0.15529493987560272  |
|  19  |       Adefovir       |     YES     |   0.9688928127288818   | 0.14851942658424377  |
|  20  |     Glecaprevir      |     YES     |   0.905486524105072    | 0.21173739433288574  |
|  21  |      Boceprevir      |     YES     |   0.7066730260848999   | 0.15546026825904846  |
|  22  |      Indinavir       |      NO     |  0.009367047809064388  | 0.18778075277805328  |
|  23  |      Etravirine      |      NO     | 0.0009644171223044395  | 0.13998864591121674  |
|  24  |      Peramivir       |      NO     | 0.00016467933892272413 | 0.11879550665616989  |
|  25  |      Simeprevir      |      NO     | 4.344771150499582e-05  | 0.13806137442588806  |
|  26  |      Doravirine      |      NO     | 7.302066478587221e-06  | 0.11595325917005539  |
|  27  |      Efavirenz       |      NO     | 5.0562639444251545e-06 | 0.13604062795639038  |
|  28  |     Elvitegravir     |      NO     |  2.1288449261192e-06   | 0.09184033423662186  |
|  29  |     Zalcitabine      |      NO     | 5.353192591428524e-07  | 0.09474369138479233  |
|  30  |      Ribavirin       |      NO     | 4.402370734624128e-08  | 0.07295691967010498  |
|  31  |      Pleconaril      |      NO     | 1.5305635869822254e-08 | 0.09400348365306854  |
|  32  | Tenofovir_disoproxil |      NO     | 1.010846073512539e-08  | 0.07316455990076065  |
|  33  |       Arbidol        |      NO     | 4.022041810713972e-09  | 0.09375815838575363  |
|  34  |      Lamivudine      |      NO     | 1.6988835893627652e-09 | 0.09011731296777725  |
|  35  |      Nelfinavir      |      NO     | 1.196819532367499e-09  | 0.08871698379516602  |
|  36  |     Raltegravir      |      NO     | 1.1846462699693916e-09 | 0.08042722195386887  |
|  37  |     Oseltamivir      |      NO     | 4.642930484521912e-10  | 0.08514077961444855  |
|  38  |      Atazanavir      |      NO     | 2.3937030135812165e-10 |  0.0773882046341896  |
|  39  |      Baloxavir       |      NO     |  9.56685702879767e-11  | 0.07977620512247086  |
|  40  |   Podophyllotoxin    |      NO     | 1.9823488336956352e-11 | 0.07506337016820908  |
|  41  |     Dolutegravir     |      NO     |  7.53349552878868e-12  | 0.06664511561393738  |
|  42  |     Bictegravir      |      NO     | 1.7600688701630007e-12 | 0.06880970299243927  |
|  43  |       Loviride       |      NO     | 4.684839213613123e-13  | 0.06581269204616547  |
|  44  |      Cidofovir       |      NO     | 1.1102960727465278e-13 | 0.06283582001924515  |
|  45  |     Delavirdine      |      NO     | 1.0704509652813277e-13 |  0.0627637505531311  |
|  46  |     Penciclovir      |      NO     | 4.687500658896032e-14  | 0.06117837131023407  |
|  47  |  Hydroxychloroquine  |      NO     | 2.6587603578725853e-14 | 0.060135308653116226 |
|  48  |     Methisazone      |      NO     | 9.331097059037032e-15  | 0.058299820870161057 |
|  49  |     Chloroquine      |      NO     | 8.779341491261475e-15  | 0.05819642171263695  |
|  50  |     Valacyclovir     |      NO     | 7.282774033899424e-15  | 0.05788164958357811  |
|  51  |      Nevirapine      |      NO     | 2.8178240821144073e-15 | 0.05633356794714928  |
|  52  |       Aspirin        |      NO     | 1.879640648856651e-15  | 0.05569836497306824  |
|  53  |     Nitazoxanide     |      NO     | 1.4003089391701785e-15 | 0.05524544045329094  |
|  54  |     Rimantadine      |      NO     | 1.0830121970737305e-15 | 0.05485609546303749  |
|  55  |      Entecavir       |      NO     | 9.142414117289032e-16  | 0.05460238456726074  |
|  56  |     Tromantadine     |      NO     | 1.8789944950667938e-16 | 0.052341461181640625 |
|  57  |      Amantadine      |      NO     | 1.1294451138741128e-16 | 0.05165337771177292  |
|  58  |      Zidovudine      |      NO     | 6.863771345908633e-17  | 0.05099739506840706  |
|  59  |     Enfuvirtide      |      NO     | 4.849240461973718e-17  | 0.050549570471048355 |
|  60  |      Pyrimidine      |      NO     | 1.6818598519536457e-17 | 0.04923192411661148  |
|  61  |    Emtricitabine     |      NO     |  9.83335685872727e-18  | 0.048589978367090225 |
|  62  |      Edoxudine       |      NO     | 8.649912865070103e-18  | 0.04843907058238983  |
|  63  |       Abacavir       |      NO     | 1.0318954064507137e-18 | 0.04606688395142555  |
|  64  |       Descovy        |      NO     | 2.5430853677027354e-19 | 0.04462717846035957  |
|  65  |      Tenofovir       |      NO     | 2.5430853677027354e-19 | 0.04462717846035957  |
|  66  |     Famciclovir      |      NO     | 1.845469414915127e-19  | 0.044310152530670166 |
|  67  |      Penicillin      |      NO     | 1.3352844463728797e-19 | 0.04399474710226059  |
|  68  |      Stavudine       |      NO     | 3.8197481702988356e-20 | 0.04281599447131157  |
|  69  |      Aciclovir       |      NO     |  3.81487006650676e-20  | 0.042814821004867554 |
|  70  |      Imiquimod       |      NO     | 3.7398140887926566e-20 | 0.04279661551117897  |
|  71  |     Taribavirin      |      NO     | 5.548250546904428e-21  | 0.04111774265766144  |
|  72  |       Inosine        |      NO     | 3.4351970734017615e-21 | 0.04071643576025963  |
|  73  |      Docosanol       |      NO     | 2.1571670801936046e-21 | 0.04033437743782997  |
|  74  |     Ibacitabine      |      NO     | 5.167394358367084e-22  | 0.039204537868499756 |
|  75  |    Valganciclovir    |      NO     | 3.0890780782141243e-22 | 0.03881309553980827  |
|  76  |      Moroxydine      |      NO     | 2.153293659552999e-22  | 0.03854316473007202  |
|  77  |      Vidarabine      |      NO     | 1.5737614762623245e-22 |  0.0383116789162159  |
|  78  |      Didanosine      |      NO     |  9.26801596424501e-24  | 0.036340199410915375 |
|  79  |      Zanamivir       |      NO     | 1.126411429729075e-24  | 0.03499991074204445  |
|  80  |     Trifluridine     |      NO     | 1.4770005431014706e-25 | 0.033798277378082275 |
|  81  |     Ganciclovir      |      NO     | 2.4736024056267565e-27 | 0.03161349520087242  |
|  82  |     Telbivudine      |      NO     | 1.7103415768094214e-31 | 0.027456142008304596 |
|  83  |      Amoxicillin     |      NO     | 3.0067233019826624e-34 | 0.025256657972931862 |
|  84  |     Idoxuridine      |      NO     | 2.9860802895675288e-34 | 0.02525446005165577  |
|  85  |       Imunovir       |      NO     |          0.0           | 0.01636103354394436  |
+------+----------------------+-------------+------------------------+----------------------+

```

</details>

## (b) Uncertainty estimation

- Run the script **"uncertainty_estimation_expirement/uncertainty_evaluation.py"** to filter out the trustworthy
  prediction datapoints and and only give the prediction results for them (For the detailed description of this
  experiment, please refer to **4.4 Uncertainty Estimation Performance** in the paper.):

<details>
  <summary>Click here for the results!</summary>

```
-------------------- Uncertainty_Testing_On_Celegans_Dataset --------------------
+---------------------+-------------------------------+--------------------+--------------------+
|      threshold      | Number of filtered datapoints |     Proportion     |        ACC         |
+---------------------+-------------------------------+--------------------+--------------------+
|         0.1         |              3215             | 0.6844794549712583 | 0.991912841796875  |
| 0.15000000000000002 |              3872             | 0.8243559718969555 | 0.9888945817947388 |
| 0.20000000000000004 |              4161             | 0.8858846071960826 | 0.9875029921531677 |
| 0.25000000000000006 |              4291             | 0.9135618479880775 | 0.9855511784553528 |
| 0.30000000000000004 |              4371             | 0.9305939961677666 | 0.9842141270637512 |
|  0.3500000000000001 |              4433             | 0.9437939110070258 | 0.9828558564186096 |
| 0.40000000000000013 |              4493             | 0.9565680221417926 | 0.9806365370750427 |
| 0.45000000000000007 |              4525             | 0.9633808814136683 | 0.9790055155754089 |
|  0.5000000000000001 |              4554             | 0.9695550351288056 | 0.9778217077255249 |
|  0.5500000000000002 |              4586             | 0.9763678944006813 | 0.9753597974777222 |
|  0.6000000000000002 |              4610             | 0.981477538854588  | 0.9731019735336304 |
|  0.6500000000000001 |              4622             | 0.9840323610815415 | 0.9712245464324951 |
|  0.7000000000000002 |              4638             | 0.9874387907174792 | 0.9706770181655884 |
|  0.7500000000000002 |              4665             | 0.9931871407281243 | 0.967845618724823  |
|  0.8000000000000002 |              4682             | 0.9968064722163082 | 0.9658265113830566 |
|  0.8500000000000002 |              4687             | 0.9978709814775388 | 0.9654363393783569 |
|  0.9000000000000002 |              4697             |        1.0         | 0.9646583199501038 |
|  0.9500000000000003 |              4697             |        1.0         | 0.9646583199501038 |
+---------------------+-------------------------------+--------------------+--------------------+
-------------------- Uncertainty_Testing_On_Human_Dataset --------------------
+---------------------+-------------------------------+--------------------+--------------------+
|      threshold      | Number of filtered datapoints |     Proportion     |        ACC         |
+---------------------+-------------------------------+--------------------+--------------------+
|         0.1         |              2989             | 0.7393024981449419 | 0.9899632334709167 |
| 0.15000000000000002 |              3409             | 0.8431857531535988 | 0.9873862862586975 |
| 0.20000000000000004 |              3622             | 0.8958694039079891 | 0.9839867353439331 |
| 0.25000000000000006 |              3752             | 0.928023744744002  | 0.9813433289527893 |
| 0.30000000000000004 |              3831             | 0.9475636903289636 | 0.9793787002563477 |
|  0.3500000000000001 |              3869             | 0.9569626514964136 | 0.9769966006278992 |
| 0.40000000000000013 |              3898             | 0.964135542913678  | 0.9753720164299011 |
| 0.45000000000000007 |              3923             | 0.9703190699975266 | 0.975019097328186  |
|  0.5000000000000001 |              3939             | 0.9742765273311897 | 0.9743589758872986 |
|  0.5500000000000002 |              3956             | 0.9784813257482068 | 0.9734580516815186 |
|  0.6000000000000002 |              3971             | 0.9821914419985159 | 0.9728027582168579 |
|  0.6500000000000001 |              3980             | 0.9844175117487015 | 0.9721105694770813 |
|  0.7000000000000002 |              3989             | 0.986643581498887  | 0.9709200263023376 |
|  0.7500000000000002 |              4001             | 0.9896116744991343 | 0.9702574610710144 |
|  0.8000000000000002 |              4009             | 0.9915904031659659 | 0.9695684909820557 |
|  0.8500000000000002 |              4020             | 0.9943111550828593 | 0.9681592583656311 |
|  0.9000000000000002 |              4043             |        1.0         | 0.9661142826080322 |
|  0.9500000000000003 |              4043             |        1.0         | 0.9661142826080322 |
+---------------------+-------------------------------+--------------------+--------------------+

```

</details>

## (c) OOD (Out-Of-Distribution) Detection

- Run the script **"uncertainty_estimation_expirement/OOD_detection_evaluation.py"** to to evaluate the OOD data
  detection capability of the TrustworthyCPI model. Intuitively, the prediction uncertainty u of OOD data should be higher than
  that of ID (In Distribution) data. For example, when we train the model on C.elegans, the uncertainty u of C.elegans
  test datapoints should be lower than that of Human's test datapoints in general (For the detailed description of this
  experiment, please refer to **4.5 OOD (Out-Of-Distribution) Detection** in the paper).

<details>
  <summary>Click here for the results!</summary>

```
-------------------- Celegans->Celegans (In Distribution) --------------------
+---------------------+--------------------------------+--------------------+--------------------+
|         0.1         |              3215              | 0.6844794549712583 | 0.991912841796875  |
| 0.15000000000000002 |              3872              | 0.8243559718969555 | 0.9888945817947388 |
| 0.20000000000000004 |              4161              | 0.8858846071960826 | 0.9875029921531677 |
| 0.25000000000000006 |              4291              | 0.9135618479880775 | 0.9855511784553528 |
| 0.30000000000000004 |              4371              | 0.9305939961677666 | 0.9842141270637512 |
|  0.3500000000000001 |              4433              | 0.9437939110070258 | 0.9828558564186096 |
| 0.40000000000000013 |              4493              | 0.9565680221417926 | 0.9806365370750427 |
| 0.45000000000000007 |              4525              | 0.9633808814136683 | 0.9790055155754089 |
|  0.5000000000000001 |              4554              | 0.9695550351288056 | 0.9778217077255249 |
|  0.5500000000000002 |              4586              | 0.9763678944006813 | 0.9753597974777222 |
|  0.6000000000000002 |              4610              | 0.981477538854588  | 0.9731019735336304 |
|  0.6500000000000001 |              4622              | 0.9840323610815415 | 0.9712245464324951 |
|  0.7000000000000002 |              4638              | 0.9874387907174792 | 0.9706770181655884 |
|  0.7500000000000002 |              4665              | 0.9931871407281243 | 0.967845618724823  |
|  0.8000000000000002 |              4682              | 0.9968064722163082 | 0.9658265113830566 |
|  0.8500000000000002 |              4687              | 0.9978709814775388 | 0.9654363393783569 |
|  0.9000000000000002 |              4697              |        1.0         | 0.9646583199501038 |
|  0.9500000000000003 |              4697              |        1.0         | 0.9646583199501038 |
+---------------------+--------------------------------+--------------------+--------------------+
-------------------- Celegans->Human (Out of Distribution) --------------------
+---------------------+--------------------------------+--------------------+--------------------+
|      threshold      | Number of filtered data points |     Proportion     |        ACC         |
+---------------------+--------------------------------+--------------------+--------------------+
|         0.1         |              1637              | 0.4048973534504081 | 0.9407452940940857 |
| 0.15000000000000002 |              2365              | 0.5849616621320801 | 0.9395349025726318 |
| 0.20000000000000004 |              2803              | 0.6932970566411081 | 0.9389939308166504 |
| 0.25000000000000006 |              3002              | 0.7425179322285431 | 0.9320453405380249 |
| 0.30000000000000004 |              3157              | 0.7808558001484046 | 0.9249287843704224 |
|  0.3500000000000001 |              3293              | 0.8144941874845412 | 0.9198299646377563 |
| 0.40000000000000013 |              3408              | 0.8429384120702449 | 0.9107981324195862 |
| 0.45000000000000007 |              3498              | 0.8651991095720999 | 0.9036592245101929 |
|  0.5000000000000001 |              3580              | 0.8854810784071234 | 0.8930167555809021 |
|  0.5500000000000002 |              3660              | 0.9052683650754391 | 0.8844262361526489 |
|  0.6000000000000002 |              3727              | 0.9218402176601533 | 0.8760397434234619 |
|  0.6500000000000001 |              3807              | 0.941627504328469  | 0.8655109405517578 |
|  0.7000000000000002 |              3873              | 0.9579520158298294 | 0.8587657809257507 |
|  0.7500000000000002 |              3936              | 0.9735345040811278 | 0.8523882031440735 |
|  0.8000000000000002 |              3989              | 0.986643581498887  | 0.8503383994102478 |
|  0.8500000000000002 |              4019              | 0.9940638139995053 | 0.8502114415168762 |
|  0.9000000000000002 |              4043              |        1.0         | 0.8501113057136536 |
|  0.9500000000000003 |              4043              |        1.0         | 0.8501113057136536 |
+---------------------+--------------------------------+--------------------+--------------------+
-------------------- Human->Human (In Distribution) --------------------
+---------------------+--------------------------------+--------------------+--------------------+
|      threshold      | Number of filtered data points |     Proportion     |        ACC         |
+---------------------+--------------------------------+--------------------+--------------------+
|         0.1         |              2989              | 0.7393024981449419 | 0.9899632334709167 |
| 0.15000000000000002 |              3409              | 0.8431857531535988 | 0.9873862862586975 |
| 0.20000000000000004 |              3622              | 0.8958694039079891 | 0.9839867353439331 |
| 0.25000000000000006 |              3752              | 0.928023744744002  | 0.9813433289527893 |
| 0.30000000000000004 |              3831              | 0.9475636903289636 | 0.9793787002563477 |
|  0.3500000000000001 |              3869              | 0.9569626514964136 | 0.9769966006278992 |
| 0.40000000000000013 |              3898              | 0.964135542913678  | 0.9753720164299011 |
| 0.45000000000000007 |              3923              | 0.9703190699975266 | 0.975019097328186  |
|  0.5000000000000001 |              3939              | 0.9742765273311897 | 0.9743589758872986 |
|  0.5500000000000002 |              3956              | 0.9784813257482068 | 0.9734580516815186 |
|  0.6000000000000002 |              3971              | 0.9821914419985159 | 0.9728027582168579 |
|  0.6500000000000001 |              3980              | 0.9844175117487015 | 0.9721105694770813 |
|  0.7000000000000002 |              3989              | 0.986643581498887  | 0.9709200263023376 |
|  0.7500000000000002 |              4001              | 0.9896116744991343 | 0.9702574610710144 |
|  0.8000000000000002 |              4009              | 0.9915904031659659 | 0.9695684909820557 |
|  0.8500000000000002 |              4020              | 0.9943111550828593 | 0.9681592583656311 |
|  0.9000000000000002 |              4043              |        1.0         | 0.9661142826080322 |
|  0.9500000000000003 |              4043              |        1.0         | 0.9661142826080322 |
+---------------------+--------------------------------+--------------------+--------------------+
-------------------- Human->Celegans (Out of Distribution) --------------------
+---------------------+--------------------------------+--------------------+--------------------+
|      threshold      | Number of filtered data points |     Proportion     |        ACC         |
+---------------------+--------------------------------+--------------------+--------------------+
|         0.1         |              1813              | 0.3859910581222057 | 0.9658024907112122 |
| 0.15000000000000002 |              2672              | 0.568873749201618  | 0.9509730935096741 |
| 0.20000000000000004 |              3115              | 0.6631892697466468 | 0.938683807849884  |
| 0.25000000000000006 |              3410              | 0.7259953161592506 | 0.9237536787986755 |
| 0.30000000000000004 |              3628              | 0.7724079199489036 | 0.9095920920372009 |
|  0.3500000000000001 |              3782              | 0.8051948051948052 | 0.8955578804016113 |
| 0.40000000000000013 |              3893              | 0.828826910794124  | 0.8872334957122803 |
| 0.45000000000000007 |              4004              | 0.8524590163934426 | 0.8771228790283203 |
|  0.5000000000000001 |              4110              | 0.8750266127315308 | 0.8688564300537109 |
|  0.5500000000000002 |              4177              | 0.8892910368320205 | 0.8637778162956238 |
|  0.6000000000000002 |              4237              | 0.9020651479667873 | 0.8567382097244263 |
|  0.6500000000000001 |              4299              | 0.9152650628060464 | 0.8508955240249634 |
|  0.7000000000000002 |              4359              | 0.9280391739408133 | 0.8458362221717834 |
|  0.7500000000000002 |              4408              | 0.9384713647008729 | 0.8411978483200073 |
|  0.8000000000000002 |              4460              | 0.9495422610176708 | 0.8369954824447632 |
|  0.8500000000000002 |              4534              | 0.9652969980838834 | 0.835906445980072  |
|  0.9000000000000002 |              4697              |        1.0         | 0.8337236642837524 |
|  0.9500000000000003 |              4697              |        1.0         | 0.8337236642837524 |
+---------------------+--------------------------------+--------------------+--------------------+

Process finished with exit code 0



```

</details>

## (d) Parameters Selection

In the ablation experiment in our paper (please refer to **4.2 Factors of Influencing the Performance of Trustwor
thyCPI** in the paper for the details), we found that when the feature learning module consists of 3-layer
convolutional neural network, and the annealing coefficient of the regularized term is set to λ_t=min(1,t/3), the model can obtain
the best uncertainty prediction performance. You can also try to adjust these two factors based on your own dataset to
achieve better performance. For the detailed parameters of the feature learning module, please refer to the **Figure 4**
of **3.2 Convolutional Representation Learning** in the paper.The pre-trained models with different convolution blocks
and different annealing coefficients are saved in "
/ablation_conv_models" and "/ablation_loss_models" folders, you can use them for the further researches. For the other
hyper parameters related to optimizer etc., please refer to **(4.1 Experimental Setup)** in the paper。

<details>
  <summary>Click here for the detailed effect of the annealing coefficient!</summary>

Effect of the different sizes of the annealing coefficient on the performance of uncertainty prediction

| --------------------Human Dataset-------------------- |--------------------C.elegans Dataset--------------------|
|:------:|:---------------------------------------------:|
|![Alt](Figures/effect_of%20annealing_coefficient_on_Human.jpg) |![Alt](Figures/effect_of%20annealing_coefficient_on_Celegans.jpg) |



</details>

## (e) Training of TrustworthyCPI Model using your customized CPI dataset and make trustworthy drug discoveries

**We recommend you to run "run_training.py" script to reproduce our experiment before attempting to train the TrustworthyCPI model
using your own custom CPI dataset to familiarize yourself with the training and prediction processes of the TrustworthyCPI.**

**- Step-1: Raw data format**

Please refer to **"data/CelegansByStr/data.txt"** file to store your Compound-Protein pairs in rows according to the
following format "(SMILES of Compound,Sequence of Protein,Interaction)" :

```
CCNC(C)CC1=CC(=CC=C1)C(F)(F)F,MHRASLICRLASPSRINAIRNASSGKSHISASTLVQHRNQSVAAAVKHEPFLNGSSSIYIEQMYEAWLQDPSSVHTSWDAYFRNVEAGAGPGQAFQAPPATAYAGALGVSPAAAQVTTSSAPATRLDTNASVQSISDHLKIQLLIRSYQTRGHNIADLDPLGINSADLDDTIPPELELSFYGLGERDLDREFLLPPTTFISEKKSLTLREILQRLKDIYCTSTGVEYMHLNNLEQQDWIRRRFEAPRVTELSHDQKKVLFKRLIRSTKFEEFLAKKWPSEKRFGLEGCEVLIPAMKQVIDSSSTLGVDSFVIGMPHRGRLNVLANVCRQPLATILSQFSTLEPADEGSGDVKYHLGVCIERLNRQSQKNVKIAVVANPSHLEAVDPVVMGKVRAEAFYAGDEKCDRTMAILLHGDAAFAGQGVVLETFNLDDLPSYTTHGAIHIVVNNQIGFTTDPRSSRSSPYCTDVGRVVGCPIFHVNVDDPEAVMHVCNVAADWRKTFKKDVIVDLVCYRRHGHNELDEPMFTQPLMYQRIKQTKTALEKYQEKILNEGVANEQYVKEELTKYGSILEDAYENAQKVTYVRNRDWLDSPWDDFFKKRDPLKLPSTGIEQENIEQIIGKFSQYPEGFNLHRGLERTLKGRQQMLKDNSLDWACGEALAFGSLLKEGIHVRLSGQDVQRGTFSHRHHVLHDQKVDQKIYNPLNDLSEGQGEYTVCNSSLSEYAVLGFELGYSMVDPNSLVIWEAQFGDFSNTAQCIIDQFISSGQSKWIRQSGLVMLLPHGYEGMGPEHSSARPERFLQMCNEDDEIDLEKIAFEGTFEAQQLHDTNWIVANCTTPANIYHLLRRQVTMPFRKPAVVFSPKSLLRHPMARSPVEDFQSGSNFQRVIPETGAPSQNPPDVKRVVFCTGKVYYDMVAARKHVGKENDVALVRVEQLSPFPYDLVQQECRKYQGAEILWAQEEHKNMGAWSFVQPRINSLLSIDGRATKYAGRLPSSSPATGNKFTHMQEQKEMMSKVFGVPKSKLEGFKA,0
C1CNCCN(C1)S(=O)(=O)C2=CC=CC3=C2C=CN=C3,MSRRSTTTSTNFGLSWSLVDVISSSTAVFKVPMNGGCDLWIGCARWLRDMKVLTTDKNGTMLEFASVLRDGILLCRLANTLVPNGIDQKKIMRTNQPSPFLCCNNINYFAMFCKTYFNLEDADLFTAEDLYYMNGFQKVLKTLSFLSHTKESLSRGVDPFPDTDNNQEGTSNGSEFEDDVEIYQSLHDNIENVDPNRTIYGPITSADPEEQQSEQLYDRIVTNRKPSMNENDLQNTPTLKRNRCIRELYDTEKNYVAQALVTIIKTFYEPLKGIIPTSDYNIIFGNIEEINVLHTALLADLEYPVKVALGLSDATPPRPISLNECVPQTIGEVFIKYRDQFLAYGKYCSNLPDSRKLSNELLKTNEFISRNINELTAQGNCKFGMNDLLCVPFQRLTKYPLLLKELQKKTDLASPDRKSLEEAVEVMEDVCNYINEESRDTNAIKVIDEIEQSITDLSMPLNVKLHDYGRVNLDGEVKMAESTLTQAGKPKQRYIFLFDKVIVVCKAANKVMAAKTTGASARTNTFTYKNAYVMSELTIDKNASLDVKSGGTITRRTQYVIIMTRDRNENNEITQLTFYFKNEATRNNWMTALLLSKSNVSPTDYLRDTNHKVAFHSFRVDVKNPATCDVCDKLMKGLQYQGYKCESCNMSMHKECLGLKKCEAVRKSTHETRSSQSFNCNRPRFHIHEGDIVVANSNSTPSDLSYLQFAKGDRIEVIKMQGHNRFTGCLINNRNRTGLVHLDHVSQSRTTSMIGLSPIDSPAGSIAPRVVRNESTVLPNKLLSDGSSRSLSGPHGSRSSRNSSSSTINGSMDSVPRQQDYVNTEISEFLWYMGEMERAKAESTLKGTPNGTFLVRYSKNRKQTAISLSYKNDVKHMIIEQNSDGKVYLDEDYIFNSTVELVQYYRSNNLIEIFAALDTCLKNPYSQCKVFKAVHDYDAPSPNNEGKFLSFKTGDIVVLLDTVGEDRGWWKGQVNNKSGFFPLSYVKPYDPATEGSSSPVTPTSSSS,0
C1=CC=C(C=C1)C(=O)Cl,MSTENGKSADAPVAAPAAKELTSKDYYFDSYAHFGIHEEMLKDEVRTTTYRNSIYHNSHLFKDKVVMDVGSGTGILSMFAAKAGAKKVFAMEFSNMALTSRKIIADNNLDHIVEVIQAKVEDVHELPGGIEKVDIIISEWMGYCLFYESMLNTVLVARDRWLAPNGMLFPDKARLYVCAIEDRQYKEDKIHWWDSVYGFNMSAIKNVAIKEPLVDIVDNAQVNTNNCLLKDVDLYTVKIEDLTFKSDFKLRCTRSDYIQAFVTFFTVEFSKCHKKTGFSTGPDVQYTHWKQTVFYLKDALTVKKGEEITGSFEMAPNKNNERDLDINISFDFKGEVCDLNEQNTYTMH,0
C1C2C(C(C(O2)N3C4=NC=NC(=C4N=C3Br)N)O)OP(=O)(O1)[O-],MEAVAEHDFQAGSPDELSFKRGNTLKVLNKDEDPHWYKAELDGNEGFIPSNYIRMTECNWYLGKITRNDAEVLLKKPTVRDGHFLVRQCESSPGEFSISVRFQDSVQHFKVLRDQNGKYYLWAVKFNSLNELVAYHRTASVSRTHTILLSDMNVETKFVQALFDFNPQESGELAFKRGDVITLINKDDPNWWEGQLNNRRGIFPSNYVCPYNSNKSNSNVAPGFNFGN,0
C1=CC=C2C(=C1)NC3=CC=CC=C3S2,MFARIVSRRAATGLFAGASSQCKMADRQVHTPLAKVQRHKYTNNENILVDHVEKVDPEVFDIMKNEKKRQRRGLELIASENFTSKAVMDALGSAMCNKYSEGYPGARYYGGNEFIDQMELLCQKRALEVFGLDPAKWGVNVQPLSGSPANFAVYTAIVGSNGRIMGLDLPDGGHLTHGFFTPARKVSATSEFFQSLPYKVDPTTGLIDYDKLEQNAMLFRPKAIIAGVSCYARHLDYERFRKIATKAGAYLMSDMAHISGLVAAGLIPSPFEYSDVVTTTTHKSLRGPRGALIFYRKGVRSTNAKGVDTLYDLEEKINSAVFPGLQGGPHNHTIAGIAVALRQCLSEDFVQYGEQVLKNAKTLAERMKKHGYALATGGTDNHLLLVDLRPIGVEGARAEHVLDLAHIACNKNTCPGDVSALRPGGIRLGTPALTSRGFQEQDFEKVGDFIHEGVQIAKKYNAEAGKTLKDFKSFTETNEPFKKDVADLAKRVEEFSTKFEIPGNETF,0
[Mg+2],MAAPWTPLESNPSVINPMIEKMGVSGVKTVDVLFFDDESIGKPQHAVILCFPEYKKVDEIMKPIYEQAKAADDSVFFMKQKISNACGTFALFHSLANLEDRINLGDGSFAKWLAEAKKVGIEERSDFLANNAELAGIHAAAATDGQTAPSGDVEHHFICFVGKNGILYEIDSRRPFAREIGPTSDATLVKDAGAACQHLIEKLDNVSFSAIAVVNQ,1
...
```

**- Step-2: Data preprocessing**

Run file **"data/CelegansByStr/data_preprocess.py"** to convert the original data to Encoding Vector (please refer
to **3.2 Convolutional Representation Learning** for the details).

```
[   2   80   81  115  164  284  ...],[11.  9. 18.  1. 17. 12.  ...],0
[ 306  315  351  367  379  393  ...],[11. 17. 18. 18. 17. 20. ...],0
[ 390  651  808 1089 1200 1293  ...],[11. 17. 20.  4. 14.  6. ...],0
[ 180  201  210  222  379  394  ...],[11.  4.  1. 22.  1.  4. ...],0
[ 199  398  463  676  950 1061  ...],[11.  7.  1. 18.  8. 22. ...],0
[273   0   0   0   0   0   0   0 ...],[11.  1.  1. 16. 21. 20. ...],1
```

**- Step-3: Encapsulate your own torch dataset**

Refer to the file **"MyUtils/MyData.py"** and encapsulate your own dataset into the Class "torch.utils.data.Dataset"
recommended by Pytorch.

**- Step-4: Training**

Run the file **"run_training.py"** to train and save your own TrustworthyCPI Model.

**- Step-5: Trustworthy Drug discovery**

Store and preprocess your Compound-Protein pairs to be predicted as described above and use file **"
drugs_trustworthy_predict.py"** for your own **trustworthy drug discovery**！

## Disclaimer

Please manually verify the reliability of the results by experts before conducting further drug experiments. Do not
directly use these drugs for disease treatment.

##Thanks
Thanks for the support of the following repositories:

| Source |                    Detail                     |
|:------:|:---------------------------------------------:|
| https://github.com/dougbrion/pytorch-classification-uncertainty |      Implement of Evidence Loss Function      |
| https://github.com/hkmztrk/DeepDTA | Implement of Protein Character Encoding Table |

## Cite Us
If you found this work useful to you, please our paper:
```
@article{XXX,
  title={TrustworthyCPI: Trustworthy Compound–Protein Interaction Prediction},
  author={XXX},
  journal={XXX},
  year={2022}
}
```