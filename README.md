# Automated ECG Interpretation

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<br>

<div align="center">
<img src="https://user-images.githubusercontent.com/46399191/191921241-495090db-a088-46b6-bd09-0f7f21170b0a.png" height="350"/>
</div>

## Summary 

Electrocardiography (ECG) is a key diagnostic tool to assess the cardiac condition of a patient. Automatic ECG interpretation algorithms as diagnosis support systems promise large reliefs for the medical personnel - only based on the number of ECGs that are routinely taken. However, the development of such algorithms requires large training datasets and clear benchmark procedures.

## Data Description

The [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/) is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. In total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements. Combined with the extensive annotation, this turns the dataset into a rich resource for training and evaluating automatic ECG interpretation algorithms. The dataset is complemented by extensive metadata on demographics, infarction characteristics, likelihoods for diagnostic ECG statements, and annotated signal properties.

In general, the dataset is organized as follows:

```
ptbxl
├── ptbxl_database.csv
├── scp_statements.csv
├── records100
├── 00000
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hea
│   │   ├── ...
│   │   ├── 00999_lr.dat
│   │   └── 00999_lr.hea
│   ├── ...
│   └── 21000
│        ├── 21001_lr.dat
│        ├── 21001_lr.hea
│        ├── ...
│        ├── 21837_lr.dat
│        └── 21837_lr.hea
└── records500
   ├── 00000
   │     ├── 00001_hr.dat
   │     ├── 00001_hr.hea
   │     ├── ...
   │     ├── 00999_hr.dat
   │     └── 00999_hr.hea
   ├── ...
   └── 21000
          ├── 21001_hr.dat
          ├── 21001_hr.hea
          ├── ...
          ├── 21837_hr.dat
          └── 21837_hr.hea
```          
          
The dataset comprises 21837 clinical 12-lead ECG records of 10 seconds length from 18885 patients, where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22). The value of the dataset results from the comprehensive collection of many different co-occurring pathologies, but also from a large proportion of healthy control samples.

| Records | Superclass | Description |
|:---|:---|:---|
9528 | NORM | Normal ECG |
5486 | MI | Myocardial Infarction |
5250 | STTC | ST/T Change |
4907 | CD | Conduction Disturbance |
2655 | HYP | Hypertrophy |

The waveform files are stored in WaveForm DataBase (WFDB) format with 16-bit precision at a resolution of 1μV/LSB and a sampling frequency of 500Hz (records500/) beside downsampled versions of the waveform data at a sampling frequency of 100Hz (records100/).

All relevant metadata is stored in ptbxldatabase.csv with one row per record identified by ecgid and it contains 28 columns.

All information related to the used annotation scheme is stored in a dedicated scp_statements.csv that was enriched with mappings to other annotation standards.

## Setup

### Install dependencies
Install the dependencies (wfdb, pytorch, torchvision, cudatoolkit, fastai, fastprogress) by creating a conda environment:

    conda env create -f requirements.yml
    conda activate autoecg_env

### Get data
Download the dataset (PTB-XL) via the follwing bash-script:

    get_dataset.sh

This script first downloads [PTB-XL from PhysioNet](https://physionet.org/content/ptb-xl/) and stores it in `data/ptbxl/`. 

## Usage

    python main.py

This will perform all experiments for inception1d. 
Depending on the executing environment, this will take up to several hours. 
Once finished, all trained models, predictions and results are stored in `output/`, 
where for each experiment a sub-folder is created each with `data/`, `models/` and `results/` sub-sub-folders. 

| Model | AUC &darr; | Experiment |
|:---|:---|:---|
| inception1d | 0.927(00) | All statements |
| inception1d | 0.929(00) | Diagnostic statements |
| inception1d | 0.926(00) | Diagnostic subclasses |
| inception1d | 0.919(00) | Diagnostic superclasses |
| inception1d | 0.883(00) | Form statements |
| inception1d | 0.949(00) | Rhythm statements |

### Download model and results

We also provide a [compressed zip-archive](https://drive.google.com/drive/folders/17za6IanRm7rpb1ZGHLQ80mJvBj_53LXJ?usp=sharing) containing the `output` folder corresponding to our runs including trained model and predictions.

## Results for Inception1d Model

| Experiment name  | Accuracy |  Precision | Recall | F1_Score | Specificity |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| All  | 0.9792  | 0.8949 | 0.1408 | 0.4824 | 0.9921 |
| Diagnostic  | 0.9806 | 0.8440 | 0.1556 | 0.4746 | 0.9952 |
| Sub-Diagnostic | 0.9660 | 0.8315 | 0.3021 | 0.5119 | 0.9887 |
| Super-Diagnostic | 0.8847 | 0.7938 | 0.6757 | 0.7157 | 0.9251 |
| Form | 0.9452 | 0.5619 | 0.1420 | 0.3843 | 0.9916 |
| Rhythm | 0.9844 | 0.7676 | 0.4489 | 0.7290 | 0.9722 |

For more evaluation (Confusion Matrix, ROC curve)  information and visualizations visit: [Model Evaluation](https://github.com/AutoECG/Automated-ECG-Interpretation/blob/main/evaluation/Model_Evaluation.ipynb)

## Contribution

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Works

1. Model Deployment.
2. Continue Preprocessing new ECG data from hospitals to test model reliability and accuracy. 
3. Figure out different parsing options for xml ecg files from different ECG machines versions.


## Contact

Feel free to reach out to us:
- DM [Zaki Kurdya](https://twitter.com/ZakiKurdya)
- DM [Zeina Saadeddin](https://twitter.com/jszeina)
- DM [Salam Thabit](https://twitter.com/salamThabetDo)

<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=blue
[contributors-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=lightgray
[forks-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/network/members
[stars-shield]: https://img.shields.io/github/stars/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=yellow
[stars-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/stargazers
[issues-shield]: https://img.shields.io/github/issues/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=red
[issues-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/issues
