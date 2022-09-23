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

Electrocardiography (ECG) is a key diagnostic tool to assess the cardiac condition of a patient. Automatic ECG interpretation algorithms as diagnosis support systems promise large reliefs for the medical personnel - only on the basis of the number of ECGs that are routinely taken. However, the development of such algorithms requires large training datasets and clear benchmark procedures. In our opinion, both aspects are not covered satisfactorily by existing freely accessible ECG datasets.

The [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/) is a large dataset of 21837 clinical 12-lead ECGs from 18885 patients of 10 second length. The raw waveform data was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each record. The in total 71 different ECG statements conform to the SCP-ECG standard and cover diagnostic, form, and rhythm statements. To ensure comparability of machine learning algorithms trained on the dataset, we provide recommended splits into training and test sets. In combination with the extensive annotation, this turns the dataset into a rich resource for the training and the evaluation of automatic ECG interpretation algorithms. The dataset is complemented by extensive metadata on demographics, infarction characteristics, likelihoods for diagnostic ECG statements as well as annotated signal properties.

## Background

The waveform data underlying the PTB-XL ECG dataset was collected with devices from Schiller AG over the course of nearly seven years between October 1989 and June 1996. With the acquisition of the original database from Schiller AG, the full usage rights were transferred to the PTB. The records were curated and converted into a structured database within a long-term project at the Physikalisch-Technische Bundesanstalt (PTB). The database was used in a number of publications, but the access remained restricted until now. The Institutional Ethics Committee approved the publication of the anonymous data in an open-access database (PTB-2020-1). During the public release process in 2019, the existing database was streamlined with particular regard to usability and accessibility for the machine learning community. Waveform and metadata were converted to open data formats that can easily processed by standard software.

## Methods
### Data Acquisition
1. Raw signal data was recorded and stored in a proprietary compressed format. For all signals, we provide the standard set of 12 leads (I, II, III, AVL, AVR, AVF, V1, …, V6) with reference electrodes on the right arm.
2. The corresponding general metadata (such as age, sex, weight and height) was collected in a database.
3. Each record was annotated with a report string (generated by cardiologist or automatic interpretation by ECG-device) which was converted into a standardized set of SCP-ECG statements (scp_codes). For most records also the heart’s axis (`heart_axis`) and infarction stadium (`infarction_stadium1` and `infarction_stadium2`, if present) were extracted.
4. A large fraction of the records was validated by a second cardiologist.
5. All records were validated by a technical expert focusing mainly on signal characteristics.

### Data Preprocessing
ECGs and patients are identified by unique identifiers (`ecg_id` and `patient_id`). Personal information in the metadata, such as names of validating cardiologists, nurses and recording site (hospital etc.) of the recording was pseudonymized. The date of birth only as age at the time of the ECG recording, where ages of more than 89 years appear in the range of 300 years in compliance with HIPAA standards. Furthermore, all ECG recording dates were shifted by a random offset for each patient. The ECG statements used for annotating the records follow the SCP-ECG standard.

### Data Description
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
          
The dataset comprises 21837 clinical 12-lead ECG records of 10 seconds length from 18885 patients, where 52% are male and 48% are female with ages covering the whole range from 0 to 95 years (median 62 and interquantile range of 22). The value of the dataset results from the comprehensive collection of many different co-occurring pathologies, but also from a large proportion of healthy control samples. The distribution of diagnosis is as follows, where we restrict for simplicity to diagnostic statements aggregated into superclasses (note: sum of statements exceeds the number of records because of potentially multiple labels per record):


| Records | Superclass | Description |
|:---|:---|:---|
9528 | NORM | Normal ECG |
5486 | MI | Myocardial Infarction |
5250 | STTC | ST/T Change |
4907 | CD | Conduction Disturbance |
2655 | HYP | Hypertrophy |


The waveform files are stored in WaveForm DataBase (WFDB) format with 16 bit precision at a resolution of 1μV/LSB and a sampling frequency of 500Hz (records500/). For the user’s convenience we also release a downsampled versions of the waveform data at a sampling frequency of 100Hz (records100/).

All relevant metadata is stored in ptbxldatabase.csv with one row per record identified by ecgid. It contains 28 columns that can be categorized into:

1. Identifiers: Each record is identified by a unique `ecg_id`. The corresponding patient is encoded via patient_id. The paths to the original record (500 Hz) and a downsampled version of the record (100 Hz) are stored in `filename_hr` and `filename_lr`.
2. General Metadata: demographic and recording metadata such as age, sex, height, weight, nurse, site, device and recording_date
3. ECG statements: core components are `scp_codes` (SCP-ECG statements as a dictionary with entries of the form statement: likelihood, where likelihood is set to 0 if unknown) and report (report string). Additional fields are heart_axis, infarction_stadium1, infarction_stadium2, validated_by, second_opinion, initial_autogenerated_report and validated_by_human.
4. Signal Metadata: signal quality such as noise (static_noise and burst_noise), baseline drifts (baseline_drift) and other artifacts such as electrodes_problems. We also provide extra_beats for counting extra systoles and pacemaker for signal patterns indicating an active pacemaker.
5. Cross-validation Folds: recommended 10-fold train-test splits (strat_fold) obtained via stratified sampling while respecting patient assignments, i.e. all records of a particular patient were assigned to the same fold. Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. We therefore propose to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.


<img src= "https://i.ibb.co/VgtRfzH/META.png" alt ="META" style='width: 850px;'>


All information related to the used annotation scheme is stored in a dedicated scp_statements.csv that was enriched with mappings to other annotation standards such as AHA, aECGREFID, CDISC and DICOM. We provide additional side-information such as the category each statement can be assigned to (diagnostic, form and/or rhythm). For diagnostic statements, we also provide a proposed hierarchical organization into diagnostic_class and diagnostic_subclass.


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

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Feel free to reach out to us at: [autoecgucas@gmail.com](mailto:autoecgucas@gmail.com)

<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=blue
[contributors-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=lightgray
[forks-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/network/members
[stars-shield]: https://img.shields.io/github/stars/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=yellow
[stars-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/stargazers
[issues-shield]: https://img.shields.io/github/issues/AutoECG/Automated-ECG-Interpretation.svg?style=flat-square&color=red
[issues-url]: https://github.com/AutoECG/Automated-ECG-Interpretation/issues
