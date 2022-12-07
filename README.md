# Syntactically Robust Training on Partially-Observed Data for Open Information Extraction
Source code for EMNLP 2022 paper: [Syntactically Robust Training on Partially-Observed Data for Open Information Extraction](https://2022.emnlp.org/)

> Open Information Extraction models have shown promising results with sufficient supervision. However, these models face a fundamental challenge that the syntactic distribution of training data is partially observable in comparison to the real world. In this paper, we propose a syntactically robust training framework that enables models to be trained on a syntactic-abundant distribution based on diverse paraphrase generation. To tackle the intrinsic problem of knowledge deformation of paraphrasing, two algorithms based on semantic similarity matching and syntactic tree walking are used to restore the expressionally transformed knowledge. The training framework can be generally applied to other syntactic partial observable domains. Based on the proposed framework, we build a new evaluation set called CaRB-AutoPara, a syntactically diverse dataset consistent with the real-world setting for validating the robustness of the models. Experiments including a thorough analysis show that the performance of the model degrades with the increase of the difference in syntactic distribution, while our framework gives a robust boundary.


## 0. Package Description
```
RobustOIE/
|-- src/
|   |-- cluster/: clusters data into syntactic-specific subsets
|   |   |-- cluster.sh:
|   |   |-- k_means.py:
|   |   |-- correlation.ipynb
|   |   |-- plot.ipynb
|   |   |-- tag_dict.json
|   |-- generation/: generates robust samples
|   |   |-- generate.sh
|   |-- restoration/: restores structural knowledge
|   |   |-- restore.sh
|   |   |-- map_from_tree.py
|   |   |-- repair.py
|   |-- run.sh: main script
|
|-- LICENSE
|-- README.md
```

## 1. Environments

- python         (3.9.1)
- cuda           (11.3)
- Ubuntu-18.0.4  (4.15.0-136-generic)

## 2. Dependencies

- numpy          (1.23.4)
- matplotlib     (3.6.2)
- torch          (1.13.0)
- transformers   (4.24.0)


## 3. Preparation

### 3.1. Dataset

Downloading dataset using the following command

```bash
>> zenodo_get 3775983
```


### 3.2. (Optional) Pre-trained Language Models

Download [BERT](https://huggingface.co/bert-base-uncased) and [T5](https://huggingface.co/t5-base) models.

## 4. Execution

You might run each stage (e.g, generation, restoration and clustering) separately according to the main script.

```bash
>> cd src
>> ./run.sh
```

## 5. Resources
- Download the robust training data from this [link](https://cloud.tsinghua.edu.cn/f/b595ffbfcb8c494d874c/).
- Download the CaRB-AutoPara from this [link](https://cloud.tsinghua.edu.cn/f/24a97b734cb94c80be6d/).


## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 8. Citation

If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{qi-etal-2022-syntactically,
    title = "Syntactically Robust Training on Partially-Observed Data for Open Information Extraction",
    author = "Qi, Ji and
      Chen, Yuxiang and
      Hou, Lei and
      Li, Juanzi and
      Xu, Bin",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```

## 9. Contacts

If you have any questions, please feel free to contact [Ji Qi](qj20@mails.tsinghua.edu.cn), we will reply it as soon as possible.
