
  <a align="left" href="https://doi.org/10.5753/stil.2025.37822">
    <img src="https://img.shields.io/badge/DOI-10.5753/stil.2025.37822-blue" alt="DOI">
  </a>
  <a href="lamid.ufs.br"><img  align="right" src="https://github.com/user-attachments/assets/915d65fb-281c-42db-b81b-05c785c2473e" alt="LAMID" height="34" /></a> <br/>


# Avaliação de eficiência na leitura: uma abordagem baseada em PLN
> Túlio S. de Gois <a href="https://orcid.org/0009-0000-5270-8033" target="blank"><img align="top" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/ORCID_iD.svg/2048px-ORCID_iD.svg.png" height="16" width="16" /></a> ; Raquel M. Ko. Freitag <a href="https://orcid.org/0000-0002-4972-4320" target="blank"><img align="top" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/ORCID_iD.svg/2048px-ORCID_iD.svg.png" height="16" width="16" /></a>

Repositório do trabalho *Avaliação de eficiência na leitura: uma abordagem baseada em PLN*, apresentado no XVI Simpósio Brasileiro de Tecnologia da Informação e da Linguagem Humana (STIL 2025).

## Estrutura do repositório

### 📁 src
Os códigos desenvolvidos neste trabalho.

### 📁 data
Conjuntos de dados utilizados para avaliação.

### 📁 docs 
Materiais suplementares relacionados ao artigo e às análises.

### 📄 `requirements.txt`
Dependências para executar os códigos da pasta `src`.

## Ambiente e execução
### Infraestrutura
Os códigos desenvolvidos foram executados no Google Colaboratory (cota gratuita em ambiente T4) e também em uma máquina com as especificações a seguir:

```
OS: Windows 11 Home
CPU: Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz (2.11 GHz)
RAM: 20 GB
GPU: Intel(R) UHD Graphics (128 MB)
Python: 3.11.2
```

### Instruções de execução

#### 1. Clonar o repo e criar o env

```bash
git clone https://github.com/tuliosg/nlp-cloze.git
cd nlp-cloze
python -m venv venv
```

#### 2. Ativar o ambiente virtual
 
```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
 
# Linux
source venv/bin/activate
```

#### 3. Executar os notebooks
Siga a ordem abaixo:

1. `/src/Organização dos dados.ipynb`
2. `/src/Análise dos modelos de linguagem.ipynb`
3. `/src/Avaliação de eficiência na leitura - uma abordagem baseada em PLN.ipynb`

> [!IMPORTANT]
> Para a execução do último notebook (3), é necessário inserir o `nlpcloze.py` no mesmo diretório para importá-lo.

## Como citar?
Para citar este trabalho, utilize a referência abaixo ou [acesse o artigo completo](https://doi.org/10.5753/stil.2025.37822).
```
@inproceedings{stil,
 author = {Túlio Gois and Raquel Freitag},
 title = { Avaliação de eficiência na leitura: uma abordagem baseada em PLN},
 booktitle = {Anais do XVI Simpósio Brasileiro de Tecnologia da Informação e da Linguagem Humana},
 location = {Fortaleza/CE},
 year = {2025},
 keywords = {},
 issn = {0000-0000},
 pages = {161--169},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/stil.2025.37822},
 url = {https://sol.sbc.org.br/index.php/stil/article/view/37822}
}
```


## Agradecimentos
Este trabalho está vinculado ao projeto *Impactos da pandemia de COVID-19 na linguagem da criança e do adulto: foco no desenvolvimento e na aprendizagem da leitura*,
financiado pelo edital Capes 12/2021 - PDPG Impactos da Pandemia. Agradecemos o suporte de infraestrutura e equipe do Laboratório Multiusuário de Informática e Documentação Linguística ([LAMID](https://github.com/lamid-ufs)). Agradecemos também à [Rede Brasileira de Reprodutibilidade](https://www.reprodutibilidade.org) pelo suporte financeiro, permitindo a participação no STIL.




