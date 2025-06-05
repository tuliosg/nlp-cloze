# Sobre os dados
## Dados brutos
O conjunto de dados apresentado neste trabalho é oriundo da tabulação dos resultados da aplicação de Testes Cloze que ocorreu no Colégio de Aplicação da Universidade Federal de Sergipe (CODAP - UFS) no dia 29/02/2024. Os textos originais dos testes foram: "Consumismo entre os jovens", "Inseguranças no uso das redes sociais", "A importância da responsabilidade ambiental" e "O uso dos celulares por crianças".  

No arquivo `textos.json` estão contidos os dados relativos à quantidade de lacunas dos testes, os títulos dos textos utilizados e os gabaritos.

## Dados processados
São os arquivos resultantes da avaliação dos dados dos TCs. Estes arquivos preservam as colunas originais e possuem as colunas relacionadas às análises.

## Dicionário de dados 

| Nome da Coluna          | Descrição                                                                 | Tipo de Dado       |
|-------------------------|---------------------------------------------------------------------------|--------------------|
| `discente`            | Identificador único do aluno que realizou o teste.                        | String             |
| `ano`                 | Ano escolar do discente (ex: 6º ano).                                    | Inteiro            |
| `turma`               | Turma ou grupo ao qual o discente pertence (ex: "B").                    | String             |
| `texto`               | Título do texto original utilizado no teste cloze.                       | String             |
| `respostas`           | Lista de respostas fornecidas pelo discente para preencher as lacunas.   | Lista (String)     |
| `data`                | Data de aplicação do teste (formato DD-MM-AAAA).                         | Data               |
| `tempo_inicial`       | Horário de início da aplicação do teste (formato HH:MM).                 | Hora               |
| `tempo_final`         | Horário de conclusão do teste (formato HH:MM).                           | Hora               |
| `duracao`             | Tempo total gasto para completar o teste (em minutos).                   | Decimal            |
| `correcao`            | Lista de classificações das respostas (ex: `"erro"`, `"exata"`, etc.).   | Lista (String)     |
| `quadrante_1`         | Tupla com tipo de maior ocorrência em sequência e a quantidade de ocorrências no 1º quarto das lacunas (ex: `('erro', 3)`). | Tupla (String, Inteiro) |
| `quadrante_2`         | Tupla com tipo de maior ocorrência em sequência e a quantidade de ocorrências no 2º quarto das lacunas                 | Tupla (String, Inteiro) |
| `quadrante_3`         | Tupla com tipo de maior ocorrência em sequência e a quantidade de ocorrências no 3º quarto das lacunas                 | Tupla (String, Inteiro) |
| `quadrante_4`         | Tupla com tipo de maior ocorrência em sequência e a quantidade de ocorrências no 4º quarto das lacunas                 | Tupla (String, Inteiro) |
| `compreensao`         | Percentual de compreensão do texto, baseado nas pontuações das lacunas.  | Decimal            |
| `coeficiente_variacao`| Coeficiente de variação da compreensão (medida de dispersão).            | Decimal            |
| `taxa_exatas`         | Percentual de respostas exatas em relação ao total de lacunas. | Decimal |
| `exata`               | Quantidade de respostas exatas.                       | Inteiro            |
| `grafia_incorreta`    | Quantidade de respostas esperadas (exatas) com erros ortográficos. | Inteiro         |
| `aceitavel`           | Quantidade de respostas semanticamente aceitáveis.          | Inteiro            |
| `classe_correta`      | Quantidade de respostas com classe gramatical correta mas não-aceitáveis.                   | Inteiro            |
| `erro`                | Quantidade de respostas incorretas.                        | Inteiro            |
| `branco`              | Quantidade de lacunas deixadas em branco.                                | Inteiro            |

