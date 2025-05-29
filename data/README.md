# Dicionário de dados

| Nome da Coluna          | Descrição                                                                | Tipo de Dado  |
|-------------------------|-------------------------------------------------------------------------------------|---------------|
| `discente`            | Identificador único do aluno que realizou o teste.                                  | String        |
| `ano`                 | Ano escolar do discente (ex: 6º ano).                                               | Inteiro       |
| `turma`               | Turma ou grupo ao qual o discente pertence (ex: "B").                               | String        |
| `texto`               | Título do texto original utilizado no teste cloze.                              | String        |
| `respostas`           | Lista de respostas fornecidas pelo discente para preencher as lacunas.               | Lista (String)|
| `data`                | Data de aplicação do teste (formato DD-MM-AAAA).                                    | Data          |
| `tempo_inicial`       | Horário de início da aplicação do teste (formato HH:MM).                            | Hora          |
| `tempo_final`         | Horário de conclusão do teste (formato HH:MM).                                      | Hora          |
| `duracao`             | Tempo total gasto para completar o teste (em minutos).                              | Decimal       |
| `compreensao`         | Percentual de compreensão do texto, calculado com base nas pontuações das lacunas.  | Decimal       |
| `coeficiente_variacao`| Medida de dispersão das respostas (quanto variaram em relação à média).             | Decimal       |
| `percentual_corretas` | Percentual de respostas exatas em relação ao total de lacunas.                      | Decimal       |
| `correta`             | Quantidade de respostas exatas fornecidas pelo discente.                            | Inteiro       |
| `grafia_incorreta`    | Quantidade de respostas esperadas porém com erros ortográficos.     | Inteiro       |
| `aceitavel`           | Quantidade de respostas semanticamente aceitáveis.      | Inteiro       |
| `classe_correta`      | Quantidade de respostas com classe gramatical correta (ex: substantivo, verbo).     | Inteiro       |
| `erro`                | Quantidade de respostas completamente incorretas.      | Inteiro       |
| `branco`              | Quantidade de lacunas deixadas em branco pelo discente.                             | Inteiro       |

