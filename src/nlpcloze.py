import ast
import json
import os
import pickle
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import torch
import torch.nn.functional as F
from nltk.metrics import edit_distance
from transformers import AutoTokenizer, AutoModel


class NLPCloze:
    """
    Classe para avaliação automática de testes Cloze.
    """

    def __init__(self,
                 model_name: str = "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
                 spacy_model: str = "pt_core_news_lg",
                 cache_dir: str = "./cache"
                 ):
        """
        Inicialização da classe.

        Args:
            model_name: Nome do modelo BERT para embeddings
            spacy_model: Modelo spaCy para análise linguística
            cache_dir: Diretório para cache de embeddings
        """
        self.model_name = model_name
        self.spacy_model = spacy_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache de embeddings
        self.cache_embeddings = {}
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"


    def _load_models(self):
        """Carrega os modelos necessários."""
        print("Carregando modelos...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            print(f"✓ {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")

        try:
            self.nlp = spacy.load(self.spacy_model, disable=["parser", "ner", "lemmatizer"])
            print(f"✓ {self.spacy_model}")
        except OSError:
            raise RuntimeError(f"Modelo '{self.spacy_model}' não encontrado. "
                             f"Instale com: python -m spacy download {self.spacy_model}")

    def _load_cache(self):
        """Carrega cache de embeddings"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache_embeddings = pickle.load(f)
            except Exception as e:
                print(f"Aviso: Erro ao carregar cache - {e}")
                self.cache_embeddings = {}

    def _save_cache(self):
        """Salva cache de embeddings"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache_embeddings, f)
        except Exception as e:
            print(f"Aviso: Erro ao salvar cache - {e}")

    def get_embedding(self, palavra: str) -> torch.Tensor:
        """
        Obtém embedding de uma palavra.

        Args:
            palavra: Palavra para embedding

        Returns:
            Tensor normalizado do embedding
        """
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
            self._load_models()
            self._load_cache()

        palavra = palavra.strip().lower()

        # Verifica cache
        if palavra in self.cache_embeddings:
            return self.cache_embeddings[palavra]

        # Calcula embedding
        try:
            encoded_input = self.tokenizer([palavra],
                                           return_tensors='pt',
                                           padding=True,
                                           truncation=True)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embedding = model_output.last_hidden_state[:, 0, :].squeeze(0)
            embedding = F.normalize(embedding, p=2, dim=0)

            # Armazena no cache
            self.cache_embeddings[palavra] = embedding.clone().detach()

            return embedding

        except Exception as e:
            print(f"Erro ao gerar embedding para '{palavra}': {e}")
            # Retorna embedding zero em caso de erro
            return torch.zeros(self.model.config.hidden_size)

    def similaridade(self, palavra1: str, palavra2: str) -> float:
        """
        Calcula similaridade semântica entre duas palavras.

        Args:
            palavra1: Primeira palavra
            palavra2: Segunda palavra

        Returns:
            Similaridade cosseno entre as palavras
        """
        emb1 = self.get_embedding(palavra1)
        emb2 = self.get_embedding(palavra2)

        cos_sim = torch.dot(emb1, emb2).item()
        return cos_sim

    def pos_tag(self, palavra: str) -> Optional[str]:
        """
        Obtém classe gramatical de uma palavra usando o spaCy.

        Args:
            palavra: Palavra de interesse

        Returns:
            Classe gramatical (POS tag) ou None se erro
        """
        if not hasattr(self, 'nlp'):
            self._load_models()
        try:
            doc = self.nlp(palavra.strip())
            if len(doc) > 0:
                return doc[0].pos_
            return None
        except Exception as e:
            print(f"Erro ao processar POS tag para '{palavra}': {e}")
            return None

    def avaliar_lacuna(self, gabarito: str, resposta: str) -> Tuple[float, str]:
        """
        Avalia uma lacuna individual do teste.

        Args:
            gabarito: Resposta esperada
            resposta: Resposta do aluno

        Returns:
            Tupla com (pontuação, tipo_resposta)
        """
        gabarito = str(gabarito).strip().lower()
        resposta = str(resposta).strip().lower()

        # Resposta exata
        if resposta == gabarito:
            return 1.0, 'exata'

        # Resposta em branco
        if resposta in ["-", "", " ", "none", "nan"]:
            return 0.0, 'branco'

        # Erro de grafia
        max_dist = max(1, len(resposta) // 2)
        if edit_distance(resposta, gabarito) <= max_dist:
            return 1.0, 'grafia_incorreta'

        pos_resposta = self.pos_tag(resposta)
        pos_gabarito = self.pos_tag(gabarito)

        # Classe gramatical correta
        if pos_resposta and pos_gabarito and pos_resposta == pos_gabarito:
            sim = self.similaridade(resposta, gabarito)

            # Alta similaridade semântica
            if sim >= 0.75:
                return 1.0, 'aceitavel'
            else:
                return 0.5, 'classe_correta'

        return 0.0, 'erro'

    def analisar_correcao(self, correcao: List[str]) -> Dict[str, int]:
        """
        Conta a ocorrência de cada tipo de resposta.

        Args:
            correcao: Lista com os tipos de resposta de cada lacuna.

        Returns:
            Dicionário com a contagem de cada tipo de resposta.
        """
        tipos_possiveis = ['exata', 'grafia_incorreta', 'aceitavel', 'classe_correta', 'erro', 'branco']
        contagens = {tipo: 0 for tipo in tipos_possiveis}
        contagens.update(Counter(correcao))
        return contagens

    def avaliar_respostas_cloze(self, gabarito: List[str],
                              respostas: Union[List[str], str]) -> Tuple[float, float, Dict]:
        """
        Avalia todas as lacunas de um teste Cloze.

        Args:
            gabarito: Lista de respostas corretas.
            respostas: Lista de respostas do aluno ou string formatada como lista.

        Returns:
            Tupla com (compreensão, coeficiente de variação, avaliação detalhada).
        """
        # Processa respostas se vier como string
        if isinstance(respostas, str):
            try:
                respostas = ast.literal_eval(respostas)
            except (ValueError, SyntaxError):
                respostas = [respostas]

        pontuacoes = []
        correcao_tipos = []
        n_lacunas = len(gabarito)

        # Avalia cada lacuna
        for i in range(n_lacunas):
            resposta_aluno = respostas[i] if i < len(respostas) else ""
            pontuacao, tipo = self.avaliar_lacuna(gabarito[i], resposta_aluno)
            pontuacoes.append(pontuacao)
            correcao_tipos.append(tipo)

        # Calcula métricas
        media = np.mean(pontuacoes) if pontuacoes else 0
        cv = np.std(pontuacoes) / media if media > 0 else 0

        compreensao = round(media * 100, 3)
        coef_variacao = round(cv, 3)

        # Monta o dicionário de avaliação
        contagens = self.analisar_correcao(correcao_tipos)
        avaliacao = {
            'pontuacao': pontuacoes,
            'correcao': correcao_tipos,
            **contagens
        }

        return compreensao, coef_variacao, avaliacao

    def analisar_quadrantes(self, tipos_resposta: List[str],
                            modo_analise: str = 'moda') -> List[Tuple[Optional[str], int]]:
        """
        Analisa respostas em quadrantes para identificar categorias dominantes.

        Args:
            tipos_resposta: Lista de tipos de resposta ('exata', 'grafia_incorreta', etc.)
            modo_analise: 'moda' para o tipo mais frequente ou 'sequencia' para a maior sequência.

        Returns:
            Lista com (categoria_dominante, contagem) para cada quarto.
        """
        quadrantes = np.array_split(tipos_resposta, 4)
        resultados = []

        for quadrante in quadrantes:
            if len(quadrante) == 0:
                resultados.append((None, 0))
                continue

            if modo_analise == 'moda':
                if not quadrante.any(): # Verifica se o quadrante não está vazio
                    resultados.append((None, 0))
                else:
                    contagem_quadrante = Counter(quadrante)
                    categoria_max, max_ocorrencias = contagem_quadrante.most_common(1)[0]
                    resultados.append((categoria_max, max_ocorrencias))

            elif modo_analise == 'sequencia':
                sequencias = [(tipo, len(list(grupo))) for tipo, grupo in groupby(quadrante)]
                categoria_max, max_rep = max(sequencias, key=lambda x: x[1])
                resultados.append((categoria_max, max_rep))
            else:
                raise ValueError("modo_analise deve ser 'moda' ou 'sequencia'")

        return resultados

    def intervalo_tempo(self, tempo_inicial: str, tempo_final: str) -> float:
        """
        Calcula duração do teste em minutos.

        Args:
            tempo_inicial: Horário de início
            tempo_final: Horário de término

        Returns:
            Duração em minutos
        """
        try:
            tempo_inicial = pd.to_datetime(tempo_inicial)
            tempo_final = pd.to_datetime(tempo_final)
            duracao = tempo_final - tempo_inicial
            return duracao.total_seconds() / 60
        except Exception as e:
            print(f"Erro ao calcular duração: {e}")
            return 0.0

    def processar_dataframe(self, df: pd.DataFrame, gabarito: List[str]) -> pd.DataFrame:
        """
        Processa dataframe completo com avaliações.

        Args:
            df: DataFrame com dados originais
            gabarito: Lista de respostas corretas

        Returns:
            DataFrame expandido com avaliações
        """
        df_resultado = df.copy()

        # Calcula duração
        if 'tempo_inicial' in df.columns and 'tempo_final' in df.columns:
            df_resultado['duracao'] = df_resultado.apply(
                lambda row: self.intervalo_tempo(row['tempo_inicial'], row['tempo_final']),
                axis=1
            )

        # Inicializa colunas de resultado
        resultados = {
            'correcao': [], 'quadrante_1': [], 'quadrante_2': [], 'quadrante_3': [], 'quadrante_4': [],
            'compreensao': [], 'coeficiente_variacao': [], 'taxa_exatas': [], 'exata': [],
            'grafia_incorreta': [], 'aceitavel': [], 'classe_correta': [], 'erro': [], 'branco': []
        }

        # Processa cada linha
        for _, row in df_resultado.iterrows():
            compreensao, coef_var, avaliacao = self.avaliar_respostas_cloze(gabarito, row['respostas'])

            # Adiciona resultados básicos
            resultados['correcao'].append(avaliacao['correcao'])
            resultados['compreensao'].append(compreensao)
            resultados['coeficiente_variacao'].append(coef_var)
            resultados['taxa_exatas'].append(
                round(avaliacao['exata'] / len(gabarito) * 100, 3) if gabarito else 0
            )

            # Adiciona contagens por tipo
            for tipo in ['exata', 'grafia_incorreta', 'aceitavel', 'classe_correta', 'erro', 'branco']:
                resultados[tipo].append(avaliacao[tipo])

            # Análise de quadrantes (usando o novo modo padrão 'moda')
            analise_quadrantes = self.analisar_quadrantes(avaliacao['correcao'], modo_analise='moda')
            for i, (categoria, contagem) in enumerate(analise_quadrantes, 1):
                quadrante_key = f'quadrante_{i}'
                resultados[quadrante_key].append((str(categoria), contagem))

        # Adiciona colunas ao DataFrame
        for coluna, valores in resultados.items():
            df_resultado[coluna] = valores

        return df_resultado

    def get_gabarito(self, titulo: str, arquivo_textos: str) -> List[str]:
        """
        Carrega gabarito de teste específico.

        Args:
            titulo: Título do texto
            arquivo_textos: Caminho para arquivo JSON com os dados dos textos

        Returns:
            Lista de respostas corretas
        """
        try:
            with open(arquivo_textos, 'r', encoding='utf-8') as file:
                dados_textos = json.load(file)

            df_textos = pd.DataFrame(dados_textos)
            gabarito = df_textos[df_textos['titulo'] == titulo]['respostas'].values

            if len(gabarito) > 0:
                return gabarito[0]
            else:
                raise ValueError(f"Texto '{titulo}' não encontrado no arquivo")

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar gabarito: {e}")

    def plot_dist_respostas(self, df: pd.DataFrame, tipo: str = 'compreensao',
                            salvar: bool = True, output_dir: str = "./plots"):
        """
        Plota distribuição de desempenho dos alunos.

        Args:
            df: DataFrame com dados de desempenho
            tipo: Tipo de métrica ('compreensao' ou 'taxa_exatas')
            salvar: Se deve salvar o gráfico
            output_dir: Diretório para salvar gráficos
        """
        plt.figure(figsize=(7, 5))
        sns.histplot(df[tipo], bins=20, kde=True)
        plt.xlim(0, 100)

        # Labels e título
        if tipo == 'compreensao':
            titulo = f"Distribuição da Compreensão de Leitura"
            xlabel = 'Taxa de Compreensão (%)'
        else:
            titulo = f"Distribuição da Taxa de Acerto"
            xlabel = 'Taxa de Acerto (%)'

        # Adiciona informações da turma
        if 'ano' in df.columns and 'turma' in df.columns:
            titulo += f" - {df['ano'].iloc[0]}º {df['turma'].iloc[0]}"

        plt.title(titulo)
        plt.xlabel(xlabel)
        plt.ylabel('Frequência')

        if salvar:
            os.makedirs(output_dir, exist_ok=True)
            sufixo = f"{df['ano'].iloc[0]}-{df['turma'].iloc[0]}" if 'ano' in df.columns else "resultado"
            plt.savefig(f"{output_dir}/[{tipo}]_distribuicao_{sufixo}.png", dpi=300, bbox_inches='tight')

        plt.show()

    def plot_eficiencia_leitura(self, df: pd.DataFrame, tipo: str = 'compreensao',
                                salvar: bool = True, output_dir: str = "./plots"):
        """
        Plota eficiência de leitura (desempenho vs tempo).
        Baseado em Cardoso et al. (2024) - https://osf.io/47m93/.

        Args:
            df: DataFrame com dados de desempenho
            tipo: Tipo de métrica ('compreensao' ou 'taxa_exatas')
            salvar: Se deve salvar o gráfico
            output_dir: Diretório para salvar gráficos
        """
        plt.figure(figsize=(6, 6))

        # Scatter plot
        sns.scatterplot(x=tipo, y='duracao', data=df, color='black', s=30)

        # Linhas de referência
        media_desempenho = np.mean(df[tipo])
        media_duracao = np.mean(df['duracao'])

        plt.axvline(x=media_desempenho, color='blue', linestyle='--', linewidth=2)
        plt.axvline(x=44, color='red', linewidth=2)
        plt.axvline(x=58, color='green', linewidth=2)
        plt.axhline(y=media_duracao, color='blue', linestyle='--', linewidth=2)

        # Labels e título
        titulo_base = "Eficiência de Leitura: "
        if tipo == 'compreensao':
            titulo_base += "Compreensão vs Tempo"
            xlabel = "Compreensão (%)"
        else:
            titulo_base += "Taxa de Acerto vs Tempo"
            xlabel = "Taxa de Acerto (%)"

        # Adiciona informações da turma se disponível
        if 'ano' in df.columns and 'turma' in df.columns:
            titulo_base += f"\n{df['ano'].iloc[0]}º {df['turma'].iloc[0]}"

        plt.title(titulo_base, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Tempo (min)", fontsize=12)
        plt.xlim(0, 100)
        plt.ylim(0, 60)

        plt.tight_layout()

        if salvar:
            os.makedirs(output_dir, exist_ok=True)
            sufixo = f"{df['ano'].iloc[0]}-{df['turma'].iloc[0]}" if 'ano' in df.columns else "resultado"
            plt.savefig(f"{output_dir}/[{tipo}]_eficiencia_leitura_{sufixo}.png",
                        dpi=300, bbox_inches='tight')

        plt.show()

    def __del__(self):
        """Salva cache ao destruir objeto."""
        if hasattr(self, 'cache_embeddings') and self.cache_embeddings:
            self._save_cache()