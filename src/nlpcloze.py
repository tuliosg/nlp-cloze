import ast
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
import spacy
from spacy.cli.download import download as spacy_download
import torch
import torch.nn.functional as F
from nltk.metrics import edit_distance
from transformers import AutoTokenizer, AutoModel


class NLPCloze:
    """
    Classe para avaliação automática e contextual de testes Cloze.
    """
    def __init__(self,
                 model_name: str = "neuralmind/bert-base-portuguese-cased",
                 spacy_model: str = "pt_core_news_lg",
                 limiar_aceitacao: float = 0.652,
                 cache_path: str = "../src/cache/embeddings_cache.pt"):
        """
        Inicializa a classe com o modelo e limiar otimizados.

        Args:
            model_name (str): Nome do modelo Transformer a ser usado do Hugging Face.
            spacy_model (str): Nome do modelo spaCy para análise linguística.
            limiar_aceitacao (float): Limiar de similaridade para considerar uma resposta 'aceitável'.
            cache_path (str): Caminho para o arquivo de cache de embeddings.
        """
        self.model_name = model_name
        self.spacy_model_name = spacy_model
        self.limiar_otimo = limiar_aceitacao
        self.cache_file = Path(cache_path)
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self.textos_data = {}

        self.embedding_cache = self._load_cache()

    def _load_models(self):
        """Carrega os modelos Transformer e spaCy se ainda não foram carregados."""
        if self.model is None:
            try:
                print(f"Carregando modelo Transformer: {self.model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar modelo Transformer: {e}")

        if self.nlp is None:
            try:
                print(f"Carregando modelo spaCy: {self.spacy_model_name}...")
                self.nlp = spacy.load(self.spacy_model_name, disable=["parser", "ner"])
            except OSError:
                print(f"Modelo spaCy '{self.spacy_model_name}' não encontrado. Baixando...")
                spacy_download(self.spacy_model_name)
                self.nlp = spacy.load(self.spacy_model_name, disable=["parser", "ner"])

    def _load_cache(self) -> dict:
        """
        Carrega o cache de embeddings do disco usando torch.load.

        Returns:
            dict: O dicionário de cache carregado ou um dicionário vazio se não for encontrado.
        """
        if self.cache_file.exists():
            try:
                return torch.load(self.cache_file, map_location=self.device)
            except Exception as e:
                print(f"Aviso: Erro ao carregar cache - {e}")
        return {}

    def _save_cache(self):
        """Salva o cache de embeddings usando torch.save."""
        if self.embedding_cache:
            try:
                torch.save(self.embedding_cache, self.cache_file)
            except Exception as e:
                print(f"Aviso: Erro ao salvar cache - {e}")

    def _get_contextual_embedding(self, contexto: str, palavra: str) -> torch.Tensor:
        """
        Obtém o embedding contextual de uma palavra.

        Args:
            contexto (str): A frase completa com o marcador '[LACUNA]'.
            palavra (str): A palavra a ser inserida na lacuna.

        Returns:
            torch.Tensor: O vetor de embedding contextual normalizado.
        """
        cache_key = f"{self.model_name}::{contexto}::{palavra.strip()}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        if '[LACUNA]' not in contexto or not palavra.strip():
            return torch.zeros(self.model.config.hidden_size, device=self.device)

        frase_preenchida = contexto.replace('[LACUNA]', palavra.strip(), 1)
        inputs = self.tokenizer(frase_preenchida, return_tensors="pt", truncation=True).to(self.device)

        try:
            start_char = contexto.find('[LACUNA]')
            end_char = start_char + len(palavra.strip()) - 1
            start_token = inputs.char_to_token(start_char)
            end_token = inputs.char_to_token(end_char)
            if start_token is None or end_token is None: raise ValueError

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            vecs = outputs.last_hidden_state[0, start_token : end_token + 1]
            embedding = torch.mean(vecs, dim=0)
        except Exception:
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[0, 0, :] # Fallback para o token [CLS]

        normalized_embedding = F.normalize(embedding, p=2, dim=0)
        self.embedding_cache[cache_key] = normalized_embedding.clone().detach()
        return normalized_embedding

    def _get_pos_tag_contextual(self, contexto: str, palavra: str) -> Optional[str]:
        """
        Obtém a classe gramatical contextual da palavra, agrupando AUX em VERB.

        Args:
            contexto (str): A frase completa com o marcador '[LACUNA]'.
            palavra (str): A palavra a ser inserida na lacuna e analisada.

        Returns:
            Optional[str]: A string da POS tag (ex: 'VERB') ou None se ocorrer um erro.
        """
        palavra = str(palavra).strip()
        if '[LACUNA]' not in contexto or not palavra: return None
        
        try:
            frase_preenchida = contexto.replace('[LACUNA]', palavra, 1)
            doc = self.nlp(frase_preenchida)
            start_char = contexto.find('[LACUNA]')
            for token in doc:
                if token.idx == start_char:
                    pos_tag = token.pos_
                    return 'VERB' if pos_tag == 'AUX' else pos_tag
            return None
        except Exception:
            return None

    def _calcular_similaridade_contextual(self, contexto: str, p1: str, p2: str) -> float:
        """
        Calcula a similaridade de cosseno entre dois embeddings contextuais.

        Args:
            contexto (str): A frase que provê o contexto.
            p1 (str): A primeira palavra.
            p2 (str): A segunda palavra.

        Returns:
            float: O score de similaridade de cosseno entre -1.0 e 1.0.
        """
        emb1 = self._get_contextual_embedding(contexto, p1)
        emb2 = self._get_contextual_embedding(contexto, p2)
        return torch.dot(emb1, emb2).item()

    def avaliar_lacuna(self, contexto: str, gabarito: str, resposta: str) -> Tuple[float, str]:
        """
        Avalia uma lacuna individual do teste.

        Args:
            contexto (str): A frase com o marcador '[LACUNA]'.
            gabarito (str): A resposta esperada do gabarito.
            resposta (str): A resposta fornecida pelo aluno.

        Returns:
            Tuple[float, str]: Uma tupla contendo a pontuação (de 0.0 a 1.0) e o tipo da resposta.
        """
        gabarito_norm = str(gabarito).strip().lower()
        resposta_norm = str(resposta).strip().lower()

        if resposta_norm == gabarito_norm:
            return 1.0, 'exata'
        if resposta_norm in ["-", "", " ", "none", "nan"]:
            return 0.0, 'branco'

        max_dist = max(1, len(resposta_norm) // 3)
        if edit_distance(resposta_norm, gabarito_norm) <= max_dist:
            return 1.0, 'grafia_incorreta'
        
        pos_resposta = self._get_pos_tag_contextual(contexto, resposta_norm)
        pos_gabarito = self._get_pos_tag_contextual(contexto, gabarito_norm)
        
        sim = self._calcular_similaridade_contextual(contexto, resposta_norm, gabarito_norm)

        if pos_resposta and pos_gabarito and pos_resposta == pos_gabarito:
            if sim >= self.limiar_otimo:
                return 1.0, 'aceitavel'
            else:
                return 0.5, 'classe_correta'
        
        return 0.0, 'erro'

    def _get_text_data(self, titulo: str, arquivo_textos: str) -> Tuple[list, list]:
        """
        Carrega e armazena em cache os contextos e gabaritos de um texto do arquivo JSON.

        Args:
            titulo (str): O título do texto a ser procurado no JSON.
            arquivo_textos (str): O caminho para o arquivo JSON.

        Returns:
            Tuple[list, list]: Uma tupla contendo (lista de contextos, lista de respostas do gabarito).
        """
        if titulo in self.textos_data:
            return self.textos_data[titulo]
        
        try:
            with open(arquivo_textos, 'r', encoding='utf-8') as f:
                dados_json = json.load(f)
            texto_info = next((item for item in dados_json if item.get("titulo") == titulo), None)
            if not texto_info: raise ValueError
            
            texto_completo = texto_info.get('texto', '')
            respostas = texto_info.get('respostas', [])

            frases_finais, indice_resposta = [], 0
            frases = texto_completo.replace('\n', ' ').split('.')
            for frase in frases:
                num_lacunas = frase.count('[LACUNA]')
                if not num_lacunas: continue
                partes = frase.split('[LACUNA]')
                respostas_locais = respostas[indice_resposta : indice_resposta + num_lacunas]
                for i in range(num_lacunas):
                    preenchimentos = list(respostas_locais)
                    preenchimentos[i] = '[LACUNA]'
                    nova_frase_lista = [val for pair in zip(partes, preenchimentos + ['']) for val in pair]
                    nova_frase = "".join(nova_frase_lista[:-1]).strip()
                    if nova_frase: frases_finais.append(nova_frase)
                indice_resposta += num_lacunas

            self.textos_data[titulo] = (frases_finais, respostas)
            return frases_finais, respostas

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar gabarito/contextos para '{titulo}': {e}")
            
    def avaliar_respostas_cloze(self, titulo_texto: str, arquivo_textos: str,
                                respostas_aluno: Union[List[str], str]) -> Tuple[float, Dict]:
        """
        Avalia todas as lacunas de um teste Cloze para um único respondente.

        Args:
            titulo_texto (str): O título do texto para buscar contextos e gabaritos.
            arquivo_textos (str): O caminho para o arquivo JSON com os dados dos textos.
            respostas_aluno (Union[List[str], str]): Lista de respostas do aluno ou string formatada.

        Returns:
            Tuple[float, Dict]: Uma tupla com (taxa de compreensão, dicionário com avaliação detalhada).
        """
        self._load_models() 
        
        contextos, gabarito = self._get_text_data(titulo_texto, arquivo_textos)
        
        if isinstance(respostas_aluno, str):
            try: respostas_aluno = ast.literal_eval(respostas_aluno)
            except: respostas_aluno = []

        pontuacoes, correcao_tipos = [], []
        for i, gabarito_item in enumerate(gabarito):
            resposta = respostas_aluno[i] if i < len(respostas_aluno) else ""
            contexto = contextos[i] if i < len(contextos) else ""
            
            pontuacao, tipo = self.avaliar_lacuna(contexto, gabarito_item, resposta)
            pontuacoes.append(pontuacao)
            correcao_tipos.append(tipo)

        media = np.mean(pontuacoes) if pontuacoes else 0
        compreensao = round(media * 100, 3)
        
        tipos_possiveis = ['exata', 'grafia_incorreta', 'aceitavel', 'classe_correta', 'erro', 'branco']
        contagens = {tipo: 0 for tipo in tipos_possiveis}
        contagens.update(Counter(correcao_tipos))

        return compreensao, {'pontuacao': pontuacoes, 'correcao': correcao_tipos, **contagens}

    def processar_dataframe(self, df: pd.DataFrame, arquivo_textos: str) -> pd.DataFrame:
        """
        Processa um DataFrame de respostas de múltiplos alunos, aplicando a avaliação contextual.

        Args:
            df (pd.DataFrame): DataFrame com dados originais dos alunos (requer coluna 'respostas').
            arquivo_textos (str): Caminho para o arquivo JSON com os dados dos textos.

        Returns:
            pd.DataFrame: O DataFrame original expandido com todas as colunas de avaliação.
        """
        
        titulo_texto = df["texto"].iloc[0]

        resultados_processados = []
        for _, row in df.iterrows():
            compreensao, avaliacao_detalhada = self.avaliar_respostas_cloze(
                titulo_texto,
                arquivo_textos,
                row['respostas']
            )
            avaliacao_detalhada['compreensao'] = compreensao
            resultados_processados.append(avaliacao_detalhada)
        
        df_avaliacoes = pd.DataFrame(resultados_processados)
        df_resultado = pd.concat([df.reset_index(drop=True), df_avaliacoes], axis=1)

        return df_resultado 

    def __del__(self):
        """Salva o cache em disco ao destruir o objeto ou ao final da sessão."""
        self._save_cache()