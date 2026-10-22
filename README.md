# Guia de Execução do Projeto

Este guia apresenta o passo a passo necessário para clonar, configurar o ambiente e executar o pipeline de classificação automática do projeto **Case Vício**.

> 💡 **Dica de ambiente:** O projeto pode ser executado em ambientes como Kaggle Notebooks e Google Colab, que oferecem suporte a GPUs e à instalação de bibliotecas Python.

---

## Passo a Passo para Execução

### 1. Clonar o Repositório

Clone a branch `desafio-b` do repositório GitHub:

```bash
git clone -b desafio-b --single-branch https://github.com/gaspartino/Case_Vicio.git
```

### 2. Acessar o Diretório do Projeto

Entre na pasta do repositório:

```bash
cd Case_Vicio
```

### 3. Instalar as Dependências

Instale as bibliotecas necessárias para executar o pipeline de classificação, incluindo embeddings e busca vetorial com FAISS:

```bash
pip install -r requirements.txt
```

### 4. Executar o Pipeline Principal

Execute o script principal para iniciar o processamento das questões, a recuperação de temas candidatos e a classificação automática:

```bash
python main.py
```

---

## 📁 Arquivos Gerados

Ao término da execução, o pipeline gera o seguinte arquivo na pasta do projeto:

- **`publicacao.csv`**: contém os resultados da classificação automática das questões, incluindo:
  - `id`: identificador da questão.
  - `decisao`: decisão do sistema, podendo ser `publicar`, `revisar` ou `fora`.
  - `tema_central_id`: identificador do tema central associado à questão.
  - `extras_ids`: identificadores de temas adicionais associados, quando aplicável.
  - `confianca`: valor de confiança registrado pelo sistema para a classificação.
  - `motivo`: justificativa textual, quando aplicável.

O arquivo é gerado a partir das previsões produzidas pelo pipeline. Questões classificadas como `fora` podem apresentar o campo `confianca` vazio, conforme as regras de classificação implementadas.

### 5. Executar o Pipeline em Modo de Teste

O parâmetro `--test` permite avaliar o comportamento do classificador utilizando uma parcela das questões presentes na folha-verdade, em vez de gerar previsões para as questões destinadas à classificação em produção.

Para executar o modo de teste:

```bash
python main.py --test
```

Nesse modo, o pipeline realiza as seguintes etapas:

1. **Separação dos dados:** divide as questões da folha-verdade em conjuntos de treinamento e teste, utilizando 95% para treinamento e 5% para teste.
2. **Preservação da distribuição dos rótulos:** utiliza amostragem estratificada com base no campo `fora_da_taxonomia`, buscando manter a proporção entre as classes nas duas partições.
3. **Treinamento:** constrói os dados de treinamento e ajusta o classificador utilizando apenas a partição de treinamento.
4. **Classificação:** executa as previsões sobre as questões reservadas para teste, que não foram utilizadas no treinamento do classificador.
5. **Exportação dos resultados:** salva as previsões em `publicacao.csv`, permitindo inspecionar as decisões, os temas atribuídos, a confiança e os motivos registrados.

**Importante:** o modo de teste não gera um relatório de métricas de avaliação automaticamente. Embora o código importe `classification_report` e `confusion_matrix`, essas funções não são utilizadas no trecho apresentado. Para avaliar formalmente o desempenho, é necessário comparar as previsões com os rótulos reais da folha-verdade.

A execução normal, sem `--test`, utiliza a folha-verdade completa para treinar o classificador e gera previsões para as questões destinadas à produção:

```bash
python main.py
```

Nesse caso, o arquivo `publicacao.csv` contém as previsões para as questões de produção, enquanto no modo de teste contém somente as previsões das questões reservadas para avaliação.
