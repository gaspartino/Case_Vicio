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

### 5. Executar o Pipeline por Modo de Operação

O parâmetro `--mode` permite selecionar entre três modos de execução do pipeline: `test`, `all` e `prod`. Cada modo define quais questões serão utilizadas no treinamento e quais receberão previsões.

#### 5.1. Modo de Teste (`test`)

O modo `test` permite avaliar o comportamento do classificador utilizando uma parcela das questões presentes na folha-verdade.

Para executar:

```bash
python main.py --mode test
```

Nesse modo, o pipeline realiza as seguintes etapas:

1. **Separação dos dados:** divide as questões da folha-verdade em conjuntos de treinamento e teste, utilizando 95% para treinamento e 5% para teste.
2. **Preservação da distribuição dos rótulos:** utiliza amostragem estratificada com base no campo `fora_da_taxonomia`, buscando manter a proporção entre as classes nas duas partições.
3. **Treinamento:** constrói os dados de treinamento e ajusta o classificador utilizando apenas a partição de treinamento.
4. **Classificação:** gera previsões para as questões reservadas para teste, que não devem participar do treinamento.
5. **Exportação dos resultados:** salva as previsões em `publicacao.csv`, permitindo inspecionar as decisões, os temas atribuídos, a confiança e as justificativas registradas.

**Importante:** o pipeline não gera automaticamente métricas de avaliação. Para avaliar formalmente o desempenho, é necessário comparar as previsões com os rótulos reais das questões de teste.

#### 5.2. Modo Completo (`all`)

O modo `all` gera previsões para todas as questões presentes em `data/questoes.jsonl`, sem distinguir entre questões previamente utilizadas no treinamento e questões destinadas à produção.

Para executar:

```bash
python main.py --mode all
```

Nesse modo, o classificador é treinado utilizando a folha-verdade completa e as questões disponíveis no conjunto de dados, conforme a construção dos dados de treinamento implementada no pipeline. Em seguida, são geradas previsões para todas as questões.

Esse modo é útil quando se deseja obter uma classificação completa do conjunto, inclusive para questões que já possuem informações na folha-verdade.

**Importante:** como as questões previstas também podem ter participado do treinamento, os resultados desse modo não devem ser utilizados, isoladamente, para estimar a capacidade de generalização do classificador para questões inéditas.

#### 5.3. Modo de Produção (`prod`)

O modo `prod` corresponde à execução padrão do pipeline. Utiliza a folha-verdade completa para treinar o classificador e gera previsões apenas para as questões destinadas à produção, conforme a separação realizada por `split_questions_by_training_data`.

Para executar explicitamente:

```bash
python main.py --mode prod
```

Como `prod` é o modo padrão, também é possível executar o pipeline sem informar o parâmetro:

```bash
python main.py
```

Nesse modo, as questões presentes na folha-verdade são utilizadas na construção dos dados de treinamento, enquanto as questões separadas para produção são submetidas à classificação.

#### 5.4. Resumo dos modos de execução

| Modo | Treinamento | Questões previstas |
|---|---|---|
| `test` | Partição de treinamento da folha-verdade | Partição de teste (5%) |
| `all` | Folha-verdade e questões utilizadas na construção dos dados de treinamento | Todas as questões |
| `prod` | Folha-verdade completa | Apenas questões destinadas à produção |

Os três modos exportam os resultados para `publicacao.csv`. Portanto, cada nova execução pode sobrescrever o arquivo gerado anteriormente.
```

Nesse caso, o arquivo `publicacao.csv` contém as previsões para as questões de produção, enquanto no modo de teste contém somente as previsões das questões reservadas para avaliação.
