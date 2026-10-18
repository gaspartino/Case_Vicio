# Guia de Execução do Projeto

Este guia orienta o passo a passo necessário para clonar, configurar o ambiente e executar o pipeline de classificação automática do projeto **Case Vício**.

> 💡 **Dica de Ambiente:** Os comandos e o código do projeto foram projetados para serem altamente compatíveis com ambientes em nuvem como o **Kaggle Notebooks** e o **Google Colab**, que já oferecem suporte nativo a GPUs e ao gerenciamento de bibliotecas.

---

## Passo a Passo para Execução

### 1. Clonar o Repositório
Faça o download do código-fonte do projeto diretamente do GitHub para o seu ambiente ou notebook:


```

git clone [https://github.com/gaspartino/Case_Vicio.git](https://github.com/gaspartino/Case_Vicio.git)

```

### 2. Acessar o Diretório do Projeto
Entre na pasta do repositório recém-clonado:


```

cd Case_Vicio

```

### 3. Instalar as Dependências
Instale as bibliotecas necessárias para a execução do pipeline de embeddings, FAISS e LLM:


```

pip install -r requirements.txt

```

### 4. Executar o Pipeline Principal
Rode o script principal para iniciar a extração, busca por similaridade e classificação das questões:


```

python main.py

```

---

## 📁 Arquivos Gerados
Após o término da execução (ou durante o salvamento progressivo a cada 10 questões), o script gerará automaticamente os seguintes arquivos de saída na pasta do projeto:

* **`publicacao.csv`**: Contém o resultado do processamento com a decisão (`publicar`, `revisar` ou `fora`), nota de confiança, justificativa e o tema associado.
* **`candidatos.csv`**: Registra os temas candidatos recuperados pela etapa de similaridade vetorial (FAISS) para cada questão.
