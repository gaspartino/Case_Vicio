# Relatório Técnico: Classificação Automação Temática de Questões Jurídicas

**Projeto:** Case Vício

**Repositório:** [github.com/gaspartino/Case_Vicio](https://github.com/gaspartino/Case_Vicio)

---

## 1. Como Você Pensou (Raciocínio, Hipóteses e Becos Sem Saída)

### 1.1. Hipóteses e Arquitetura do Raciocínio

A ideia central foi construir uma solução inspirada na arquitetura **RAG (Retrieval-Augmented Generation)** com pipeline em duas etapas:

1. **Recuperação Semanticamente Orientada (Retrieval):** Utilizar embeddings e uma base vetorial (FAISS) para filtrar os temas mais aderentes à questão jurídica.
2. **Raciocínio Jurídico Fino (Generation / Decision):** Utilizar um modelo de linguagem (LLM) para analisar a questão completa (enunciado, comando, alternativas e informações auxiliares) contra os temas candidatos retornados.

---

### 1.2. Becos Sem Saída e Testes Descartados

* **Modelo de Embedding Inicial (`all-MiniLM-L6-v2`):**
* *Hipótese:* Usar um modelo leve e consagrado para geração de vetores.
* *Resultado:* O modelo não obteve bons resultados na captura das nuances e vocabulário técnico da linguagem jurídica em português, gerando candidatos pouco precisos.
* *Decisão:* Descartado em favor de um modelo de embedding mais robusto e capaz de representar similaridade temática em domínio jurídico.


* **Modelo de Linguagem (`Qwen/Qwen3-4B`):**
* *Hipótese:* Utilizar o `Qwen/Qwen3-4B` para extrair capacidade superior de raciocínio, análise de alternativas e contexto amplo.
* *Resultado:* O modelo esbarrou na restrição de **VRAM da GPU** no ambiente gratuito do Kaggle, provocando erros de estouro de memória (*Out of Memory - OOM*).
* *Decisão:* Descartado devido às limitações do ambiente de execução, direcionando a escolha para LLMs de menor porte otimizadas para o limite de VRAM disponível sem custo.



---

### 1.3. Visão Geral da Solução Desenvolvida

O pipeline executa as seguintes etapas no arquivo `main.py`:

```
[ Questão Jurídica ]
        │
        ▼
┌──────────────────────────────────────────────┐
│  Etapa 1: Recuperação por Similaridade      │
│  - Gera embedding da questão                 │
│  - Busca vetorial com FAISS (`top_k=15`)     │
└──────────────────────────────────────────────┘
        │
        ▼  (Temas Candidatos)
┌──────────────────────────────────────────────┐
│  Etapa 2: Classificação e Decisão com LLM    │
│  - Avalia enunciado, comando e alternativas  │
│  - Compara com títulos e descrições dos temas│
│  - Retorna ID, Decisão, Confiança e Motivo   │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Etapa 3: Validação, Retry e Persistência    │
│  - Formato e presença na lista de candidatos │
│  - Segunda tentativa em caso de falha        │
│  - Gravação progressiva (`publicacao.csv`)   │
└──────────────────────────────────────────────┘

```

### 1.4. Regras de Decisão da LLM

* **`publicar`:** Correspondência temática segura e precisa.
* **`revisar`:** Existência de candidato plausível, mas com incertezas residuais.
* **`fora`:** Nenhuma opção representa adequadamente o conteúdo jurídico cobrado.

---

### 1.5. Gestão de Limitações de Recursos e Ambiente

O projeto foi executado integralmente em ambiente gratuito (Kaggle), cujas especificidades técnicas do ambiente configuraram restrições diretas e decisões de engenharia:

#### A. Especificações do Ambiente de Execução

* **Sistema Operacional:** Linux-6.18.48+-x86_64-with-glibc2.39
* **Python:** 3.13.15
* **Framework de IA:** PyTorch 2.11.0+cu128 (com CUDA 12.8 habilitado)
* **Hardware de Aceleração:** 2x GPUs NVIDIA Tesla T4 (Memória total de 14.56 GiB por GPU)

#### B. Impacto na Arquitetura e Mitigação

* **Restrição de VRAM:** Embora o hardware disponha de duas unidades Tesla T4 com 14.56 GiB cada, os requisitos de inferência simultânea para modelos maiores (como o `Qwen/Qwen3-4B`) excederam o limite viável para uma execução estável em lote sem estouro de memória, exigindo o dimensionamento para modelos menores ou quantizados.
* **Resiliência e Salvamento Progressivo:** Para contornar possíveis interrupções de sessão inerentes a ambientes de infraestrutura gratuita, o script foi programado para salvar os resultados parcial e progressivamente a cada 10 questões (`publicacao.csv` e `candidatos.csv`).
* **Tratamento de Exceções:** Implementou-se rotinas de *retry* duplo para as respostas da LLM e blocos de captura de exceção para que falhas pontuais em uma única questão não corrompessem ou interrompessem o lote completo.

---

## 2. Diagnóstico: Dificuldades e Gargalos no Classificador

O desempenho do sistema de classificação enfrenta gargalos específicos inerentes à estrutura dos dados e à natureza das questões jurídicas:

### 2.1. Limitações na Definição dos Temas

* **Ausência de Descrições Detalhadas:** A principal limitação do *dataset* é a falta de uma descrição elaborada para cada tema (restringindo-se apenas ao caminho da hierarquia de assuntos fornecido). Sem um detalhamento conceitual, o modelo perde contexto sobre a abrangência real de cada tema.
* **Sobreposição e Similaridade Temática:** Existem múltiplos temas com alta similaridade semântica entre si. A falta de descrições conceituais faz com que tanto o modelo de *embedding* quanto a LLM fiquem dependentes puramente da similaridade textual direta, dificultando a diferenciação fina entre assuntos correlatos.

### 2.2. Complexidade e Extensão das Questões

Ambos os extremos quanto ao tamanho do enunciado geram desafios distintos tanto para o modelo quanto para um avaliador humano:

* **Questões Extensas (Excesso de Informação):** Apresentam casos práticos longos, contextos fáticos detalhados e narrativas complexas. O excesso de detalhes secundários pode atuar como "ruído", desviando a atenção da LLM do núcleo jurídico principal e confundindo o mapeamento para o tema correto.
* **Questões Curtas (Escassez de Contexto):** Apresentam comandos extremamente curtos ou diretos, oferecendo pouquíssimas pistas semânticas. Nesses casos, o modelo de *embedding* pode gerar representações vetoriais genéricas, dificultando a recuperação do tema correto pelo FAISS.

---

## 3. Arquitetura de Produção e Escalabilidade

Considerando o cenário em que o pipeline precise processar **50 mil questões semanais**, a abordagem atual — baseada em inferência sequencial com LLM de uso geral em ambiente gratuito (onde a execução de 359 questões demandou cerca de 3 horas) — torna-se inviável, exigindo uma reestruturação para alta performance e escalabilidade industrial.

### 3.1. Tipos de IA e Abordagem Utilizada
* **Embeddings + Vetor Search em Lote (Milvus / Qdrant):** Manutenção da recuperação vetorial de primeira etapa, otimizada para buscas rápidas em índices distribuídos de alta escala.
* **Classificador Especialista Treinado (Fine-Tuning / Machine Learning Tradicional):** Substituição ou complemento da LLM de uso geral por um classificador leve baseado em arquiteturas de *Encoder* (como modelos BERT especializados em português jurídico, ex: BERT-imbabel ou similar) fine-tuned especificamente com o catálogo de temas. Isso elimina a latência e o custo de inferência de uma LLM generativa pura para a grande maioria dos casos claros.
* **LLM como Orquestrador / Agente de Exceção:** Uso de LLMs menores, porém altamente destiladas (ou APIs de alta performance com *batch processing*), ativadas **apenas** quando o classificador indicar baixa confiança ou na zona de fronteira (casos de revisão).
* **Agentes Especializados e Revisão Humana:**
  * *Agentes de Validação:* Agentes auxiliares que cruzam o resultado com metadados e histórico de questões similares.
  * *Revisão Humana em Loop (Human-in-the-Loop):* Apenas as questões classificadas com a decisão `revisar` ou com baixa confiança são direcionadas para um painel operacional onde especialistas validam e corrigem os dados, retroalimentando o modelo.

### 3.2. Estimativa de Tempo por Questão e Projeção de Escalabilidade

* **Análise da Linha de Base (Ambiente Atual):** No ambiente gratuito utilizado (Kaggle), o processamento sequencial de **359 questões** demandou cerca de **3 horas** (aproximadamente **30 segundos por questão**). Esse tempo elevado reflete a execução sequencial síncrona, a sobrecarga de chamadas de LLM por API/modelo local e a ausência de paralelismo em lote.
* **Projeção para 50 Mil Questões com Abordagem Sequencial Atual:** Se mantido o mesmo modelo e infraestrutura atual (30s por item), processar 50 mil questões levaria cerca de **416 horas** (mais de 17 dias ininterruptos de processamento), o que torna a execução sequencial inviável para uma rotina semanal.
### 3.3. Estimativa de Tempo por Questão e Projeção de Escalabilidade

* **Análise da Linha de Base (Ambiente Atual):** No ambiente gratuito utilizado (Kaggle), o processamento sequencial de **359 questões** demandou cerca de **3 horas** (aproximadamente **30 segundos por questão**). Esse tempo elevado reflete a execução sequencial síncrona, a sobrecarga de chamadas de LLM por API/modelo local, a ausência de paralelismo em lote e as limitações do ambiente de execução gratuito.
* **Necessidade de Melhoria do Ambiente de Execução:** Para suportar altas demandas produtivas, é imprescindível evoluir a infraestrutura de hardware e software — saindo do ambiente restrito do Kaggle para instâncias de computação em nuvem dedicadas e robustas (com GPUs de maior capacidade, maior largura de banda de memória e recursos elásticos de escalabilidade).
* **Projeção para 50 Mil Questões com Abordagem Sequencial Atual:** Se mantido o mesmo modelo e infraestrutura atual (30s por item), processar 50 mil questões levaria cerca de **416 horas** (mais de 17 dias ininterruptos de processamento), o que torna a execução sequencial inviável para uma rotina semanal.
* **Otimização de Tempo em Produção (Paralelismo e Modelos Leves):** Substituindo a arquitetura atual por processamento assíncrono em lote (*batch processing*), melhorias no ambiente de execução, instâncias escaláveis em nuvem e classificadores leves (ou modelos locais otimizados com inferência concorrente), o tempo médio por questão poderia cair para menos de **0,2 a 0,5 segundos**. Dessa forma, o lote semanal de 50 mil questões pode ser concluído em **poucos minutos** (cerca de 3 a 5 horas de processamento distribuído ou consideravelmente menos dependendo da paralelização), tornando o fluxo perfeitamente viável para o ciclo semanal de importação.

### 3.4. Métricas de Qualidade Contínua (MLOps)
* **Acurácia em Amostra Cega:** Monitoramento contínuo de uma fração aleatória (ex: 5%) das questões publicadas que passam por validação humana cega.
* **Taxa de Rejeição e Desvio (*Drift*):** Acompanhamento métrico da proporção de questões enviadas para a fila de `revisar` ou `fora` para detectar mudanças no perfil das bancas de concursos ou obsolescência dos temas.
* **Feedback de Alunos (Reportes):** Monitoramento automatizado de feedbacks e contestações enviadas por alunos na plataforma para identificar erros de classificação em produção e transformá-los automaticamente em novos dados de treinamento (*hard negatives*).

### 3.5. Integração no Fluxo de Importação, Revisão e Relatórios
* **Pipeline Assíncrono Orientado a Eventos:** Assim que um lote de questões é importado para a base, ele é enfileirado em um sistema de mensageria (ex: Kafka ou filas assíncronas) para processamento paralelo em lote (*batch processing*), reduzindo o tempo total de processamento de horas para poucos minutos.
* **Painel de Triagem para Revisores:** Uma interface integrada onde as questões com decisão de `revisar` aparecem organizadas por ordem de urgência ou menor nível de confiança da IA, agilizando o trabalho da equipe pedagógica.
* **Geração Automatizada de Relatórios:** Atualização automática dos dashboards de acompanhamento de importação e publicação, alimentando relatórios executivos semanais sobre a saúde do banco de questões e a produtividade da curadoria.

## 4. Memorando para o Professor / Especialista Jurídico

**De:** Equipe de Desenvolvimento de IA  
**Para:** Professor / Curador do Conteúdo Jurídico  
**Assunto:** Diagnóstico do Algoritmo de Classificação Automática e Recomendações para o Catálogo de Temas  

---

### 4.1. O que Encontramos ao Processar as Questões
Desenvolvemos um sistema inteligente para analisar as questões jurídicas e classificá-las automaticamente dentro do seu catálogo de temas. Ao testar o modelo em um conjunto de questões, observamos os seguintes comportamentos:

* **Precisão em temas bem definidos:** Quando a questão trata de um assunto direto e bem delimitado, a IA consegue identificar o tema correto com alta segurança (`publicar`).
* **Dificuldade com "ruídos" e contextos longos:** Em questões com historinhas práticas extensas ou múltiplos detalhes secundários, o sistema por vezes se confunde e desvia do núcleo do assunto cobrado.
* **Empasse em temas parecidos:** Quando dois ou mais temas do catálogo possuem nomes muito similares, o sistema hesita e tende a enviar a questão para a categoria `revisar`.

---

### 4.2. O que Recomendamos Mudar no Catálogo de Temas
Para que o sistema de inteligência artificial atinja a meta de 95%+ de precisão e reduza a necessidade de revisão humana, recomendamos as seguintes melhorias na estrutura dos dados:

1. **Inclusão de Descrições Conceituais nos Temas:** Atualmente, o catálogo fornece apenas o nome e o caminho do assunto (ex: *Direito Civil > Contratos > Espécies*). Precisamos de uma breve descrição jurídica (2 a 3 frases) explicando **o que está coberto** e **o que não está coberto** em cada tema.
2. **Fagocitação ou Fusão de Temas Sobrepostos:** Identificamos temas muito similares na árvore do catálogo que causam margem de dúvida até para um especialista. Recomendamos unificar esses tópicos ou delimitar claramente suas fronteiras.
3. **Exemplos-Chave (Questões de Referência):** Associar 2 a 3 questões-exemplo definitivas a cada tema ajudará muito o modelo a entender o padrão de cobrança daquele assunto.

---

### 4.3. Como Você (Especialista em Direito) Pode nos Ajudar
Para avançarmos para a próxima etapa do projeto, precisamos do seu apoio nas seguintes frentes pedagógicas:

* **Validação do Lote de Revisão:** Avaliar as questões que o sistema classificou como `revisar` para nos dizer se a dúvida da IA fazia sentido ou se era um caso claro.
* **Elaboração/Revisão dos Resumos dos Temas:** Auxiliar na criação do pequeno resumo explicativo para os temas que mais geraram divergências durante os testes.
* **Definição de Regras de Desempate:** Nos orientar sobre qual critério jurídico priorizar quando uma questão abordar simultaneamente dois temas correlatos.
