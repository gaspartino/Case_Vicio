# Relatório Técnico: Classificação Automática de Questões Jurídicas por Similaridade Semântica e Aprendizado Supervisionado

**Projeto:** Case Vício

**Repositório:** [github.com/gaspartino/Case_Vicio](https://github.com/gaspartino/Case_Vicio)

---

## 1. Como a Solução Foi Concebida: Hipóteses, Testes e Decisões de Arquitetura

### 1.1. Hipóteses e Evolução da Arquitetura

A solução foi desenvolvida para automatizar a classificação temática de questões jurídicas, identificando tanto questões que podem ser associadas aos temas existentes quanto aquelas que não encontram correspondência adequada no catálogo.

A principal dificuldade inicial foi a disponibilidade limitada de dados rotulados: a ficha `folha_verdade.csv` contém aproximadamente 100 exemplos de referência, enquanto o catálogo utilizado na classificação possui cerca de 377 temas possíveis.

A primeira abordagem consistiu em construir dois classificadores supervisionados independentes:

1. **Classificador de pertinência à taxonomia:** responsável por identificar se a questão pertence ao escopo dos temas disponíveis ou deve ser classificada como fora da taxonomia.
2. **Classificador temático:** responsável por identificar o tema jurídico específico associado à questão.

Entretanto, a quantidade reduzida de exemplos em relação ao número de temas dificultou o treinamento do segundo classificador. Mesmo após testar diferentes configurações e hiperparâmetros, os resultados não apresentaram qualidade suficiente para sustentar uma classificação temática confiável.

Esse problema motivou a adoção de uma arquitetura híbrida, que preserva o classificador binário para a identificação de questões fora da taxonomia e substitui a classificação temática supervisionada direta por uma estratégia de recuperação semântica.

O fluxo atual é composto por três etapas:

- **Classificação supervisionada binária:** estima a probabilidade de a questão estar fora da taxonomia.
- **Recuperação temática por embeddings e FAISS:** identifica os temas semanticamente mais próximos da questão.
- **Decisão baseada em limiares:** utiliza as probabilidades e similaridades calculadas para determinar se a questão deve ser publicada, revisada ou classificada como fora da taxonomia.

Dessa forma, a arquitetura reduz a dependência de exemplos rotulados para cada tema individual, embora continue sujeita à qualidade das representações semânticas e à adequação dos limiares utilizados.

### 1.2. Abordagens Testadas e Limitações Encontradas

#### A. Dois classificadores supervisionados

**Hipótese:** utilizar um classificador para a detecção de questões fora da taxonomia e outro para selecionar o tema específico.

**Limitação encontrada:** o classificador binário possui um objetivo relativamente restrito, enquanto o classificador temático precisa distinguir entre um conjunto muito maior de categorias. Com aproximadamente 100 exemplos de referência e cerca de 377 temas possíveis, a cobertura dos temas no conjunto de treinamento mostrou-se insuficiente para sustentar uma aprendizagem supervisionada temática robusta.

A quantidade reduzida de exemplos também dificulta a representação da diversidade de enunciados, comandos, alternativas e formas de cobrança associadas a cada assunto.

**Decisão:** manter o classificador binário e substituir a classificação temática supervisionada direta pela recuperação semântica dos temas.

#### B. Ajustes de hiperparâmetros no classificador temático

**Hipótese:** melhorar o desempenho por meio de diferentes configurações dos modelos de classificação.

**Resultado:** os testes de parâmetros não resolveram a limitação fundamental da quantidade e da distribuição dos exemplos rotulados.

O problema não estava necessariamente restrito à escolha do algoritmo ou de seus hiperparâmetros. Um modelo supervisionado não consegue aprender de maneira confiável as fronteiras entre centenas de temas quando dispõe de poucos exemplos e não possui representações adequadas de todas essas categorias.

**Decisão:** alterar a estratégia de classificação, em vez de continuar concentrando os esforços exclusivamente no ajuste dos parâmetros.

#### C. Recuperação semântica com embeddings e FAISS

**Hipótese:** representar as questões e os temas em um espaço vetorial compartilhado, recuperando os temas mais semelhantes por similaridade de cosseno.

**Implementação:** o catálogo ativo é convertido em embeddings, normalizados e armazenados em um índice `faiss.IndexFlatIP`. Como os vetores são normalizados, o produto interno utilizado pelo índice corresponde à similaridade de cosseno.

Na inferência, o sistema recupera os três candidatos mais semelhantes para a etapa de decisão temática, embora a função de recuperação aceite outros valores de `top_k`.

**Vantagem esperada:** a abordagem permite pesquisar entre os temas do catálogo sem exigir um exemplo rotulado de treinamento para cada categoria.

**Limitação:** similaridade semântica não equivale necessariamente à correção jurídica. Temas com vocabulário semelhante podem tratar de institutos distintos, e o melhor candidato recuperado ainda pode ser incorreto.

### 1.3. Visão Geral do Fluxo Atual

O processamento é coordenado pelo arquivo `main.py`, com apoio dos métodos definidos em `utils.py` e dos modelos disponibilizados por `model.py`.

```text
                 [ Questão Jurídica ]
                          |
                          v
              [ Preparação do Texto ]
              - Enunciado
              - Comando
              - Alternativas
                          |
                          v
                 [ Geração do Embedding ]
                          |
                          v
             [ Classificador Supervisionado ]
              Probabilidade de estar fora
                          |
               +----------+----------+
               |                     |
               v                     v
        Fora da Taxonomia       Dentro / Incerto
               |                     |
               v                     v
          Decisão "fora"      [ Busca FAISS ]
                              Recupera os temas
                              mais semelhantes
                                      |
                                      v
                             [ Limiar Temático ]
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                 Similaridade suficiente    Insuficiente
                         |                         |
                         v                         v
                    "publicar"                 "revisar"
```

O fluxo possui duas fontes distintas de evidência: a probabilidade estimada pelo classificador binário e a similaridade entre a questão e os temas recuperados. Esses valores representam grandezas diferentes e não devem ser interpretados como se fossem uma única medida de confiança.

### 1.4. Regras de Decisão do Sistema

O método `predict_question` implementa as seguintes regras:

- **`fora`:** quando a probabilidade estimada pelo classificador para a classe positiva `fora_da_taxonomia = 1` é igual ou superior a `outside_threshold`, configurado inicialmente como `0.50`.
- **`publicar`:** quando a questão não é classificada como fora e pelo menos um dos três temas recuperados apresenta similaridade igual ou superior a `theme_threshold`, configurado inicialmente como `0.875`.
- **`revisar`:** quando a questão não é classificada como fora, mas nenhum dos candidatos recuperados atinge o limiar temático de publicação.

O tema com maior similaridade é registrado como `tema_central_id`. Os candidatos que atingem o limiar temático podem ser registrados em `extras_ids`, excluindo-se o tema central.

O campo `confianca` recebe a similaridade do melhor candidato, arredondada para três casas decimais. Portanto, esse campo representa uma medida de proximidade semântica, e não uma probabilidade calibrada de acerto.

Quando o melhor candidato apresenta similaridade superior a `0.80` e existe um segundo candidato, o código registra inicialmente uma justificativa de possível dúvida entre os temas. Essa justificativa é removida caso a questão seja aprovada para publicação.

É importante observar que a decisão `publicar` depende do limiar individual de similaridade, não da diferença entre o primeiro e o segundo candidatos. Assim, a regra atual não garante que o tema central esteja suficientemente separado dos demais candidatos.

### 1.5. Gestão dos Dados e Preparação das Questões

O método `build_question_text` reúne as informações disponíveis sobre cada questão:

- Enunciado;
- Comando;
- Alternativas.

Antes de gerar o texto final, o método remove tags HTML, converte entidades HTML e normaliza os espaços em branco.

O método `encode_question` transforma esse texto em uma representação vetorial, utilizando uma instrução voltada à recuperação de temas jurídicos. O mesmo padrão de consulta é aplicado aos exemplos usados no treinamento do classificador binário.

O método `create_theme_index` prepara as representações dos temas ativos, combinando título, caminho hierárquico e descrição. O índice vetorial permite pesquisar os candidatos sem executar uma comparação manual entre a questão e todos os temas durante cada decisão.

O treinamento do classificador binário utiliza os exemplos da `folha_verdade.csv`, enriquecidos com os dados correspondentes de `questoes.jsonl` e `temas.csv`.

### 1.6. Separação entre Treinamento, Avaliação e Produção

O fluxo utiliza `split_questions_by_training_data` para separar as questões identificadas na ficha de referência das demais questões disponíveis.

No modo normal, o classificador é treinado com os dados rotulados disponíveis e as previsões são geradas para as questões que não pertencem à ficha de referência.

No modo de teste, ativado por `--test`, os exemplos de referência são divididos entre treinamento e avaliação, com estratificação pela variável `fora_da_taxonomia`. O código gera as previsões para o subconjunto de teste e salva o resultado em `publicacao.csv`.

Essa avaliação permite observar o desempenho do classificador no objetivo binário, mas exige cautela na interpretação: a separação atual não garante, por si só, uma avaliação independente da qualidade temática. Além disso, a chamada a `train_test_split` não fixa `random_state`, de modo que a partição pode variar entre execuções.

A avaliação deve considerar separadamente:

1. A capacidade de reconhecer questões fora da taxonomia;
2. A capacidade de recuperar o tema correto entre os candidatos;
3. A precisão das questões efetivamente publicadas;
4. A proporção de questões encaminhadas para revisão.

---

## 2. Diagnóstico: Dificuldades e Gargalos do Classificador

### 2.1. Desproporção entre Exemplos Rotulados e Temas Disponíveis

A principal limitação encontrada durante o desenvolvimento é a diferença entre a quantidade de exemplos de referência e a quantidade de categorias temáticas disponíveis.

A ficha `folha_verdade.csv` contém aproximadamente 100 exemplos, enquanto o catálogo possui cerca de 377 temas possíveis. Essa relação não significa necessariamente que todos os temas precisem de exemplos de treinamento para serem recuperados semanticamente, mas representa um obstáculo importante para uma classificação supervisionada direta.

Os exemplos disponíveis também podem estar concentrados em determinados assuntos, sem representar adequadamente a diversidade do catálogo.

Por isso, o classificador temático inicial não apresentou resultados satisfatórios mesmo após vários testes de parâmetros.

A estratégia atual contorna parcialmente essa limitação por meio da recuperação semântica, que permite comparar as questões com as representações textuais de todos os temas ativos. Entretanto, essa mudança não elimina a necessidade de avaliar se o tema correto aparece entre os candidatos recuperados.

### 2.2. Qualidade e Detalhamento do Catálogo

A recuperação temática utiliza o título, o caminho hierárquico e a descrição de cada tema.

Quando a descrição está vazia, genérica ou pouco informativa, o embedding depende principalmente do título e do caminho. Isso reduz a quantidade de informações disponíveis para distinguir categorias próximas.

A hierarquia ajuda a contextualizar os temas, mas não substitui uma descrição conceitual que delimite o conteúdo abrangido e suas diferenças em relação a categorias semelhantes.

A melhoria do catálogo pode beneficiar tanto a recuperação dos candidatos quanto a futura construção de um conjunto de treinamento mais representativo.

### 2.3. Sobreposição entre Temas Juridicamente Próximos

O catálogo pode conter temas que compartilham terminologia, institutos jurídicos ou contextos de aplicação.

Nessas situações, uma questão pode apresentar alta similaridade com várias categorias, sem que o candidato mais próximo seja necessariamente o correto.

A utilização de um limiar fixo de `0.875` reduz a publicação de questões com similaridade inferior a esse valor, mas não assegura precisão de 95% ou superior. O desempenho depende da distribuição real das similaridades, da qualidade dos embeddings e da frequência com que os candidatos recuperados estão corretos.

Também é necessário distinguir a identificação do tema central da seleção de temas extras. O fato de um tema ultrapassar o limiar de similaridade não comprova, isoladamente, que ele deva ser associado à questão.

### 2.4. Questões Curtas e Extensas

Questões curtas podem apresentar poucas pistas semânticas, dificultando a recuperação de um tema específico. Um comando como “Assinale a alternativa correta”, por exemplo, possui pouca informação temática quando analisado isoladamente.

Questões extensas podem incluir fatos secundários, nomes de personagens, datas, exceções e múltiplos institutos jurídicos. Esses elementos podem afetar a representação vetorial, ainda que o texto contenha informações suficientes para um especialista identificar o assunto central.

A inclusão do enunciado, do comando e das alternativas reduz a dependência de um único campo, mas não resolve completamente os problemas de ambiguidade e de relevância dos diferentes trechos.

### 2.5. Limitações do Classificador de Questões Fora da Taxonomia

O classificador binário utiliza um modelo de aprendizado supervisionado treinado com embeddings das questões e rótulos `0` e `1`.

O método `predict_question` considera a probabilidade da classe `1` para aplicar o limiar `outside_threshold`. A qualidade dessa decisão depende da quantidade e da diversidade dos exemplos rotulados, da representação das questões e da calibração do classificador.

Como a ficha de referência é pequena, mudanças na composição do conjunto de treinamento podem alterar significativamente as probabilidades estimadas.

Além disso, uma questão pode pertencer ao domínio jurídico e ainda assim não corresponder a um tema específico do catálogo. Essa distinção precisa estar bem representada nos exemplos usados para treinar o classificador.

Consequentemente, o limiar de `0.50` deve ser tratado como um parâmetro inicial a ser validado empiricamente, e não como um ponto de corte universalmente adequado.

### 2.6. Limitações da Avaliação Atual

A avaliação deve separar o desempenho binário do desempenho temático. Um sistema pode identificar corretamente que uma questão pertence à taxonomia e, ainda assim, associá-la ao tema errado.

Da mesma forma, um sistema pode obter alta precisão entre as questões publicadas e encaminhar muitas outras para revisão. Essa situação pode ser aceitável quando a prioridade é reduzir erros de publicação, mas precisa ser medida explicitamente.

Para uma avaliação mais consistente, recomenda-se acompanhar:

- Precisão, revocação e F1-score da classificação fora/dentro da taxonomia;
- Taxa de recuperação do tema correto entre os três candidatos;
- Precisão temática das questões publicadas;
- Cobertura de publicação, calculada como a fração de questões publicadas;
- Taxa de revisão e taxa de classificação como fora;
- Resultados separados por disciplina e por grupos de temas.

Os resultados devem ser calculados sobre um conjunto de avaliação independente, com rótulos temáticos confiáveis. Os aproximadamente 100 exemplos existentes podem fornecer uma primeira estimativa, mas não necessariamente permitem conclusões robustas sobre todas as categorias do catálogo.

---

## 3. Arquitetura de Produção e Escalabilidade

A arquitetura atual foi concebida para processar um conjunto limitado de questões em um ambiente de experimentação. Para atender a uma demanda de 50 mil questões semanais, seria necessário avaliar o custo computacional, a capacidade de paralelização e os requisitos de qualidade antes de definir a infraestrutura definitiva.

### 3.1. Tipos de IA e Abordagem Utilizada

A solução atual combina dois mecanismos complementares:

**1. Classificador supervisionado binário**

Um modelo de aprendizado de máquina recebe os embeddings das questões e estima se elas estão fora da taxonomia. Esse mecanismo é relativamente leve em comparação com a geração de respostas por uma LLM, mas sua confiabilidade depende da qualidade dos dados rotulados.

**2. Embeddings e busca vetorial**

Um modelo de embeddings representa semanticamente as questões e os temas. O FAISS recupera os candidatos com maior similaridade, permitindo pesquisar em um catálogo com centenas de categorias sem treinar um classificador independente para cada tema.

O índice atual utiliza `IndexFlatIP`, que realiza busca exata por produto interno. Para aproximadamente 377 temas, essa abordagem é simples e adequada como ponto de partida. A utilização de um banco vetorial distribuído, como Qdrant ou Milvus, só deve ser considerada quando houver requisitos de escala, atualização, concorrência ou operação que justifiquem essa complexidade adicional.

**3. Decisão por limiares e revisão humana**

A decisão final é determinada pelos limiares de probabilidade e similaridade. Questões que não atendem aos critérios de publicação são encaminhadas para revisão, enquanto as classificadas como fora da taxonomia recebem uma decisão própria.

Essa arquitetura dispensa uma LLM generativa na inferência atual, reduzindo a dependência de geração de texto, mas também deixa de contar com uma etapa explícita de raciocínio jurídico generativo.

### 3.2. Linha de Base e Projeção para 50 Mil Questões

Na arquitetura atual, o processamento de 659 questões levou aproximadamente 1 minuto, evidenciando uma redução expressiva no tempo de execução em comparação com a abordagem anterior, baseada em inferência generativa por questão.

Considerando essa medição como referência, temos:

- **Tempo de processamento:** aproximadamente 1 minuto para 659 questões.
- **Tempo médio por questão:** aproximadamente 0,091 segundo.
- **Volume semanal projetado:** 50.000 questões.
- **Tempo total estimado:** aproximadamente 75,9 minutos, ou 1 hora e 16 minutos.
- **Duração equivalente:** cerca de 1,27 hora de processamento contínuo.

Essa projeção considera uma extrapolação linear do desempenho observado no novo pipeline, mantendo as mesmas condições de execução. O tempo real poderá variar conforme o tamanho dos enunciados, a capacidade computacional disponível, o volume de dados processados e a sobrecarga de leitura e gravação dos resultados.

É importante destacar que a estimativa se refere ao fluxo atual, baseado em embeddings, classificação supervisionada binária e recuperação temática por FAISS, sem a etapa de inferência generativa utilizada na abordagem anterior.

### 3.3. Otimização de Tempo em Produção

Embora o novo pipeline apresente um desempenho significativamente superior ao da abordagem anterior, algumas otimizações podem melhorar sua eficiência e preparar a solução para um ambiente de produção com maior volume de dados.

**A. Inferência em lote para embeddings**

A geração de embeddings em lotes pode aumentar a utilização dos recursos computacionais e reduzir a sobrecarga associada ao processamento individual das questões. A implementação deve considerar o tamanho dos lotes e a memória disponível para evitar gargalos.

**B. Reutilização dos embeddings dos temas**

Os embeddings do catálogo devem ser calculados uma única vez e reutilizados enquanto os dados dos temas permanecerem inalterados. Quando houver atualizações, somente os temas modificados precisarão ter suas representações recalculadas, desde que seja implementado um mecanismo de atualização incremental.

**C. Paralelização do processamento**

A execução concorrente pode ser considerada para ampliar a capacidade de processamento, especialmente na geração dos embeddings e na preparação dos resultados. Entretanto, sua adoção deve ser orientada por medições, uma vez que o pipeline atual já apresenta um tempo de execução reduzido.

**D. Uso de infraestrutura dedicada**

A migração para uma infraestrutura dedicada deve ser avaliada conforme os requisitos de disponibilidade, concorrência, volume de dados e custo operacional. O desempenho atual indica que o processamento de 50 mil questões semanais pode ser viável sem exigir, necessariamente, uma arquitetura distribuída complexa.

**E. Benchmark antes do dimensionamento**

Antes de definir a infraestrutura definitiva, recomenda-se medir separadamente o tempo de geração dos embeddings, a classificação binária, a recuperação dos temas pelo FAISS e a gravação dos resultados. Também é importante executar testes com lotes de diferentes tamanhos para verificar se o desempenho observado em 659 questões se mantém em volumes maiores.

Com base na medição atual, a estimativa inicial de aproximadamente 76 minutos para processar 50 mil questões constitui uma referência de planejamento. Essa projeção deverá ser validada por testes em maior escala antes de ser utilizada como compromisso operacional de produção.

---

## 4. Os 100 Rótulos: Uso, Calibração, Teste e Convenção da Equipe

### 4.1. Estratégia de Utilização (Treino, Calibração e Teste)

A ficha `folha_verdade.csv` contém aproximadamente 100 exemplos de referência rotulados. Esses dados desempenharam um papel central e multifacetado no desenvolvimento da solução:

* **Treinamento do classificador binário:** Os exemplos foram empregados para treinar o modelo responsável por estimar a probabilidade de uma questão estar fora da taxonomia. Cada questão foi convertida em embedding e associada ao rótulo binário correspondente (`0` para dentro da taxonomia, `1` para fora).


* **Calibração de limiares:** Os rótulos permitiram ajustar iterativamente os parâmetros de decisão do sistema, como o limiar de pertinência (`outside_threshold`) e o limiar de similaridade temática (`theme_threshold`), buscando equilibrar a taxa de publicação e a segurança das predições.


* **Avaliação e Teste:** No modo de teste (`--test`), o subconjunto de referência foi dividido estratificadamente para mensurar o desempenho do classificador binário de forma isolada, simulando o comportamento do sistema diante de dados não vistos.



### 4.2. O que os Dados Revelaram sobre o Sistema e a Convenção da Equipe

A análise dos 100 rótulos e do processo de rotulação revelou aspectos cruciais sobre a consistência dos dados e o alinhamento da equipe:

* **Ambiguidade conceitual e subjetividade:** Diferentes avaliadores podem interpretar de forma distinta a fronteira entre temas jurídicos correlatos ou decidir se uma questão está estritamente "fora da taxonomia". Essa variação introduz ruído no conjunto de treinamento supervisionado.
* **Concentração temática:** Os 100 exemplos de referência cobrem apenas uma fração reduzida dos cerca de 377 temas possíveis do catálogo, evidenciando uma esparsez severa que justificou a impossibilidade de treinar um classificador supervisionado multiclasse tradicional.


* **Necessidade de padronização:** O exercício de consolidar os rótulos demonstrou que a qualidade das regras de decisão depende diretamente de diretrizes de rotulação claras e consensuais entre os especialistas humanos, uma vez que o modelo replica os critérios adotados na ficha de referência.

---

## 5. Memo para o Professor

### O que se encontrou na análise dos dados

1. **Incompatibilidade dimensional inicial:** Buscou-se construir classificadores tradicionais para cada um dos 377 temas do catálogo, mas dispunha-se de apenas cerca de 100 exemplos rotulados. Adicionalmente, constatou-se uma severa desproporção na distribuição das classes, visto que apenas 8 desses exemplos representam questões situadas fora da taxonomia. Na prática, essa escassez e o desbalanceamento extremo significam que a imensa maioria dos temas não possui nenhum exemplo histórico de treino, e a classe minoritária carece de volume estatístico suficiente, tornando uma abordagem puramente supervisionada inviável.


2. **Sobreposição e proximidade conceitual:** Constatou-se que o mecanismo de busca semântica (baseado em embeddings e FAISS) recupera com frequência temas que compartilham o mesmo vocabulário jurídico, mas que tratam de institutos ou ramos processuais distintos. O modelo calcula proximidade de texto, mas não compreende a dogmática jurídica subjacente.


3. **Descrições de temas heterogêneas:** Vários tópicos do catálogo possuem apenas o título e o caminho hierárquico, carecendo de descrições textuais ricas que delimitem claramente o escopo de incidência de cada norma ou assunto.



---

### O que se recomenda mudar nos temas

Para elevar a precisão jurídica da ferramenta sem exigir um volume impraticável de novas marcações manuais, recomenda-se:

* **Enriquecimento descritivo do catálogo:** Adicionar breves e precisas descrições conceituais aos 377 temas, explicitando sinônimos, palavras-chave típicas de cobrança em exames e exclusões explícitas (o que o tema *não* abrange).
