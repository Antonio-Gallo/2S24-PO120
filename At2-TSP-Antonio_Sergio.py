#!pip install pulp
import pulp
import numpy as np

# XML
import xml.etree.ElementTree as ET
import os

# Grafos
import networkx as nx
import matplotlib.pyplot as plt

# Tsplib95
#!pip install tsplib95
import tsplib95


def problema_tsp_mtz (matriz_distancias):
    
    '''
    Função:
    Resolve o problema TSP a partir de um problema geral adicionada
    a restrição da formulação MTZ apresentado no artigo
    Integer Programming Formulation of Traveling Salesman Problems.
    
    '''
    
    # Número de cidades
    n_cidades = len(matriz_distancias)
    
    # Etapa 1- Cria a função objetivo de minimização
    modelo_tsp = pulp.LpProblem('Problema do Caixeiro Viajante (PCV)', pulp.LpMinimize)
    
    # Etapa 2- Cria as variáveis de decisão (Binária, incluir ou não)
    # Matriz de variáveis - Verifica se há aresta entre cidades i e j
    variaveis_decisao = pulp.LpVariable.dicts('x', [(i, j) for i in range(n_cidades) for j in range(n_cidades)], cat='Binary')

    # Vetor de valores inteiros positivos de acordo com a Formulação MTZ
    vetor_real = pulp.LpVariable.dicts('u', range(n_cidades), lowBound=0, cat='Integer')

    
    # Etapa 3: Define a função objetivo. Minimizar a distância total percorrida
    modelo_tsp += pulp.lpSum(matriz_distancias[i][j] * variaveis_decisao[(i, j)] for i in range(n_cidades) for j in range(n_cidades))

    # Etapa 4: Define as restrições (Cada cidade deve ser visitada ao menos uma vez)
    for i in range(n_cidades):
        # Deve-se entrar 1 vez de cada cidade
        modelo_tsp += pulp.lpSum(variaveis_decisao[(i, j)] for j in range(n_cidades) if j != i) == 1
        # Deve-se sair 1 vez de cada cidade
        modelo_tsp += pulp.lpSum(variaveis_decisao[(j, i)] for j in range(n_cidades) if j != i) == 1

    # Etapa 4.1. Elimina subrotas (Formulação MTZ)
    for i in range(1, n_cidades):
        for j in range(1, n_cidades):
            if i != j:
                modelo_tsp += vetor_real[i] - vetor_real[j] + n_cidades * variaveis_decisao[(i, j)] <= n_cidades - 1
                
    # Etapa 5: Resolve o problema
    modelo_tsp.solve()

    # Etapa 6: Apresenta os resultados
    
    if pulp.LpStatus[modelo_tsp.status] == 'Optimal':
        
        solucoes = []
        for i in range(n_cidades):
            for j in range(n_cidades):
                if pulp.value(variaveis_decisao[(i, j)]) == 1:
                    solucoes.append((i, j))
        
        distancia_total = pulp.value(modelo_tsp.objective)
        tempo = modelo_tsp.solutionTime
            
        return distancia_total, solucoes, tempo
    else:
        print("Não foram encontradas soluções ótimas.")
        return None

def problema_tsp_eliminacao_subrota (matriz_distancias, maxrounds):
    
    '''
    Função:
    Resolve o problema TSP a partir de um problema geral com subrotas e, a cada iteração,
    é adicionado restrições para eliminar subrotas. 
    Depende de outras duas funções "encontrar_subrotas()"  e "add_restrições_arco()".
    
    '''
    
    # Número de cidades
    n_cidades = len(matriz_distancias)
    
    # Etapa 1- Cria a função objetivo de minimização
    modelo_subrota = pulp.LpProblem('Problema do Caixeiro Viajante (PCV) por eliminação de subrotas', pulp.LpMinimize)
    
    # Etapa 2- Cria as variáveis de decisão (Binária, incluir ou não)
    # Verifica se há aresta entre cidades i e j.
    variaveis_decisao = pulp.LpVariable.dicts('x', [(i, j) for i in range(n_cidades) for j in range(n_cidades)], cat='Binary')

    # Etapa 3: Define a função objetivo. Minimizar a distância total percorrida
    modelo_subrota += pulp.lpSum(matriz_distancias[i][j] * variaveis_decisao[(i, j)] for i in range(n_cidades) for j in range(n_cidades))

    # Etapa 4: Define as restrições (Cada cidade deve ser visitada ao menos uma vez)
    # Restrições de designação:
    for i in range(n_cidades):
        # Deve-se entrar 1 vez de cada cidade
        modelo_subrota += pulp.lpSum(variaveis_decisao[(i, j)] for j in range(n_cidades) if j != i) == 1
        # Deve-se sair 1 vez de cada cidade
        modelo_subrota += pulp.lpSum(variaveis_decisao[(j, i)] for j in range(n_cidades) if j != i) == 1

    # Loop Iterativo com adição de restrições de subrotas
    k = 1
    while k <= maxrounds:
        
        # Etapa 5: Resolve o problema (com ou sem restrições de subrotas)
        modelo_subrota.solve()
    
        # Etapa 6: Identifica as subrotas na solução atual
        subrotas = encontrar_subrotas(variaveis_decisao,n_cidades)
        
        # Se subrotas contém apenas 1, o algoritmo termina. Caso contrário
        # o Loop continua, adicionando restrições para eliminar subrotas menores
        if len(subrotas) == 1:
            print('Solução ótima encontrada!')
            
            break
        
        # Etapa 7: adiciona restrições de subrotas ao modelo do problema
        # Garante que as subrotas sejam eliminadas
        for subrota in subrotas:
            # limita a quantidade de subrotas criadas a no máximo metade do total de cidades
            if len(subrota) <= n_cidades // 2:
                modelo_subrota += pulp.lpSum(variaveis_decisao[(i,j)] for i in subrota for j in subrota if i!= j) <= len(subrota) - 1
                
        k += 1
    
    # Adicionar restrições de arco (Formulação MTZ) e resolver novamente.
    # Procura garantir que o caminho seja o mais direto possível.
    add_restricoes_arco(modelo_subrota, variaveis_decisao, n_cidades)
    modelo_subrota.solve()
    
    # Etapa 6: Apresenta os resultados
    if pulp.LpStatus[modelo_subrota.status] == 'Optimal':
        
        solucoes = []
        for i in range(n_cidades):
            for j in range(n_cidades):
                if pulp.value(variaveis_decisao[(i, j)]) == 1:
                    
                    solucoes.append((i+1, j+1))
        
        distancia_total = pulp.value(modelo_subrota.objective)
        tempo = modelo_subrota.solutionTime
            
        return distancia_total, solucoes, tempo            
            
    else:
        print("Não foram encontradas soluções ótimas.")
        return None
    
        
def encontrar_subrotas(variaveis_decisao, n_cidades):
    '''
    Função: Encontra subrotas disponíveis à solução corrente
    do problema. Uma subrota é uma sequência de cidades que formam
    um ciclo desconectado do restante do grafo, ou seja, não cobre
    todas as cidades no percurso. Precisa percorer as variáveis de decisão
    e reconstruir os ciclos encontrados na solução corrente. 
    '''
    # Lista para armazenar as subrotas a serem encontradas
    subrotas = []
    
    # Conjunto de cidades ainda não visitadas
    cidades_n_visitadas = list(range(n_cidades))
    
    while cidades_n_visitadas:
        # Iniciar uma nova subrota
        subrota_atual = []
        cidade_inicial = cidades_n_visitadas[0]  # Escolher uma cidade para começar
        cidade_atual = cidade_inicial
        
        while True:
            subrota_atual.append(cidade_atual)
            cidades_n_visitadas.remove(cidade_atual)
            
            # Encontrar a próxima cidade conectada à cidade atual
            proxima_cidade = None
            for j in range(n_cidades):
                if cidade_atual != j and pulp.value(variaveis_decisao[(cidade_atual, j)]) == 1:
                    proxima_cidade = j
                    break
            
            if proxima_cidade is None or proxima_cidade == cidade_inicial:
                # Fechamos um ciclo (subrota completa)
                break
            else:
                cidade_atual = proxima_cidade
        
        # Adicionar a subrota encontrada à lista de subrotas
        subrotas.append(subrota_atual)
    
    return subrotas

    
def add_restricoes_arco(modelo_subrota,variaveis_decisao, n_cidades):
    '''
    Função: Adiciona restrições da formulação MTZ à situação corrente
    com o objetivo de quebrar a pesquisa nas subrotas. Desta forma, garante
    que o caminho final é ótimo e segue conexões diretas entre as cidades.
    
    '''
    # Criar variáveis u para a formulação MTZ
    u = pulp.LpVariable.dicts('u', range(n_cidades), lowBound=0, cat='Integer')

    # Adicionar as restrições de Miller-Tucker-Zemlin para eliminar subrotas
    for i in range(1, n_cidades):
        for j in range(1, n_cidades):
            if i != j:
                # Restrição de eliminação de subrotas
                modelo_subrota += u[i] - u[j] + n_cidades * variaveis_decisao[(i, j)] <= n_cidades - 1

def parse_xml (arquivo_xml, tag_aninhada):

    '''
    Função: A partir de um diretório de um arquivo xml, o
    particiona e encontra os dados de interesse.
    '''
    # Particiona o xml e retorna objeto ElementTree
    element_tree = ET.parse(filepath)
    raiz = element_tree.getroot()
    
    # Encontra todas as tags <edge>
    #nested_elements = raiz.findall(tag_aninhada)
    
    # Coleta dados para saber quantas cidades existem
    dados = {}
    for child in raiz.iter('vertex'):
        for grand in child.iter('edge'):
            dados[grand.text] = grand.attrib['cost']
        break
    
    # Cria uma matriz nxn de zeros    
    matrix = np.zeros((len(dados)+1, len(dados)+1))            
   
    # Preenche a matriz
    i=0
    j=0
    for child in raiz.iter('vertex'):
        # Para o caso Inicial (i = j = 0)
        if (i == j):
                matrix [i,j] = 0
                j+=1
        
        for grand in child.iter('edge'):
        # Para outros casos em i>0 e j >0
            if (i == j):
                matrix [i,j] = 0
                j+=1
            
            matrix [i,j] = grand.attrib['cost']
            j+=1
        j=0    
        i+=1 
    
    return matrix          
    
def leitura_tour(arquivo):
    # Comparando com o obtido
    tour = []
    with open(arquivo, 'r') as f:
        for line in f:
            tour.append(int(line.strip()))
    return tour
    
     
# Importação dos dados XML
path = os.getcwd()
arquivo = 'si535.xml'
tag_aninhada = 'graph/vertex'
filepath = os.path.join(path,arquivo)

# Trata os dados xml e retorna matrix de distâncias
matriz_distancias = parse_xml(filepath, tag_aninhada)

# Solução Miller, Tucker, Zemlin (MTZ)
'''
distancia, solucoes,tempo = problema_tsp_mtz (matriz_distancias)
print(solucoes)

# Apresenta as soluções
print('[MTZ] Distância total =', distancia)
print('[MTZ] Tempo de processamento =', tempo)
print('[MTZ] Tour:')
'''
# Variável para estabelecer regular a formulação TMZ
# e ser viável encontrar uma solução válida (uma única subrota)

maxrounds = 1000
distancia, solucoes,tempo = problema_tsp_eliminacao_subrota (matriz_distancias, maxrounds)        

# Apresenta as soluções
print('[PATAKI] Distância total =', distancia)
print('[PATAKI] Tempo de processamento =', tempo)
print('[PATAKI] Tour:')


for vertices in solucoes:
    print(f'Cidade {vertices[0]} -> Cidade {vertices[1]}')

# Cria um grafo direcionado vazio
#G = nx.DiGraph()

# Adiciona nós a partir da lista solucoes
#nodes = [i for i, _ in enumerate(solucoes)]
#G.add_nodes_from(nodes)
#for i in range(len(nodes) - 1):
#    G.add_edge(nodes[i], nodes[i + 1])

# Cria dicionário de rótulos com base na solução
#labels = {i: f"{solucoes[i][1]}" for i in range(len(solucoes))}


# Desenha e apresenta o grafo
#nx.draw_networkx(G, with_labels=True, labels=labels)
#plt.show()

# Comparando com o obtido
#arquivo = 'bays29.opt.tour'
#filepath = os.path.join(path,arquivo)

#tour = tsplib95.load_solution(filepath)
#print(tour)