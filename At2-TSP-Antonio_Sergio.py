'''
a restrição da formulação MTZ apresentado no artigo
   Problema: Problema do Caixeiro Viajante (PCV).
   Formulações: MTZ e Pataki (2003).
   Arquivos: formato xml extraídos do site TSPLIB95.
   Dados: 'bays29.xml', 'brazil58.xml', 'si535.xml', 'si1032.xml'.
   Referências: 
   Miller, Tucker, Zemlin. 1960. Integer Programming Formulation of Traveling Salesman Problems.
   Pataki. 2003. Teaching Integer Programming Formulations using the TSP.
''' 

# Solvers   
#!pip install pulp
import pulp
import numpy as np
import sys
import gurobipy as gp
#from gurobipy import Model, GRB, quicksum


# XML
import xml.etree.ElementTree as ET
import os

def solver_pulp_mtz(matriz_distancias, n_cidades):
        
    '''
    Função:
    Resolve o problema TSP pela formulação MTZ através do pulp.    
    '''
    
    # Etapa 1- Cria a função objetivo de minimização
    modelo_tsp = pulp.LpProblem('Problema do Caixeiro Viajante (PCV)', pulp.LpMinimize)
    
    # Máximo de 1 hora de execução
    solver = pulp.PULP_CBC_CMD(timeLimit=3600)
    
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
    modelo_tsp.solve(solver)
    
    
    if pulp.LpStatus[modelo_tsp.status] == 'Optimal':
        
        solucoes = []
        for i in range(n_cidades):
            for j in range(n_cidades):
                if pulp.value(variaveis_decisao[(i, j)]) == 1:
                    solucoes.append((i, j))
        
        distancia_total = pulp.value(modelo_tsp.objective)
        tempo = modelo_tsp.solutionTime
            
        return distancia_total, solucoes,tempo
    else:
        print("Não foram encontradas soluções ótimas.")
        return None
    
def solver_gurobi_mtz (matriz_distancias, n_cidades):
    
    '''
    Função:
    Resolve o problema TSP pela formulação MTZ através do gurobipy.    
    '''
    
    # Cria novo modelo
    modelo_tsp = gp.Model("Problema do Caixeiro Viajante (PCV)")

    # Máximo de 1 hora de execução
    modelo_tsp.Params.TimeLimit = 3600
    
    
    # Cria variáveis binárias para os nós (vértices)
    variaveis_decisao = modelo_tsp.addVars([(i, j) for i in range(n_cidades) for j in range(n_cidades)], vtype=gp.GRB.BINARY, name="x")

    # Adiciona restrições de entradas e saídas únicas
    for i in range(n_cidades):
        modelo_tsp.addConstr(gp.quicksum(variaveis_decisao[i, j] for j in range(n_cidades) if j != i) == 1)
    
    for j in range(n_cidades):
        modelo_tsp.addConstr(gp.quicksum(variaveis_decisao[i, j] for i in range(n_cidades) if i != j) == 1)

    # Adiciona a restrição de eliminação de subrotas pela formulação MTZ
    u = modelo_tsp.addVars(n_cidades, vtype=gp.GRB.CONTINUOUS, lb=0, ub=n_cidades-1, name="u")
    for i in range(1, n_cidades):
        for j in range(1, n_cidades):
            if i != j:
                modelo_tsp.addConstr(u[i] - u[j] + n_cidades*variaveis_decisao[i,j] <= n_cidades - 1)

    # Set objective
    modelo_tsp.setObjective(gp.quicksum(matriz_distancias[i][j] * variaveis_decisao[i,j] for i in range(n_cidades) for j in range(n_cidades)), gp.GRB.MINIMIZE)

    # Optimize the model
    modelo_tsp.optimize()

    # Extract solution
    if modelo_tsp.Status == gp.GRB.OPTIMAL:
        solucoes = [(i, j) for i, j in variaveis_decisao if variaveis_decisao[i,j].x > 0.5]
        distancia_total = modelo_tsp.ObjVal
        
        tempo = modelo_tsp.Runtime
        return distancia_total, solucoes,tempo
    else:
        print("No optimal solution found.")
        return None, None


def resolve_problema_tsp_mtz (matriz_distancias):
    
    '''
    Função:
    Função intermediária que direciona o problema TSP 
    para ser resolvido pelos solvers Pulp e Gurobi.
    
    '''
    
    # Número de cidades
    n_cidades = len(matriz_distancias)
    
    # Resolve pelo solver Pulp
    distancia_total, solucoes,tempo = solver_pulp_mtz (matriz_distancias, n_cidades)
    print('Tempo de solução do Pulp: ', tempo)
    #print('[MTZ-PULP] Tour:\n')
    #for vertices in solucoes:
    #    print(f'{vertices[0]}ª Cidade: Cidade {vertices[1]}')

    
    espera_usuario()
    
    # Resolve pelo solver gurobi
    distancia_total, solucoes,tempo = solver_gurobi_mtz (matriz_distancias, n_cidades)
    print('Tempo de solução do Gurobi: ', tempo)
    #print('[MTZ-GUROBI] Tour:\n')
    #for vertices in solucoes:
    #    print(f'{vertices[0]}ª Cidade: Cidade {vertices[1]}')

    
    return  distancia_total, solucoes

def solver_pulp_tsp_eliminação_subrotas(matriz_distancias, n_cidades):
    
    '''
    Função:
    Resolve o problema TSP a partir de um problema geral adicionada
    restrições de designação, de subrotas, para então adicionar as
    restrições de arco apresentadas na formulação MTZ.
    Nesta função é utilizado o solver pulp.
    
    '''
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

    
    # Quantidade de iterações restrita a quantidade de restrições
    maxrounds = n_cidades//2
        
    # Loop Iterativo com adição de restrições de subrotas
    k = 1
    tempo = 0
    while k <= maxrounds and tempo <=3600:
        
        # Etapa 5: Resolve o problema (com ou sem restrições de subrotas)
        modelo_subrota.solve()
    
        # Etapa 6: Identifica as subrotas na solução atual
        subrotas = encontrar_subrotas_pulp(variaveis_decisao,n_cidades)
        
        # Se subrotas contém apenas 1, o algoritmo termina. Caso contrário
        # o Loop continua, adicionando restrições para eliminar subrotas menores
        if len(subrotas) == 1:
            print('Solução ótima encontrada!')
            
            break
        
        # Etapa 7: adiciona restrições de subrotas ao modelo do problema
        # Garante que as subrotas sejam eliminadas
        for subrota in subrotas:
            # limita a quantidade de subrotas criadas a no máximo metade do total de cidades
            if (len(subrota) <= n_cidades // 2) and (len(subrota)<1000):
                modelo_subrota += pulp.lpSum(variaveis_decisao[(i,j)] for i in subrota for j in subrota if i!= j) <= len(subrota) - 1
        
        tempo = modelo_subrota.solutionTime
                
        k += 1
    
    # Adicionar restrições de arco (Formulação MTZ) e resolver novamente.
    # Procura garantir que o caminho seja o mais direto possível.
    add_restricoes_arco_pulp(modelo_subrota, variaveis_decisao, n_cidades)
    modelo_subrota.solve()
    
    # Etapa 6: Apresenta os resultados
    if pulp.LpStatus[modelo_subrota.status] == 'Optimal':
        
        solucoes = []
        for i in range(n_cidades):
            for j in range(n_cidades):
                if pulp.value(variaveis_decisao[(i, j)]) == 1:
                    
                    solucoes.append((i+1, j+1))
        
        distancia_total = pulp.value(modelo_subrota.objective)
            
        return distancia_total, solucoes, tempo            
            
    else:
        print("Não foram encontradas soluções ótimas.")
        return None

def solver_gurobi_tsp_eliminação_subrotas (matriz_distancias, n_cidades):
    
    # Cria novo modelo
    modelo_subrota = gp.Model("Problema do Caixeiro Viajante (PCV)")

    # Máximo de 1 hora de execução
    modelo_subrota.Params.TimeLimit = 3600
    
    # Etapa 1: Cria variáveis binárias para os nós (vértices)
    variaveis_decisao = modelo_subrota.addVars([(i, j) for i in range(n_cidades) for j in range(n_cidades)], vtype=gp.GRB.BINARY, name="x")
    
    # Etapa 2: Função Objetivo para minimizar a distância total
    modelo_subrota.setObjective(gp.quicksum(matriz_distancias[i][j] * variaveis_decisao[i, j] for i in range(n_cidades) for j in range(n_cidades)), gp.GRB.MINIMIZE)


    # Etapa 3: Adiciona restrições de entradas e saídas únicas
    modelo_subrota.addConstrs(gp.quicksum(variaveis_decisao[i,j] for j in range(n_cidades) if i != j) == 1 for i in range(n_cidades))
    modelo_subrota.addConstrs(gp.quicksum(variaveis_decisao[i,j] for i in range(n_cidades) if i != j) == 1 for j in range(n_cidades))
    
            
    # Etapa 4: Loop iterativo para adicionar restrições de subrota
    k = 1
    maxrounds = n_cidades//2
    for k in range(maxrounds):
        
        modelo_subrota.optimize()
        
        if modelo_subrota.status == gp.GRB.OPTIMAL:
            
            conjunto_solucoes = [(i, j) for i in range(n_cidades) for j in range(n_cidades) if variaveis_decisao[i, j].x > 0.5]
            
            # Pesquisa quais rotas foram geradas
            subrotas = encontrar_subrotas_gurobi(conjunto_solucoes,n_cidades,variaveis_decisao)
            
            print(f'-----------------ITERAÇÃO {k}-----------------------')
                        
            if len(subrotas) == 1:
                print('Solução ótima encontrada!')
                break
            
            if (modelo_subrota.Runtime>=3600):
                print('Limite de 1 hora atingido')
                break
            
            for subrota in subrotas:
                if ((len(subrota) < n_cidades // 2) and (len(subrota)<1000)):
                    modelo_subrota.addConstr(gp.quicksum(variaveis_decisao[i, j] for i in subrota for j in subrota if i != j) <= len(subrota) - 1)

            # Etapa 5: Realiza o MTZ com as restrições fortes
            #add_restricoes_arco_gurobi(modelo_subrota,variaveis_decisao,n_cidades)
            #modelo_subrota.optimize()
    
            
        else:
            print('Não foi encontrada solução ótima.')
            break
    
    # Etapa 5: Realiza o MTZ com as restrições fortes
    add_restricoes_arco_gurobi(modelo_subrota,variaveis_decisao,n_cidades)
    modelo_subrota.optimize()
    
    # Etapa 6: Extrai a solução final
    if modelo_subrota.Status == gp.GRB.OPTIMAL:
        solucoes = [(i, j) for i, j in variaveis_decisao if variaveis_decisao[i,j].x > 0.5]
        distancia_total = modelo_subrota.objVal
        tempo_execucao = modelo_subrota.Runtime
        print("Solução ótima encontrada!")
        print("Distância total:", distancia_total)
        print("Tempo de execução:", tempo_execucao)
        return distancia_total, solucoes, tempo_execucao
    else:
        print("Não foi encontrada solução ótima.")
        return None            
            
    
def resolve_problema_tsp_eliminacao_subrota (matriz_distancias):
    
    '''
    Função:
    Resolve o problema TSP a partir de um problema geral com subrotas e, a cada iteração,
    é adicionado restrições para eliminar subrotas. 
    Depende de outras duas funções "encontrar_subrotas()"  e "add_restrições_arco()".
    
    '''
    
    # Número de cidades
    n_cidades = len(matriz_distancias)
    
    # Resolve pelo solver Pulp
    distancia_total, solucoes,tempo = solver_pulp_tsp_eliminação_subrotas (matriz_distancias, n_cidades)
    print('Tempo de solução do Pulp: ', tempo)
    #print('[PATAKI-PULP] Tour:\n')
    #for vertices in solucoes:
    #    print(f'{vertices[0]}ª Cidade: Cidade {vertices[1]}')

    #espera_usuario()
    
    # Resolve pelo solver gurobi
    distancia_total, solucoes,tempo = solver_gurobi_tsp_eliminação_subrotas (matriz_distancias, n_cidades)
    print('Tempo de solução do Gurobi: ', tempo)
    #print('[PATAKI-GUROBI] Tour:')
    #for vertices in solucoes:
    #    print(f'{vertices[0]}ª Cidade: Cidade {vertices[1]}')

    

       
def encontrar_subrotas_pulp(variaveis_decisao, n_cidades):
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

def encontrar_subrotas_gurobi(conjunto_solucoes, n_cidades,variaveis_decisao):
    '''
    Função: Encontra subrotas disponíveis à solução corrente
    do problema. Uma subrota é uma sequência de cidades que formam
    um ciclo desconectado do restante do grafo, ou seja, não cobre
    todas as cidades no percurso. Precisa percorer as variáveis de decisão
    e reconstruir os ciclos encontrados na solução corrente. 
    '''
    visited = [False] * n_cidades
    subrotas = []
    for i in range(n_cidades):
        if not visited[i]:
            subrota = []
            current_city = i
            while not visited[current_city]:
                visited[current_city] = True
                subrota.append(current_city)
                # Find the next city in the current subroute
                next_city = [j for j in range(n_cidades) if variaveis_decisao[current_city, j].x > 0.5 and j!=current_city][0]
                current_city = next_city
            subrotas.append(subrota)
        
    return subrotas

    
def add_restricoes_arco_pulp(modelo_subrota,variaveis_decisao, n_cidades):
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

def add_restricoes_arco_gurobi(modelo_subrota,variaveis_decisao, n_cidades):
    '''
    Função: Adiciona restrições da formulação MTZ à situação corrente
    com o objetivo de quebrar a pesquisa nas subrotas. Desta forma, garante
    que o caminho final é ótimo e segue conexões diretas entre as cidades.
    
    '''
    
    # Adiciona a restrição de eliminação de subrotas pela formulação MTZ
    u = modelo_subrota.addVars(n_cidades, vtype=gp.GRB.INTEGER, lb=0, ub=n_cidades-1, name="u")
    for i in range(1, n_cidades):
        for j in range(1, n_cidades):
            if i != j:
                modelo_subrota.addConstr(u[i] - u[j] + n_cidades*variaveis_decisao[i,j] <= n_cidades - 1)

def parse_xml (arquivo_xml):

    '''
    Função: A partir de um diretório de um arquivo xml, o
    particiona e encontra os dados de interesse.
    '''
    # Estrutura do arquivo
    tag_aninhada = 'graph/vertex'

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
    
def solucao_pataki(distancia,maxrounds):
    # Variável para estabelecer regular a formulação TMZ
    # e ser viável encontrar uma solução válida (uma única subrota)

    distancia, solucoes = resolve_problema_tsp_eliminacao_subrota (matriz_distancias, maxrounds)        

def solucao_mtz(distancia):
    # Solução Miller, Tucker, Zemlin (MTZ)

    distancia, solucoes = resolve_problema_tsp_mtz (matriz_distancias)



def inializacao_programa():
    # Início do programa 
    print('Escolha um dos arquivos 1, 2, 3 ou 4 do repositório TSPLIB para solucionar:\n')
    print('[1] - Insira 1 para resolver o bays29.\n')
    print('[2] - Insira 2 para resolver o brazil58.\n')
    print('[3] - Insira 3 para resolver o si535.\n')
    print('[4] - Insira 4 para resolver o si1032 .\n')

    # Escolha do usuário
    try:
        valor = int(input('Insira o número:'))
        
        while (valor not in [1,2,3,4]):
            print('Número inserido incorreto. Tente novamente.')
            
            try:
                valor = int(input('Insira o número:'))
                
            
            except:
                print('Insira um número inteiro de 1 a 4.')
        
    except:
        print('Insira um número inteiro de 1 a 4.')

    return valor

def escolhe_problema():

    # Início do programa 
    print('Escolha 1 ou 2 para solucionar o problema TSP.\n')
    print('[1] - Insira 1 para resolver a partir da formulação MTZ.\n')
    print('[2] - Insira 2 para resolver a partir da formulação PATAKI.\n')

    # Escolha do usuário
    try:
        valor = int(input('Insira o número:'))
        
        while (valor not in [1,2]):
            print('Número inserido incorreto. Tente novamente.')
            
            try:
               valor = int(input('Insira o número:'))
                        
            except:
                print('Insira um número inteiro de 1 a 2.')
        
    except:
        print('Insira o inteiro 1 ou 2.')


    return valor

def finaliza_programa():
    print('Soluções apresentadas. Deseja resolver mais algum problema?')
    print('[1] Insira 1 para resolver mais problemas.')
    print('[2] Insira 2 para sair.')
    
    # Escolha do usuário
    try:
        valor = int(input())
        
        while (valor not in [1,2]):
            print('Número inserido incorreto. Tente novamente.')
            
            try:
               valor = int(input('Insira o número:'))
                        
            except:
                print('Insira um número inteiro de 1 a 2.')
       
    except:
        print('Insira o inteiro 1 ou 2.')
    
    if (valor == 1):
        return True
    else:
        print('Fim do programa.')
        sys.exit()
        return False

def espera_usuario():
    while True:
        valor = input("Press Enter to continue...")
        if valor == "":
            break
    
# Importação dos dados XML
# Diretório onde os arquivos xml estão salvos

path = os.getcwd()
while True:
    # Pede para o usuário inserir um inteiro de 1 a 4.
    valor = inializacao_programa()
    
    if (valor == 1):
        arquivo = 'bays29.xml'
        filepath = os.path.join(path,arquivo)
        # Trata os dados xml e retorna matrix de distâncias
        matriz_distancias = parse_xml(filepath)
    
            
    elif (valor == 2):
        arquivo = 'brazil58.xml'
        filepath = os.path.join(path,arquivo)
        # Trata os dados xml e retorna matrix de distâncias
        matriz_distancias = parse_xml(filepath)
    
    elif (valor == 3):
        arquivo = 'si535.xml'
        filepath = os.path.join(path,arquivo)
        # Trata os dados xml e retorna matrix de distâncias
        matriz_distancias = parse_xml(filepath)
    

    elif (valor == 4):   
        arquivo = 'si1032.xml'  
        filepath = os.path.join(path,arquivo)
        # Trata os dados xml e retorna matrix de distâncias
        matriz_distancias = parse_xml(filepath)
    
    
    
    # Escolha do método de solução pelo usuário
    valor = escolhe_problema()

    if (valor == 1):
        solucao_mtz(matriz_distancias)

    
    elif (valor == 2):
        resolve_problema_tsp_eliminacao_subrota(matriz_distancias)
    
    espera_usuario()
        
    decisao = finaliza_programa()
    # limpa os resultados anteriores
    user_input = str(input())
    os.system('cls')
    
