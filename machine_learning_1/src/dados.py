import csv

def carregar_acessos():
    X = []
    Y = []

    arquivo = open('acesso_pagina.csv', 'r')
    leitor = csv.reader(arquivo, delimiter=',')
    
    leitor.__next__()

    for (home,
         como_funciona,
         contato,
         comprou) in leitor:
        
        X.append([int(home), int(como_funciona), int(contato)])
        Y.append(int(comprou))
    
    return X, Y
