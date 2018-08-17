from collections import Counter
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("busca.csv")
X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

X = pd.get_dummies(X_df).values
Y = Y_df.values

tamanho_treino = int(.7 * len(Y))
tamanho_teste = int(len(Y) - tamanho_treino)

treino_dados = X[:tamanho_treino]
treino_marcacoes = Y[:tamanho_treino]

teste_dados = X[-tamanho_teste:]
teste_marcacoes = Y[-tamanho_teste:]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes,
                    teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)
    acertos = list(filter(lambda x: x == True, resultado == teste_marcacoes))
    margem_acertos = len(acertos) / len(teste_marcacoes) * 100

    print('margem de acertos do {0}: {1}%'.format(nome, margem_acertos))

modelo = MultinomialNB()
fit_and_predict("MultinomialNB", modelo, treino_dados, treino_marcacoes,
                    teste_dados, teste_marcacoes)

modelo = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", modelo, treino_dados, treino_marcacoes,
                    teste_dados, teste_marcacoes)

#eficácia do algoritmo que chuta tudo 1 ou 0
taxa_acerto_base = max(Counter(teste_marcacoes).values()) * 100.0 / len(teste_marcacoes)

print("Taxa de acerto base: {}%".format(taxa_acerto_base))
print("Total de testes: {}".format(len(teste_dados)))
