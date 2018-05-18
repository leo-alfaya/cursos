import pandas as pd
from sklearn.naive_bayes import MultinomialNB

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

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = list(filter(lambda x: x == True, resultado == teste_marcacoes))
margem_acertos = len(acertos) / len(teste_marcacoes) * 100

print('margem de acertos: {}%'.format(margem_acertos))
