from dados import carregar_acessos
from sklearn.naive_bayes import MultinomialNB

X, Y = carregar_acessos()

treino_dados = X[:70]
treino_marcacoes = Y[:70]

teste_dados = X[-29:]
teste_marcacoes = Y[-29:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
acertos = list(filter(lambda x: x == True, resultado == teste_marcacoes))
margem_acertos = len(acertos) / len(teste_marcacoes) * 100

print('margem de acertos: {}%'.format(margem_acertos))
