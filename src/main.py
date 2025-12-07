from modules.loaders.jogos import fetch_data
from modules.lr.slr import LR
from modules.metrics.reg.rmse import rmse


def main():
    source = "data/best_selling_switch_games.csv"

    # Carrega dados
    X, y = fetch_data(source)

    # Cria modelo
    model = LR()

    # Treina
    model.train(X, y)

    # Resultados
    print("Score:", model.get_score(X, y))
    print("Intercept:", model.get_intercept())
    print("Coefficients:", model.get_coefficients())

    # Predição
    y_pred = model.predict(X)
    print("Predicted values:", y_pred)

    # RMSE
    print("RMSE:", rmse(y, y_pred))


if __name__ == "__main__":
    main()