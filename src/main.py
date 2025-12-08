from modules.loaders.jogos import fetch_data
from modules.lr.slr import LR
from modules.metrics.reg.rmse import rmse

# Cores simples e seguras (não quebram o terminal)
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
END = "\033[0m"


def section(title: str):
    print(f"\n{BOLD}{BLUE}=== {title} ==={END}")


def main():
    source = "data/best_selling_switch_games.csv"

    section("CARREGANDO DADOS")
    X, y = fetch_data(source)
    print(f"{GREEN}Dados carregados!{END}")
    print(f"Total de registros: {len(X)}")

    section("CRIANDO MODELO")
    model = LR()
    print("Modelo criado com sucesso.")

    section("TREINANDO MODELO")
    model.train(X, y)
    print(f"{GREEN}Treinamento concluído!{END}")

    section("RESULTADOS")
    print(f"{CYAN}Score:{END}         {model.get_score(X, y)}")
    print(f"{CYAN}Intercept:{END}    {model.get_intercept()}")
    print(f"{CYAN}Coefficients:{END} {model.get_coefficients()}")

    section("PREDIÇÃO")
    y_pred = model.predict(X)
    print(f"{CYAN}Primeiras predições:{END} {y_pred[:5]}")

    section("RMSE")
    print(f"{YELLOW}RMSE:{END} {rmse(y, y_pred)}")


if __name__ == "__main__":
    main()
