import flwr as fl

def weighted_average(metrics):
    acuracias = [num_exemplos * m["acuracia"] for num_exemplos, m in metrics]
    exemplos = [num_exemplos for num_exemplos, _ in metrics]
    return {"acuracia": sum(acuracias) / sum(exemplos)}

fl.server.start_server(
    server_address="[::]:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
        ),
)