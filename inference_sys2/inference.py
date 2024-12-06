
data_path: str = "data/math_dataset.parquet",
dataset = pd.read_parquet(data_path)

generator = Generator(
    model_name_or_path,
    verifier,
    N,
)