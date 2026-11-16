from datasets import load_dataset
import pandas as pd

test = load_dataset("openai/gsm8k", "main", split="test").to_pandas().sample(60, random_state=325)
train = load_dataset("openai/gsm8k", "main", split="train").to_pandas().sample(3, random_state=326)

test["split"], train["split"] = "test", "fewshot"

df = pd.concat([test, train])
df["answer_value"] = df["answer"].str.extract(r"#### (.*)$")[0].str.replace(",", "")

df.to_csv("gsm8k_sample_64.csv", index=False)
df.to_excel("gsm8k_sample_64.xlsx", index=False)
df.to_json(
    "gsm8k_sample_64.json",
    orient="records",
    indent=2,
    force_ascii=False,
)
