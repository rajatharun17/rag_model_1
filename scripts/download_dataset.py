from datasets import load_dataset

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="valid"
)

print(dataset)
print(dataset[0])