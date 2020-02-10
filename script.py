metrics = {"auc": 0, "precision": 0, "recall": 0, "f1": 0}
for i in range(39):
	data = open(f"model_{i}/results.txt").read().split('\n')[:-1]
	for line in data:
		# print(line)
		k, v = line.split("=")
		k = k.strip()
		v = float(v)
		metrics[k] += v
for k, v in metrics.items():
	print(f"macro-{k}: {v/39}")
