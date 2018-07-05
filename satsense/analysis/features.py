import pandas as pd

from satsense.util.path import get_project_root

results_path = '{root}/results/jaccard'.format(root=get_project_root('../../'))
csv_path = "{}/{}_df_optimization.csv".format(results_path, "PANTEX")

optim = pd.read_csv(csv_path)
print(optim)

mean_jaccard = optim.groupby('classifier_name')['jaccard_index'].mean()

