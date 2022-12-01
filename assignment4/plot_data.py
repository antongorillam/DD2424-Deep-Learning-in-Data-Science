import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DIR = 'Result_Pics/'
result_df = pd.read_csv('Result_Pics/results.csv')

plt.figure()
sns.lineplot(data=result_df,
    x="iterations", 
    y="smooth_loss",
    ).set(
    title="Smooth Loss vs Iterations",  
    )
plt.grid()
plt.savefig(f'{DIR}/smooth_loss_vs_iterations')

# Ex of how to extract info needed
# result_df.loc[result_df['smooth_loss']==min(result_df['smooth_loss'])].iat[0,2]
for iter in range(0, 110000, 10000):
    # it = result_df.loc[result_df['iterations']==iter].iat[0,1]
    loss = result_df.loc[result_df['iterations']==iter].iat[0,0]
    text = result_df.loc[result_df['iterations']==iter].iat[0,2]
    print(f"\nAt iterations {iter}, smooth loss {loss:.2f}:")
    print(text)

best_iter = result_df.loc[result_df['smooth_loss']==min(result_df['smooth_loss'])].iat[0,1]
best_loss = result_df.loc[result_df['smooth_loss']==min(result_df['smooth_loss'])].iat[0,0]
best_text = result_df.loc[result_df['smooth_loss']==min(result_df['smooth_loss'])].iat[0,3]

print("\nBest generated text")
print(f"At iterations {best_iter}, smooth loss {best_loss:.2f}:")
print(best_text)