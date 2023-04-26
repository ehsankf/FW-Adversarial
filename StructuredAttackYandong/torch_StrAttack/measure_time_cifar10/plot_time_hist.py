import matplotlib.pyplot as plt
import seaborn as sns
# Generate plot of percentage of wet commutes
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")    # Set style for seaborn output
sns.set_context("notebook", font_scale=2)
attack = ['FGSM', 'PGD', 'FWnucl', 'FWnucl-group', 'StrAttack', 'CS', r'Auto-Attak $\ell_2$', r'Auto-Attak $\ell_\infty$']
com_time = [1.5, 24, 53, 963, 10051, 34947, 2290, 3697]
ax = sns.barplot(x=attack[:3] + attack[-2:], y=com_time[:3] + com_time[-2:], dodge=False, color = "blue", ec="skyblue")
plt.xlabel("Attack")
plt.ylabel("Computational time (s)")
plt.suptitle("Computational time for attacks", y=1.05, fontsize=28)
# plt.title("Based on ", fontsize=18)
plt.xticks(rotation=    60)
widthbars = [0.5, 0.5, 0.5]
for bar,newwidth in zip(ax.patches,widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x+width/2.

    bar.set_x(centre-newwidth/2.)
    bar.set_width(newwidth)
plt.savefig("attack_computational_time_auto_atc.png", bbox_inches='tight')
