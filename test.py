ax = plt.subplot(111)
x, y = np.array([11,10,11]), np.array([11,10,11])
x_range, y_range = x.max()-x.min(), y.max() - y.min()
x_neg_range = x.min()
plt.plot(x,y)

plt.title('fuck eddie', fontsize=11)
plt.xlabel(r'$\mathbf{u_2}$', va='centre', ha='centre')
plt.ylabel(r'$\mathbf{u_1}$', rotation=0, va='centre', ha='centre')

plt.axhline(color='black', lw=0.5)
plt.axvline(color='black', lw=0.5)


#ax.get_xaxis().set_label_coords(0.505, -0.05)
#ax.get_yaxis().set_label_coords(-0.05,0.48)
plt.show()


centre_spines=True
plt.figure(figsize=(8,6), dpi=100)
ax = plt.subplot(111)
# plot phase portrait
plt.plot(x, y, color='blue')

if centre_spines:
    plt.axhline(color='black', lw=0.5)
    plt.axvline(color='black', lw=0.5)

ax.set_xlabel(xlabel, va='centre', ha='centre')
ax.set_ylabel(ylabel, rotation=0, va='bottom', ha='centre')
ax.set_title(title, fontsize=title_size)
plt.show()
