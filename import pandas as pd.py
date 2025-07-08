import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Data
data = {
    'Pekerjaan': ['Tetap', 'Tetap', 'Kontrak', 'Tidak Bekerja', 'Tidak Bekerja', 'Kontrak', 'Tetap', 'Tidak Bekerja'],
    'Tempat_Tinggal': ['Milik Sendiri', 'Sewa', 'Milik Sendiri', 'Milik Sendiri', 'Tinggal dg Ortu', 'Tinggal dg Ortu', 'Tinggal dg Ortu', 'Sewa'],
    'Kelayakan': ['Layak', 'Layak', 'Layak', 'Tidak', 'Tidak', 'Tidak', 'Layak', 'Tidak']
}

df = pd.DataFrame(data)

# Encode kategori
df_encoded = df.copy()
df_encoded['Pekerjaan'] = df_encoded['Pekerjaan'].map({'Tetap': 0, 'Kontrak': 1, 'Tidak Bekerja': 2})
df_encoded['Tempat_Tinggal'] = df_encoded['Tempat_Tinggal'].map({'Milik Sendiri': 0, 'Sewa': 1, 'Tinggal dg Ortu': 2})
df_encoded['Kelayakan'] = df_encoded['Kelayakan'].map({'Layak': 1, 'Tidak': 0})

# Fitur dan label
X = df_encoded[['Pekerjaan', 'Tempat_Tinggal']]
y = df_encoded['Kelayakan']

# Model ID3
model = DecisionTreeClassifier(criterion='entropy')  # ID3
model.fit(X, y)

# Tampilkan pohon keputusan
rules = export_text(model, feature_names=['Pekerjaan', 'Tempat_Tinggal'])
print(rules)
