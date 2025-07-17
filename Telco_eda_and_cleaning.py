import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from skimage.feature import shape_index
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


a=os.getcwd()
path=os.path.join(a,"Python_FutureEngineering","Datasets")

telco = f"{path}/Telco-Customer-Churn.csv"


def load(dataset):
    data = pd.read_csv(dataset)
    return data

df = load(telco)
df.head()

#  *************************************
#  TAAK 1 - EDA
#  Numerieke en Categorische variabelen analyse
#  *************************************

#  We groeperen ‘tenure’ in categorieën omdat het een continue variabele is.
#  Door deze in duidelijke intervallen te verdelen, wordt het eenvoudiger en
#  inzichtelijker om het gemiddelde churnpercentage per groep te analyseren.

# Tenure
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
)
df["tenure_group"] = df["tenure_group"].astype("object")


# TotalCharges
# eerst de spaties van TotalCharges vervangen door NaN
# daarna omzetten naar float
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].dtype


def col_names_grab(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(col_names_grab(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = col_names_grab(df)

df.info()



#  *************************************
#  Analyse van Categorische Variabelen
#  *************************************

# We bekijken hier hoe vaak elke categorie voorkomt en welk percentage van het totaal dat is.
for col in cat_cols:
    print(f"---- {col} ----")
    print(df[col].value_counts())  # aantallen per categorie
    print(df[col].value_counts(normalize=True) * 100)  # percentages
    print("\n")


#  *************************************
#  Analyse van Numerieke Variabelen
#  *************************************

df[num_cols].describe().T

# Hier bekijken we de beschrijvende statistieken van de numerieke kolommen:
# gemiddelde, standaarddeviatie, minimum en maximum waarden, kwartielen.


#  *************************************
#  Relatie met Doelvariabele (Target)
#  *************************************

#  Als de doelvariabele bijvoorbeeld "Churn" heet:
#
# Gemiddelde churn per categorie:

df["Churn_numeric"] = df["Churn"].map({"Yes": 1, "No": 0})

for col in cat_cols:
    print(f"{col} vs Churn")
    print(pd.DataFrame({"Churn_Gemiddelde": df.groupby(col)["Churn_numeric"].mean()}))
    print("\n")


#  Gemiddelde numerieke waarden per Churn-categorie:


df.groupby("Churn")[num_cols].mean()

df.info()
#  Hier kijken we of er een verband is tussen
#  de categorische of numerieke variabelen en de kans op churn (uitstroom).


#  *********************
#  1. Categorische variabelen vs Churn
#  *********************
#  Voor elke categorische variabele kun je een gestapelde staafdiagram of percentage barplot maken
#  om het verschil tussen churn = yes/no per categorie te zien.

for col in cat_cols:
    plt.figure(figsize=(8,4))
    churn_ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    churn_ct.plot(kind='bar', stacked=True, color=['green', 'red'])
    plt.title(f"Churn verdeling per categorie: {col}")
    plt.ylabel("Percentage")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()

#  *********************
#  2. Numerieke variabelen vs Churn
#  *********************

# Hier kun je een boxplot of violinplot gebruiken om te zien of
# de verdelingen van bijvoorbeeld tenure, MonthlyCharges en TotalCharges verschillend zijn voor churn=0/1.
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} distributie per Churn")
    plt.show()

# Of een histogram/kde plot:
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=col, hue="Churn", fill=True)
    plt.title(f"{col} verdeling per Churn")
    plt.show()

#  ********************************
#  Voer een analyse van Outliers uit.
#  ********************************

# Het detecteren van uitschieters
#  *********************

df.head()
df.info()

def thresholds_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def outlier_check(dataframe, col_name):
    low_limit, up_limit = thresholds_outlier(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, outlier_check(df, col))


#  ********************************
#  Ontbrekende waarden analyse
#  ********************************

# We definiëren een functie om de ontbrekende waarden in de dataset overzichtelijk weer te geven:
def table_missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

table_missing_values(df)

table_missing_values(df, True)

# Bij de analyse van ontbrekende waarden zien we dat alleen de kolom TotalCharges ontbrekende waarden bevat.
#
# Bij nadere inspectie blijkt dat deze ontbrekende waarden voorkomen bij klanten met een tenure van 0.
# Dit betekent dat deze klanten net begonnen zijn en nog geen totale kosten hebben opgebouwd.
#
# Daarom is besloten om de ontbrekende waarden in TotalCharges op te vullen met 0,
# omdat dit logisch is gegeven hun tenure en het de dataset niet nadelig beïnvloedt.
df[df.isnull().any(axis=1)]

df["TotalCharges"].fillna(0, inplace=True)

#  ********************************
#  Correlatieanalyse
#  ********************************

na_cols = table_missing_values(df, True)

# Churn : 1 ---  No_Churn: 0

def missing_VS_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 'NaN', 'Not_NaN')

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({f"{target}_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_VS_target(df, "Churn_numeric", na_cols)


# We hebben onderzocht of de ontbrekende waarden in de kolom TotalCharges een verband hebben met de targetvariabele Churn_numeric.
# Hiervoor hebben we een nieuwe indicator (TotalCharges_NA_FLAG) aangemaakt die aangeeft of
# de waarde ontbreekt (NaN) of niet (Not_NaN).
#
# De resultaten:
#
# Flag	Gemiddelde Churn	Aantal
# NaN	0.000	11
# Not_NaN	0.266	7032
#
#  Dit betekent dat de rijen met ontbrekende waarden voor TotalCharges geen enkele churn (0%) vertonen.
#  De overige klanten (Not_NaN) hebben een churn-percentage van ongeveer 26,6%.
#
# Conclusie: de ontbrekende waarden lijken niet willekeurig te zijn, maar eerder gekoppeld aan klanten met een tenure van 0,
# wat logisch is omdat deze klanten waarschijnlijk nog niets betaald hebben.













