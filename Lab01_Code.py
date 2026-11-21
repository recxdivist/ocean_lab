# Lab01 — исходный код трёх лабораторных работ
# Датасет: CalCOFI
#
# ВАЖНО:
# Код собран из трёх исходных Colab-ноутбуков и сгруппирован по темам.
# Для воспроизведения результатов запускайте ячейки в исходном порядке
# внутри соответствующего раздела. Некоторые разделы используют общие
# переменные, загружают данные из KaggleHub и требуют исходных библиотек.
#
# Ноутбуки с результатами:
# - Lab01_Salinity_KTvD.ipynb
# - Lab01_oxygen_KTvD.ipynb
# - Lab01_temperatur_KTvD.ipynb


# ==============================================================================
# Lab01_Salinity_KTvD
# ==============================================================================

# --- 1. Библиотеки и единый стиль графиков ---
#@title Библиотеки и единый стиль графиков
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

plt.rcParams.update({
    'figure.dpi': 110, 'figure.figsize': (10, 4.5), 'figure.constrained_layout.use': True,
    'axes.grid': True, 'grid.alpha': .3, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
})
BLUE, RED, ORANGE, GRAY = '#2b6cb0', '#d9534f', '#ed8936', '#718096'
XLABEL = 'Солёность, PSU'
rng = np.random.default_rng(42)

# --- 2. Загрузка данных ---
#@title Загрузка данных
import kagglehub
path = kagglehub.dataset_download("sohier/calcofi")
df = pd.read_csv(path + "/bottle.csv", low_memory=False)
salinity = df['Salnty'].dropna().reset_index(drop=True)
print(f"Строк в таблице: {len(df):,}; измерений солёности: {len(salinity):,}; пропусков: {df['Salnty'].isna().sum():,}")

# --- 3. Разброс солёности при фиксированных условиях ---
#@title Разброс солёности при фиксированных условиях
d = df.loc[df.Depthm <= 1000, ['Depthm', 'Salnty']].dropna()
st = df.loc[df.Depthm == 0, 'Sta_ID'].mode()[0]
s0 = df[(df.Sta_ID == st) & (df.Depthm == 0)].dropna(subset=['Salnty'])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
hb = ax[0].hexbin(d.Salnty, d.Depthm, gridsize=60, bins='log', cmap='Blues', mincnt=1)
ax[0].invert_yaxis()
ax[0].set(xlabel=XLABEL, ylabel='Глубина, м', title='Солёность на разных глубинах (до 1000 м)')
fig.colorbar(hb, ax=ax[0], label='Число измерений (log)')
ax[1].plot(s0.Cst_Cnt, s0.Salnty, '.-', color=BLUE, lw=.6)
ax[1].set(xlabel='Порядковый номер станции-рейса (Cst_Cnt)', ylabel=XLABEL,
          title=f'Станция {st}, поверхность (0 м)')
plt.show()

# --- 4. Доля дисперсии, объяснённая глубиной ---
#@title Доля дисперсии, объяснённая глубиной
dd = df[['Depthm', 'Salnty']].dropna()
resid = dd.Salnty - dd.groupby('Depthm').Salnty.transform('mean')
r2 = 1 - (resid ** 2).sum() / ((dd.Salnty - dd.Salnty.mean()) ** 2).sum()
print(f'R² модели «солёность = среднее на данной глубине»: {r2:.3f}; не объяснено глубиной: {1 - r2:.1%}')

# --- 5. Описательные статистики ---
#@title Описательные статистики
mu, sigma = salinity.mean(), salinity.std()
var = sigma ** 2
desc = pd.Series({
    'Среднее, PSU': mu, 'Медиана, PSU': salinity.median(),
    'Дисперсия, PSU²': var, 'Стд. отклонение, PSU': sigma,
    'Асимметрия': salinity.skew(), 'Эксцесс (избыточный)': salinity.kurt(),
    'Минимум, PSU': salinity.min(), 'Максимум, PSU': salinity.max(),
}, name='Salnty')
display(desc.to_frame().style.format('{:.4f}'))

# --- 6. Гистограмма и боксплот ---
#@title Гистограмма и боксплот
med = salinity.median()
fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})
sns.histplot(salinity, bins=120, color=BLUE, ax=a1)
a1.axvline(mu, color=RED, ls='--', label=f'среднее = {mu:.2f}')
a1.axvline(med, color=ORANGE, ls=':', lw=2, label=f'медиана = {med:.2f}')
a1.set(ylabel='Число измерений', title='Распределение солёности')
a1.legend()
sns.boxplot(x=salinity, color='#bee3f8', fliersize=2, ax=a2)
a2.set(xlabel=XLABEL, yticks=[])
a2.grid(False)
plt.show()

# --- 7. Сравнение с нормальным законом ---
#@title Сравнение с нормальным законом
x = np.linspace(salinity.min(), salinity.max(), 500)
fig, ax = plt.subplots()
ax.hist(salinity, bins=120, density=True, color=BLUE, alpha=.6, label='данные')
ax.plot(x, stats.norm.pdf(x, mu, sigma), color=RED, lw=2, label=f'$\\mathcal{{N}}({mu:.2f};\\ {sigma:.2f}^2)$')
ax.set(xlabel=XLABEL, ylabel='Плотность, 1/PSU', title='Данные и нормальная модель с теми же μ и σ')
ax.legend()
plt.show()

# --- 8. Квантили и выбросы по правилу IQR ---
#@title Квантили и выбросы по правилу IQR
qs = [.01, .05, .25, .5, .75, .95, .99]
qt = salinity.quantile(qs)
qt.index = [f'Q{q * 100:g}%' for q in qs]
display(qt.to_frame('PSU').T.style.format('{:.4f}'))

q1, q3 = qt['Q25%'], qt['Q75%']
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
below, above = (salinity < lo).sum(), (salinity > hi).sum()
print(f'Границы: [{lo:.4f}; {hi:.4f}] PSU')
print(f'Выбросов: {below + above:,} ({(below + above) / len(salinity):.2%}): ниже границы {below:,}, выше {above:,}')

# --- 9. Стандартизация (z-преобразование) ---
#@title Стандартизация (z-преобразование)
z = (salinity - mu) / sigma
t = np.linspace(-6, 6, 400)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(salinity, bins=120, color=BLUE)
ax[0].set(xlabel=XLABEL, ylabel='Число измерений', title='Исходная шкала')
ax[1].hist(z, bins=120, range=(-6, 6), density=True, color=ORANGE, alpha=.7, label='$z$')
ax[1].plot(t, stats.norm.pdf(t), color=RED, label=r'$\mathcal{N}(0;\ 1)$')
ax[1].set(xlabel='$z$, в единицах σ', ylabel='Плотность', title='После стандартизации (показано от −6 до 6)')
ax[1].legend()
plt.show()
print(f'z: среднее = {z.mean():.1e}, стд. отклонение = {z.std():.4f}')

# --- 10. Разделение на две компоненты по порогу ---
#@title Разделение на две компоненты по порогу
THRESHOLD = 33.84
sal_low, sal_high = salinity[salinity < THRESHOLD], salinity[salinity >= THRESHOLD]
p1 = len(sal_low) / len(salinity)
mu1, s1, mu2, s2 = sal_low.mean(), sal_low.std(), sal_high.mean(), sal_high.std()

params = pd.DataFrame(
    {'n': [len(sal_low), len(sal_high)], 'доля p': [p1, 1 - p1], 'μ, PSU': [mu1, mu2], 'σ, PSU': [s1, s2]},
    index=[f'Компонента 1 (< {THRESHOLD})', f'Компонента 2 (≥ {THRESHOLD})'])
display(params.style.format({'n': '{:,.0f}', 'доля p': '{:.4f}', 'μ, PSU': '{:.4f}', 'σ, PSU': '{:.4f}'}))

bins = np.linspace(32, 36, 120)
x = np.linspace(32, 36, 500)
fig, ax = plt.subplots()
ax.hist([sal_low, sal_high], bins=bins, stacked=True, density=True, color=[ORANGE, BLUE], alpha=.6,
        label=[f'< {THRESHOLD}', f'≥ {THRESHOLD}'])
ax.plot(x, p1 * stats.norm.pdf(x, mu1, s1), color=ORANGE, ls='--')
ax.plot(x, (1 - p1) * stats.norm.pdf(x, mu2, s2), color=BLUE, ls='--')
ax.plot(x, p1 * stats.norm.pdf(x, mu1, s1) + (1 - p1) * stats.norm.pdf(x, mu2, s2), color='k', lw=2, label='модель-смесь')
ax.set(xlim=(32, 36), xlabel=XLABEL, ylabel='Плотность, 1/PSU', title='Компоненты смеси и модельная плотность (32–36 PSU)')
ax.legend()
plt.show()

# --- 11. Солёность по слоям глубины ---
#@title Солёность по слоям глубины
d = df[['Depthm', 'Salnty']].dropna().copy()
d['Слой'] = pd.cut(d.Depthm, [-1, 50, 200, np.inf], labels=['0–50 м', '50–200 м', '> 200 м'])
fig, ax = plt.subplots()
sns.histplot(d, x='Salnty', hue='Слой', multiple='stack', bins=120, palette=[ORANGE, GRAY, BLUE], ax=ax)
ax.axvline(THRESHOLD, color='k', ls='--', lw=1)
ax.set(xlim=(32, 36), xlabel=XLABEL, ylabel='Число измерений', title='Солёность по слоям глубины (32–36 PSU)')
plt.show()

# --- 12. Состоятельность: оценки по случайным подвыборкам ---
#@title Состоятельность: оценки по случайным подвыборкам
x = salinity.to_numpy()
ns = np.unique(np.logspace(2, 5, 10).astype(int))
K = 200

def estimates(n):
    out = np.empty((K, 2))
    for k in range(K):
        s = rng.choice(x, n)
        out[k] = s.mean(), s.var(ddof=1)
    return out

res = np.array([estimates(n) for n in ns])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for j, (true, ylab) in enumerate([(mu, 'Выборочное среднее, PSU'), (var, 'Выборочная дисперсия, PSU²')]):
    lo, md_, hi = np.percentile(res[:, :, j], [5, 50, 95], axis=1)
    ax[j].fill_between(ns, lo, hi, color=BLUE, alpha=.25, label='5–95% разброса оценки')
    ax[j].plot(ns, md_, color=BLUE, label='медиана оценки')
    ax[j].axhline(true, color=RED, ls='--', label='значение по всей таблице')
    ax[j].set(xscale='log', xlabel='Размер подвыборки $n$', ylabel=ylab)
ax[0].set_title('Среднее')
ax[1].set_title('Дисперсия')
ax[0].legend()
ax[2].loglog(ns, res[:, :, 0].std(axis=1), 'o-', color=BLUE, label='эмпирическая')
ax[2].loglog(ns, sigma / np.sqrt(ns), '--', color=RED, label=r'$\sigma/\sqrt{n}$')
ax[2].set(xlabel='Размер подвыборки $n$', ylabel='Стд. ошибка среднего, PSU', title='Скорость сходимости среднего')
ax[2].legend()
plt.show()

# --- 13. Статистики выборок и генеральной совокупности ---
#@title Статистики выборок и генеральной совокупности
n_s = 10_000
s_wo = rng.choice(x, n_s, replace=False)
s_w = rng.choice(x, n_s, replace=True)
cmp = pd.DataFrame({
    'Среднее, PSU': [mu, s_wo.mean(), s_w.mean()],
    'Дисперсия, PSU²': [var, s_wo.var(), s_w.var()],
}, index=[f'Генеральная (N={len(x):,})', f'Без возвращения (n={n_s:,})', f'С возвращением (n={n_s:,})'])
display(cmp.style.format('{:.4f}'))

fig, ax = plt.subplots()
bins = np.linspace(32, 36, 100)
ax.hist(x, bins, density=True, color=GRAY, alpha=.5, label='генеральная')
ax.hist(s_wo, bins, density=True, histtype='step', color=BLUE, lw=1.6, label='без возвращения')
ax.hist(s_w, bins, density=True, histtype='step', color=RED, lw=1.6, label='с возвращением')
ax.set(xlabel=XLABEL, ylabel='Плотность, 1/PSU', title='Выборки и генеральная совокупность (32–36 PSU)')
ax.legend()
plt.show()

# --- 14. Разброс среднего: с возвращением и без ---
#@title Разброс среднего: с возвращением и без
N, reps = len(x), 200
sizes = [1_000, 10_000, 100_000, 400_000]
rows = []
for n in sizes:
    m_w = [rng.choice(x, n).mean() for _ in range(reps)]
    m_wo = [rng.choice(x, n, replace=False).mean() for _ in range(reps)]
    rows.append((n, np.std(m_w), np.std(m_wo)))
se = pd.DataFrame(rows, columns=['n', 'с возвр.', 'без возвр.']).set_index('n')
se['отношение'] = se['без возвр.'] / se['с возвр.']
se['теория √((N−n)/(N−1))'] = np.sqrt((N - se.index) / (N - 1))
display(se.style.format({'с возвр.': '{:.5f}', 'без возвр.': '{:.5f}', 'отношение': '{:.3f}', 'теория √((N−n)/(N−1))': '{:.3f}'})
        .set_caption('Стд. отклонение выборочного среднего, PSU'))

fig, ax = plt.subplots()
ax.loglog(se.index, se['с возвр.'], 'o-', color=RED, label='с возвращением')
ax.loglog(se.index, se['без возвр.'], 's-', color=BLUE, label='без возвращения')
ax.loglog(se.index, sigma / np.sqrt(se.index), ':', color=RED, label=r'$\sigma/\sqrt{n}$')
ax.loglog(se.index, sigma / np.sqrt(se.index) * np.sqrt((N - se.index) / (N - 1)), ':', color=BLUE, label='с поправкой на конечность')
ax.set(xlabel='Размер выборки $n$', ylabel='Стд. отклонение среднего, PSU', title='Схемы выбора: с возвращением и без')
ax.legend()
plt.show()

# --- 15. Критерий Андерсона–Дарлинга и Q–Q графики ---
#@title Критерий Андерсона–Дарлинга и Q–Q графики
import warnings
groups = {'Вся выборка': salinity, f'Компонента 1 (< {THRESHOLD})': sal_low, f'Компонента 2 (≥ {THRESHOLD})': sal_high}
rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (name, g) in zip(ax, groups.items()):
    sub = g.sample(10_000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ad = stats.anderson(sub, dist='norm')
    crit = ad.critical_values[2]
    rows.append({'Выборка': name, '$A^2$': ad.statistic, 'крит. значение (5%)': crit,
                 'H0': 'отвергается' if ad.statistic > crit else 'не отвергается'})
    stats.probplot(sub, dist='norm', plot=a)
    a.get_lines()[0].set(color=BLUE, markersize=3)
    a.get_lines()[1].set(color=RED)
    a.set(title=name, xlabel='Теоретические квантили $\\mathcal{N}(0;1)$', ylabel=XLABEL)
plt.show()
display(pd.DataFrame(rows).set_index('Выборка').style.format({'$A^2$': '{:.2f}', 'крит. значение (5%)': '{:.3f}'}))

# --- 16. Класс генератора ---
#@title Класс генератора
class SalinityMixtureGenerator:
    def __init__(self, mu1, s1, mu2, s2, p1, seed=None):
        self.mu1, self.s1, self.mu2, self.s2, self.p1 = mu1, s1, mu2, s2, p1
        self.rng = np.random.default_rng(seed)

    def std_normal(self, n):
        u1, u2 = self.rng.random(n), self.rng.random(n)
        return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    def generate(self, n):
        first = self.rng.random(n) < self.p1
        z = self.std_normal(n)
        return np.where(first, self.mu1 + self.s1 * z, self.mu2 + self.s2 * z)

gen = SalinityMixtureGenerator(mu1, s1, mu2, s2, p1, seed=42)
generated = gen.generate(len(salinity))
print(f'Сгенерировано {len(generated):,} значений: среднее = {generated.mean():.4f}, σ = {generated.std():.4f}')

# --- 17. Сравнение метрик ---
#@title Сравнение метрик
g = pd.Series(generated)
rows = {'Среднее': (salinity.mean(), g.mean()), 'Стд. отклонение': (salinity.std(), g.std()),
        'Асимметрия': (salinity.skew(), g.skew()), 'Эксцесс': (salinity.kurt(), g.kurt())}
for q in [.05, .25, .5, .75, .95]:
    rows[f'Q{q * 100:g}%'] = (salinity.quantile(q), g.quantile(q))
tbl = pd.DataFrame(rows, index=['Реальные', 'Генератор']).T
tbl['Абс. разность'] = (tbl['Генератор'] - tbl['Реальные']).abs()
display(tbl.style.format('{:.4f}'))

# --- 18. Реальные данные и генератор ---
#@title Реальные данные и генератор
bins = np.linspace(32, 36, 120)
q = np.linspace(.001, .999, 300)
qr, qg = np.quantile(salinity, q), np.quantile(generated, q)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(salinity, bins, density=True, color=BLUE, alpha=.6, label='реальные')
ax[0].hist(generated, bins, density=True, histtype='step', color=RED, lw=1.8, label='генератор')
ax[0].set(xlabel=XLABEL, ylabel='Плотность, 1/PSU', title='Плотности (32–36 PSU)')
ax[0].legend()
ax[1].plot(qr, qg, '.', color=BLUE, label='квантили 0,1–99,9%')
ax[1].plot([qr.min(), qr.max()], [qr.min(), qr.max()], '--', color=RED, label='$y=x$')
ax[1].set(xlabel='Квантили реальных данных, PSU', ylabel='Квантили генератора, PSU', title='Q–Q: данные и генератор')
ax[1].legend()
plt.show()

# --- 19. Критерий Колмогорова–Смирнова ---
#@title Критерий Колмогорова–Смирнова
rows = []
for n in [1_000, 5_000, 10_000, 50_000]:
    r = stats.ks_2samp(salinity.sample(n, random_state=1), generated[:n])
    rows.append({'n': n, 'D': r.statistic, 'p-value': r.pvalue})
display(pd.DataFrame(rows).set_index('n').style.format({'D': '{:.4f}', 'p-value': '{:.4f}'}))

# --- 20. Автотесты генератора ---
#@title Автотесты генератора
make = lambda seed: SalinityMixtureGenerator(mu1, s1, mu2, s2, p1, seed)

def test_reproducible():
    assert np.array_equal(make(1).generate(100), make(1).generate(100))

def test_size_and_finite():
    a = make(2).generate(1000)
    assert a.shape == (1000,) and np.isfinite(a).all()

def test_std_normal():
    z = make(3).std_normal(500_000)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01

def test_mixture_moments():
    a = make(4).generate(500_000)
    var_th = p1 * s1**2 + (1 - p1) * s2**2 + p1 * (1 - p1) * (mu1 - mu2)**2
    assert abs(a.mean() - (p1 * mu1 + (1 - p1) * mu2)) < 5 * np.sqrt(var_th / len(a))
    assert abs(a.var() / var_th - 1) < 0.02

for t in (test_reproducible, test_size_and_finite, test_std_normal, test_mixture_moments):
    t()
    print('OK  ', t.__name__)

# --- 21. Интерактивный генератор ---
#@title Интерактивный генератор
import ipywidgets as w

real_sub = salinity.sample(20_000, random_state=0).to_numpy()

def show(p=p1, m1=mu1, sd1=s1, m2=mu2, sd2=s2):
    gen_ = SalinityMixtureGenerator(m1, sd1, m2, sd2, p, seed=0).generate(20_000)
    d = stats.ks_2samp(real_sub, gen_).statistic
    fig, ax = plt.subplots(figsize=(9, 4))
    b = np.linspace(32, 36, 100)
    ax.hist(real_sub, b, density=True, color=BLUE, alpha=.6, label='реальные')
    ax.hist(gen_, b, density=True, histtype='step', color=RED, lw=1.8, label='генератор')
    ax.set(xlabel=XLABEL, ylabel='Плотность, 1/PSU', title=f'KS: D = {d:.4f}')
    ax.legend()
    plt.show()

sl = lambda v, lo, hi, st: w.FloatSlider(value=v, min=lo, max=hi, step=st, continuous_update=False)
w.interact(show, p=sl(p1, .1, .9, .01), m1=sl(mu1, 32.5, 34, .01), sd1=sl(s1, .05, .6, .01),
           m2=sl(mu2, 33.5, 35, .01), sd2=sl(s2, .05, .6, .01));


# ==============================================================================
# Lab01_oxygen_KTvD
# ==============================================================================

# --- 1. Единый стиль графиков {display-mode: "form"} ---
#@title Единый стиль графиков {display-mode: "form"}
import re
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


C = SimpleNamespace(
    DATA="#0072B2",     # реальные данные / генеральная совокупность
    SAMPLE="#56B4E9",   # выборка
    MODEL="#D55E00",
    ACCENT="#009E73",
    WARN="#E69F00",
    REF="#444444",
)
FIG = SimpleNamespace(
    single=(9, 5),
    box=(9, 3),
    wide2=(13, 4.5),
    wide3=(15, 4.5),
)
_CYCLE = [C.DATA, C.MODEL, C.ACCENT, C.WARN, C.SAMPLE, "#CC79A7", "#F0E442", "#000000"]


UNITS = {
    "T_degC":  ("Температура воды", "°C"),
    "Salnty":  ("Солёность (PSS-78)", "PSU"),
    "O2ml_L":  ("Растворённый кислород", "мл/л"),
    "Depthm":  ("Глубина", "м"),
}


def lab(var: str) -> str:
    name, unit = UNITS[var]
    return f"{name}, {unit}"


def dens_lab(var: str) -> str:
    unit = UNITS[var][1]
    return f"Плотность, 1/({unit})" if "/" in unit else f"Плотность, 1/{unit}"


def apply_style() -> None:
    mpl.rcParams.update({
        # размеры и разрешение
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # оси и сетка
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=_CYCLE),
        # линии и легенда
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "mathtext.default": "regular",
    })


class _RuFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        m = re.match(r"^([\u2212-]?)(\d+)(?:\.(\d+))?$", s)
        if not m:
            return s.replace(".", ",")
        sign, ip, fp = m.groups()
        ip = re.sub(r"(?<=\d)(?=(?:\d{3})+$)", "\u2009", ip)
        return sign + ip + ("," + fp if fp else "")


def _ru_formatter():
    f = _RuFormatter(useOffset=False)
    f.set_scientific(False)
    return f


def comma_axes(ax, axis: str = "both") -> None:
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(_ru_formatter())
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(_ru_formatter())


def vline(ax, x, text, color=None, ls="--"):
    ax.axvline(x, color=color or C.REF, ls=ls, lw=1.6, label=text)


def shared_bins(*arrays, n=100):
    import numpy as np
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    return np.linspace(lo, hi, n + 1)

apply_style()

# --- 2. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import kagglehub
path = kagglehub.dataset_download("sohier/calcofi")
print(path)

import os
print(os.listdir(path))

# --- 3. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import pandas as pd

df = pd.read_csv(path + "/bottle.csv", encoding="latin-1", low_memory=False)
print("Строк и столбцов:", df.shape)

oxygen = df["O2ml_L"]
print("Пропусков в кислороде:", oxygen.isna().sum())
print(oxygen.head(10))

# --- 4. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
cast = pd.read_csv(path + "/cast.csv", encoding="latin-1", low_memory=False)
print("Столбцы cast.csv:", list(cast.columns))

cast_small = cast[["Cst_Cnt", "Year", "Month", "Lat_Dec", "Lon_Dec"]]

df = df.merge(cast_small, on="Cst_Cnt", how="left")

def season_of(month):
    if month in (12, 1, 2):
        return "зима"
    elif month in (3, 4, 5):
        return "весна"
    elif month in (6, 7, 8):
        return "лето"
    else:
        return "осень"

df["Season"] = df["Month"].apply(season_of)

print(df[["Depthm", "O2ml_L", "Year", "Month", "Season"]].head())
print(df["Season"].value_counts())

# --- 5. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
oxygen = oxygen.dropna()
print("Осталось значений:", len(oxygen))
print(oxygen.describe())

# --- 6. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=FIG.single)
ax.hist(oxygen, bins=100, color=C.DATA, edgecolor="white")
ax.set_xlabel(lab("O2ml_L")); ax.set_ylabel("Число измерений")
ax.set_title("Распределение растворённого кислорода (CalCOFI)")
comma_axes(ax); plt.show()

# --- 7. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
o2_100 = df.loc[df["Depthm"] == 100, "O2ml_L"].dropna()
print("Число измерений на глубине 100 м:", len(o2_100))
print(o2_100.describe())

# --- 8. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
# самая часто посещаемая станция летом на глубине 100 м
summer_100 = df[(df["Depthm"] == 100) & (df["Season"] == "лето")]
top_station = summer_100["Sta_ID"].value_counts().idxmax()
print("Самая частая станция:", top_station)

steps = {
    "все данные": df["O2ml_L"],
    "глубина 100 м": df.loc[df["Depthm"] == 100, "O2ml_L"],
    "100 м + лето": summer_100["O2ml_L"],
    "100 м + лето + одна станция": summer_100.loc[summer_100["Sta_ID"] == top_station, "O2ml_L"],
}

rows = []
for name, values in steps.items():
    values = values.dropna()
    rows.append({
        "условие": name,
        "n": len(values),
        "min": values.min(),
        "max": values.max(),
        "среднее": values.mean(),
        "std": values.std(),
    })

table = pd.DataFrame(rows).round(2)
print(table)

# --- 9. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
fig, ax = plt.subplots(figsize=FIG.single)
ax.hist(o2_100, bins=50, color=C.SAMPLE, edgecolor="white")
ax.set_xlabel(lab("O2ml_L") + " (глубина 100 м)"); ax.set_ylabel("Число измерений")
ax.set_title("Разброс кислорода на одной глубине")
comma_axes(ax); plt.show()

# --- 10. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
oxy = df["O2ml_L"].dropna()

stats = {
    "Количество": len(oxy),
    "Среднее": oxy.mean(),
    "Медиана": oxy.median(),
    "Стандартное отклонение": oxy.std(),
    "Минимум": oxy.min(),
    "Q1 (25%)": oxy.quantile(0.25),
    "Q3 (75%)": oxy.quantile(0.75),
    "Максимум": oxy.max(),
    "Размах": oxy.max() - oxy.min(),
    "IQR (Q3 − Q1)": oxy.quantile(0.75) - oxy.quantile(0.25),
    "Асимметрия": oxy.skew(),
    "Эксцесс": oxy.kurt(),
}
pd.Series(stats).round(3)

# --- 11. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import matplotlib.pyplot as plt

neg = (oxy < 0).sum()
print("Отрицательных значений:", neg, f"({neg / len(oxy) * 100:.4f} % от всех)")
print("Самые маленькие значения:", sorted(oxy.unique())[:5])

fig, ax = plt.subplots(figsize=FIG.box)
ax.boxplot(oxy, vert=False, whis=1.5, patch_artist=True,
           boxprops=dict(facecolor=C.DATA, edgecolor=C.REF), medianprops=dict(color="white", linewidth=2),
           whiskerprops=dict(color=C.REF), capprops=dict(color=C.REF))
ax.set_xlabel(lab("O2ml_L")); ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.set_title("Ящик с усами: кислород, все глубины")
comma_axes(ax, "x"); plt.show()

# --- 12. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import numpy as np

oxygen = df['O2ml_L'].dropna()
print("Настоящее среднее:", round(oxygen.mean(), 3))
print("Настоящая дисперсия:", round(oxygen.var(), 3))

for n in [10, 100, 1000, 10000, 100000]:
    sample = oxygen.sample(n=n, random_state=1)
    print(n, round(sample.mean(), 3), round(sample.var(), 3))

# --- 13. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import matplotlib.pyplot as plt

sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
means = []
variances = []

for n in sizes:
    sample = oxygen.sample(n=n, random_state=42)
    means.append(sample.mean())
    variances.append(sample.var())

fig, ax = plt.subplots(1, 2, figsize=FIG.wide2)

ax[0].plot(sizes, means, marker="o", ms=4, color=C.DATA, label="Выборочное среднее")
ax[0].axhline(oxygen.mean(), color=C.MODEL, linestyle="--", label="Истинное среднее")
ax[0].set_ylabel("Выборочное среднее, мл/л")
ax[0].set_title("Состоятельность оценки среднего")

ax[1].plot(sizes, variances, marker="o", ms=4, color=C.DATA, label="Выборочная дисперсия")
ax[1].axhline(oxygen.var(), color=C.MODEL, linestyle="--", label="Истинная дисперсия")
ax[1].set_ylabel("Выборочная дисперсия, (мл/л)²")
ax[1].set_title("Состоятельность оценки дисперсии")

for a in ax:
    a.set_xscale("log"); a.set_xlabel(r"Объём выборки $n$ (лог. шкала)"); a.legend(); comma_axes(a, "y")
plt.tight_layout(); plt.show()

# --- 14. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
pairs = df[["T_degC", "O2ml_L"]].dropna()
print("Проб, где есть и температура, и кислород:", len(pairs))

pairs_plot = pairs.sample(n=20000, random_state=42)

fig, ax = plt.subplots(figsize=FIG.single)
ax.scatter(pairs_plot["T_degC"], pairs_plot["O2ml_L"], s=3, alpha=0.4, color=C.DATA)
ax.set_xlabel(lab("T_degC")); ax.set_ylabel(lab("O2ml_L"))
ax.set_title("Кислород и температура в одних и тех же пробах")
comma_axes(ax); plt.show()

# --- 15. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
from matplotlib.colors import LogNorm

depth = df.loc[pairs_plot.index, "Depthm"].clip(lower=1)

fig, ax = plt.subplots(figsize=FIG.single)
sc = ax.scatter(pairs_plot["T_degC"], pairs_plot["O2ml_L"],
                c=depth, norm=LogNorm(vmin=1, vmax=depth.max()),
                s=3, alpha=0.4, cmap="viridis_r")
fig.colorbar(sc, ax=ax, label=lab("Depthm") + " (лог. шкала)")
ax.set_xlabel(lab("T_degC")); ax.set_ylabel(lab("O2ml_L"))
ax.set_title("Кислород и температура, цвет показывает глубину")
comma_axes(ax); plt.show()

# --- 16. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
from scipy.stats import pearsonr, spearmanr

r_pearson, p_pearson = pearsonr(pairs["T_degC"], pairs["O2ml_L"])
r_spearman, p_spearman = spearmanr(pairs["T_degC"], pairs["O2ml_L"])

print(f"Пирсон:  r = {r_pearson:.3f}, p-value = {p_pearson:.2e}")
print(f"Спирмен: ρ = {r_spearman:.3f}, p-value = {p_spearman:.2e}")

# --- 17. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
oxygen_minmax = (oxygen - oxygen.min()) / (oxygen.max() - oxygen.min())
oxygen_z = (oxygen - oxygen.mean()) / oxygen.std()

print("Исходные данные:   min =", round(oxygen.min(), 3),
      " max =", round(oxygen.max(), 3),
      " среднее =", round(oxygen.mean(), 3),
      " std =", round(oxygen.std(), 3))
print("После min-max:     min =", round(oxygen_minmax.min(), 3),
      " max =", round(oxygen_minmax.max(), 3),
      " среднее =", round(oxygen_minmax.mean(), 3),
      " std =", round(oxygen_minmax.std(), 3))
print("После Z-преобразования: min =", round(oxygen_z.min(), 3),
      " max =", round(oxygen_z.max(), 3),
      " среднее =", round(oxygen_z.mean(), 3),
      " std =", round(oxygen_z.std(), 3))

fig, ax = plt.subplots(1, 3, figsize=FIG.wide3)
for a, data, title, xl in zip(
        ax, [oxygen, oxygen_minmax, oxygen_z],
        ["Исходные данные", "Min-max нормализация", "Z-преобразование"],
        [lab("O2ml_L"), "Значение после min-max, безразм.", "Z-оценка, σ"]):
    a.hist(data, bins=100, color=C.DATA, edgecolor="white")
    a.set_title(title); a.set_xlabel(xl); a.set_ylabel("Число измерений"); comma_axes(a)
plt.tight_layout(); plt.show()

# --- 18. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import pandas as pd

sample_with = oxygen.sample(n=1000, replace=True, random_state=42)
sample_without = oxygen.sample(n=1000, replace=False, random_state=42)

comparison = pd.DataFrame({
    "Генеральная": [oxygen.mean(), oxygen.median(), oxygen.var(), oxygen.std()],
    "С возвращением": [sample_with.mean(), sample_with.median(), sample_with.var(), sample_with.std()],
    "Без возвращения": [sample_without.mean(), sample_without.median(), sample_without.var(), sample_without.std()],
}, index=["Среднее", "Медиана", "Дисперсия", "Ст. отклонение"])

print(comparison.round(3))

# --- 19. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np

k = 3
gm = GaussianMixture(n_components=k, random_state=42)
gm.fit(oxygen.values.reshape(-1, 1))

weights = gm.weights_
means = gm.means_.flatten()
stds = np.sqrt(gm.covariances_.flatten())

for i in range(k):
    print(f"Компонента {i+1}: вес = {weights[i]:.3f}, среднее = {means[i]:.3f}, ст. отклонение = {stds[i]:.3f}")

x = np.linspace(oxygen.min(), oxygen.max(), 500)
density = sum(weights[i] * norm.pdf(x, means[i], stds[i]) for i in range(k))

fig, ax = plt.subplots(figsize=FIG.single)
ax.hist(oxygen, bins=100, density=True, color=C.DATA, edgecolor="white", label="Данные")
ax.plot(x, density, color=C.MODEL, linewidth=2, label="Смесь трёх нормальных")
ax.set_xlabel(lab("O2ml_L")); ax.set_ylabel(dens_lab("O2ml_L"))
ax.set_title("Данные и смесь трёх нормальных распределений")
ax.legend(); comma_axes(ax); plt.show()

# --- 20. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
from scipy import stats

sample_test = oxygen.sample(n=1000, random_state=42)
stat, p_value = stats.shapiro(sample_test)

print(f"Статистика Шапиро–Уилка: {stat:.4f}")
print(f"p-value: {p_value:.2e}")

alpha = 0.05
if p_value < alpha:
    print("p-value < 0.05, отвергаем H0: распределение не нормальное")
else:
    print("p-value >= 0.05, нет оснований отвергнуть H0")


# ==============================================================================
# Lab01_temperatur_KTvD
# ==============================================================================

# --- 1. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SEED = 42
rng = np.random.default_rng(SEED)

# --- 2. Для графиков {display-mode: "form"} ---
#@title Для графиков {display-mode: "form"}
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

C = SimpleNamespace(
    DATA="#0072B2",
    SAMPLE="#56B4E9",
    MODEL="#D55E00",
    WARN="#E69F00",
    REF="#444444",
)
_CYCLE = [C.DATA, C.MODEL, C.WARN, C.SAMPLE, "#CC79A7", "#F0E442", "#000000"]


#
UNITS = {
    "T_degC":("Температура воды","°C"),
    "Salnty":("Соленость(PSS-78)","PSU"),
    "02ml_L":("Растворенный кислоро","мл/л"),
    "Depthm":("Глубина","м"),
}

def lab(var: str) -> str:
    name, unit =UNITS[var]
    return f"{name}, {unit}"


def dens_lab(var: str) -> str:
    name, unit =UNITS[var][1]
    return f"Плотность, 1/{unit}"


def apply_style() -> None:
    mpl.rcParams.update({
        # размеры и разрешение
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        # шрифты
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # оси и сетка
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=_CYCLE),
        # линии и легенда
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "mathtext.default": "regular",
    })


def _comma(x, _pos=None):
    s = f"{x:g}"
    return s.replace(".", ",")


def comma_axes(ax, axis: str = "both") -> None:
    f = FuncFormatter(_comma)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def vline(ax, x, text, color=None, ls="--"):
    ax.axvline(x, color=color or C.REF, ls=ls, lw=1.6, label=text)


def shared_bins(*arrays, n=100):
    import numpy as np
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    return np.linspace(lo, hi, n + 1)

apply_style()

# --- 3. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
import kagglehub
path = kagglehub.dataset_download("sohier/calcofi")
DATA_PATH = path + "/bottle.csv"

df = pd.read_csv(DATA_PATH, usecols=["Sta_ID", "Depthm", "T_degC", "Salnty"],
                 encoding="latin-1", low_memory=False)
print("Размер таблицы:", df.shape)
df.head()

# --- 4. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
print("Пропуски в каждой колонке:")
print(df.isna().sum())

temp = df["T_degC"].dropna().reset_index(drop=True)
N = len(temp)
print(f"\nN (размер генеральной совокупности) = {N}")

# --- 5. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
DEPTH = 0
surface = df.loc[df["Depthm"] == DEPTH, "T_degC"].dropna()

print(f"Измерений на глубине {DEPTH} м: {len(surface)}")
print(f"Количество разных значений температуры: {surface.nunique()}")
print(f"Минимум: {surface.min():.2f} °C, максимум: {surface.max():.2f} °C")
print(f"Среднее: {surface.mean():.2f} °C, стд. отклонение: {surface.std():.2f} °C")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(surface, bins=40, kde=True, color=C.DATA, ax=ax[0])
ax[0].set_title(f"Температура на глубине {DEPTH} м")
ax[0].set_xlabel(lab("T_degC")); ax[0].set_ylabel("Число измерений")

ax[1].plot(surface.values[:300], marker=".", lw=0.7, color=C.DATA)
ax[1].set_title("Первые 300 измерений на этой глубине (по порядку в файле)")
ax[1].set_xlabel("Номер измерения"); ax[1].set_ylabel(lab("T_degC"))
for a in ax: comma_axes(a, "y")
comma_axes(ax[0], "x")
plt.tight_layout(); plt.show()

# --- 6. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}

sub = df[["Depthm", "T_degC"]].dropna()
x = np.log1p(sub["Depthm"].values)
y = sub["T_degC"].values

coef = np.polyfit(x, y, deg=5)
y_hat = np.polyval(coef, x)
r2 = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
resid_std = np.std(y - y_hat)

print(f"R^2 модели 'температура = f(глубина)': {r2:.3f}")
print(f"Стд. отклонение остатков: {resid_std:.2f} °C")

idx = rng.choice(len(sub), size=min(20000, len(sub)), replace=False)
fig, ax = plt.subplots()
ax.scatter(sub["Depthm"].values[idx], y[idx], s=3, alpha=0.25, color=C.DATA, label="измерения")
xs = np.linspace(x.min(), x.max(), 300)
ax.plot(np.expm1(xs), np.polyval(coef, xs), color=C.MODEL, lw=2, label=r"лучшая кривая $T = f(\text{depth})$")
ax.set_xscale("symlog")
ax.set_xlabel(lab("Depthm") + " (симлог-шкала)"); ax.set_ylabel(lab("T_degC"))
ax.set_title("Температура воды и глубина")

# --- 7. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
desc = temp.describe()
extra = pd.Series({
    "variance": temp.var(ddof=1),
    "median": temp.median(),
    "mode (округл. до 0.1)": temp.round(1).mode().iloc[0],
    "skewness (асимметрия)": stats.skew(temp),
    "kurtosis (эксцесс)": stats.kurtosis(temp),
    "range (размах)": temp.max() - temp.min(),
})
pd.concat([desc, extra]).to_frame("T_degC").round(3)

# --- 8. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(temp, bins=60, stat="density", kde=True, color=C.DATA, ax=ax[0])
ax[0].axvline(temp.mean(), color=C.REF, ls="--", lw=1.6, label=f"среднее = {temp.mean():.2f}".replace(".", ","))
ax[0].axvline(temp.median(), color=C.REF, ls=":", lw=2.0, label=f"медиана = {temp.median():.2f}".replace(".", ","))
ax[0].set_title("Гистограмма температуры воды (все глубины)")
ax[0].set_xlabel(lab("T_degC")); ax[0].set_ylabel(dens_lab("T_degC"))
ax[0].legend()

sns.boxplot(x=temp, ax=ax[1], color=C.SAMPLE)
ax[1].set_title("Боксплот температуры")
ax[1].set_xlabel(lab("T_degC")); ax[1].grid(axis="y", visible=False)
comma_axes(ax[0]); comma_axes(ax[1], "x")
plt.tight_layout(); plt.show()

# --- 9. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
q1, q3 = temp.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = temp[(temp < low) | (temp > high)]

print(f"Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
print(f"Границы без выбросов: [{low:.2f}, {high:.2f}]")
print(f"Выбросов: {len(outliers)} ({100 * len(outliers) / N:.3f}% данных)")
if len(outliers):
    print(f"Выбросы лежат в диапазоне {outliers.min():.2f} ... {outliers.max():.2f} °C")

# --- 10. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
temp_minmax = (temp - temp.min()) / (temp.max() - temp.min())
temp_z = (temp - temp.mean()) / temp.std(ddof=1)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, data, title_, xl, yl in zip(
        ax, [temp, temp_minmax, temp_z],
        ["Исходные данные", "Min-max", "Z-преобразование"],
        [lab("T_degC"), "Значение после min-max, безразм.", "Z-оценка, σ"],
        [dens_lab("T_degC"), "Плотность, безразм.", "Плотность, 1/σ"]):
    sns.histplot(data, bins=50, stat="density", color=C.DATA, ax=a)
    a.set_title(title_); a.set_xlabel(xl); a.set_ylabel(yl); comma_axes(a)
plt.tight_layout(); plt.show()

check = pd.DataFrame({
    "исходные": [temp.min(), temp.max(), temp.mean(), temp.std(), stats.skew(temp)],
    "min-max":  [temp_minmax.min(), temp_minmax.max(), temp_minmax.mean(), temp_minmax.std(), stats.skew(temp_minmax)],
    "z-score":  [temp_z.min(), temp_z.max(), temp_z.mean(), temp_z.std(), stats.skew(temp_z)],
}, index=["min", "max", "mean", "std", "skew"]).round(4)
check

# --- 11. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
arr = temp.values
mu_true = arr.mean()
var_true = arr.var(ddof=1)

sizes = np.unique(np.logspace(1, np.log10(N // 2), 25).astype(int))
REPS = 30

mean_res = np.zeros((len(sizes), REPS))
var_res = np.zeros((len(sizes), REPS))
for i, n in enumerate(sizes):
    for r in range(REPS):
        s = rng.choice(arr, size=n, replace=False)
        mean_res[i, r] = s.mean()
        var_res[i, r] = s.var(ddof=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for a, res, true, name in zip(ax, [mean_res, var_res], [mu_true, var_true], ["Среднее", "Дисперсия"]):
    a.plot(sizes, res.mean(axis=1), marker="o", ms=4, color=C.DATA, label=f"{name} выборки (среднее по {REPS} повторам)")
    a.fill_between(sizes, res.min(axis=1), res.max(axis=1), color=C.SAMPLE, alpha=0.3, label="min–max по повторам")
    a.axhline(true, color=C.MODEL, ls="--", label=f"истинное значение = {true:.2f}".replace(".", ","))
    a.set_xscale("log"); a.set_xlabel(r"Объём выборки $n$ (лог. шкала)")
    a.set_title(f"Состоятельность: {name.lower()}")
    a.legend(); comma_axes(a, "y")
ax[0].set_ylabel("Среднее, °C"); ax[1].set_ylabel("Дисперсия, °C²")
plt.tight_layout(); plt.show()

# --- 12. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
err_mean = np.abs(mean_res - mu_true).mean(axis=1)
fig, ax = plt.subplots()
ax.loglog(sizes, err_mean, marker="o", ms=4, color=C.DATA, label=r"средняя $|\bar{x}-\mu|$")
ax.loglog(sizes, np.sqrt(var_true) / np.sqrt(sizes), color=C.MODEL, ls="--", label=r"теория $\sigma/\sqrt{n}$")
ax.set_xlabel(r"Объём выборки $n$ (лог. шкала)"); ax.set_ylabel("Ошибка оценки среднего, °C (лог. шкала)")
plt.title("Скорость сходимости среднего (двойная лог. шкала)")
plt.legend(); plt.show()

# --- 13. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
n = 5000

samp_with = rng.choice(arr, size=n, replace=True)
samp_without = rng.choice(arr, size=n, replace=False)

def stat_row(x, ddof=1):
    return {"n": len(x), "mean": np.mean(x), "median": np.median(x), "var": np.var(x, ddof=ddof),
            "std": np.std(x, ddof=ddof), "min": np.min(x), "max": np.max(x), "skew": stats.skew(x)}

table = pd.DataFrame({
    "генеральная (все данные)": stat_row(arr),
    "с возвращением": stat_row(samp_with),
    "без возвращения": stat_row(samp_without),
}).T.round(3)
table

# --- 14. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
rel = table.loc[["с возвращением", "без возвращения"], ["mean", "median", "var", "std"]]
rel = (rel - table.loc["генеральная (все данные)", ["mean", "median", "var", "std"]]).abs() \
      / table.loc["генеральная (все данные)", ["mean", "median", "var", "std"]] * 100
print("Относительное отклонение от генеральной совокупности, %:")
rel.round(3)

# --- 15. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
pop = rng.choice(arr, size=2000, replace=False)
N_small, n_small, B = len(pop), 1500, 3000

means_with = np.array([rng.choice(pop, n_small, replace=True).mean() for _ in range(B)])
means_without = np.array([rng.choice(pop, n_small, replace=False).mean() for _ in range(B)])

sigma2 = pop.var(ddof=0)
theory_with = sigma2 / n_small
theory_without = sigma2 / n_small * (N_small - n_small) / (N_small - 1)

res = pd.DataFrame({
    "Var(x̄) по эксперименту": [means_with.var(), means_without.var()],
    "Var(x̄) по теории": [theory_with, theory_without],
}, index=["с возвращением", "без возвращения"])
print(f"Среднее генеральной (малой) совокупности: {pop.mean():.3f}")
display(res.round(6))

fig, ax = plt.subplots()
bins = shared_bins(means_with, means_without, n=40)
sns.histplot(means_with, bins=bins, stat="density", color=C.DATA, alpha=0.5, label="с возвращением", ax=ax)
sns.histplot(means_without, bins=bins, stat="density", color=C.SAMPLE, alpha=0.5, label="без возвращения", ax=ax)
vline(ax, pop.mean(), "истинное среднее")
ax.set_xlabel("Выборочное среднее, °C"); ax.set_ylabel(dens_lab("T_degC"))
ax.set_title(f"Распределение выборочных средних ($n$={n_small} из $N$={N_small})")
ax.legend(); comma_axes(ax); plt.show()

# --- 16. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}

fit_data = rng.choice(arr, size=20000, replace=False)
positive = fit_data.min() > 0
print("Все значения > 0:", positive)

candidates = {"norm": stats.norm, "skewnorm": stats.skewnorm}
if positive:
    candidates.update({"lognorm": stats.lognorm, "gamma": stats.gamma, "weibull_min": stats.weibull_min})

rows, fitted = [], {}
for name, dist in candidates.items():
    params = dist.fit(fit_data)
    ll = np.sum(dist.logpdf(fit_data, *params))
    k = len(params)
    ks = stats.kstest(fit_data, dist.cdf, args=params).statistic
    rows.append({"распределение": name, "параметры": np.round(params, 3), "KS D": ks, "AIC": 2 * k - 2 * ll})
    fitted[name] = params

fit_table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
fit_table

# --- 17. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
xs = np.linspace(arr.min(), arr.max(), 500)
fig, ax = plt.subplots()
sns.histplot(arr, bins=70, stat="density", color="#CCCCCC", label="данные", ax=ax)
for name, params in fitted.items():
    ax.plot(xs, candidates[name].pdf(xs, *params), lw=2, label=name)
ax.set_xlabel(lab("T_degC")); ax.set_ylabel(dens_lab("T_degC"))
ax.set_title("Данные и подобранные теоретические распределения")
ax.legend(); comma_axes(ax); plt.show()

# --- 18. 1) Шапиро-Уилк {display-mode: "form"} ---
#@title 1) Шапиро-Уилк {display-mode: "form"}
alpha = 0.05
test_sample = rng.choice(arr, size=5000, replace=False)

# 1) Шапиро-Уилк
sh_stat, sh_p = stats.shapiro(test_sample)
# 2) Д'Агостино-Пирсон (по асимметрии и эксцессу)
da_stat, da_p = stats.normaltest(test_sample)
# 3) Колмогоров-Смирнов с оценёнными параметрами (p-value приближённый, т.к. параметры оценены по этим же данным)
ks_stat, ks_p = stats.kstest(test_sample, stats.norm(loc=test_sample.mean(), scale=test_sample.std(ddof=1)).cdf)

res = pd.DataFrame({"статистика": [sh_stat, da_stat, ks_stat], "p-value": [sh_p, da_p, ks_p]},
                   index=["Shapiro-Wilk", "D'Agostino-Pearson", "Kolmogorov-Smirnov"])
res["решение при α=0.05"] = np.where(res["p-value"] < alpha, "отвергаем H0", "не отвергаем H0")
display(res)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
stats.probplot(test_sample, dist="norm", plot=ax[0])
ax[0].get_lines()[0].set(color=C.DATA, markersize=3); ax[0].get_lines()[1].set_color(C.MODEL)
ax[0].set_title("Q–Q график: данные против нормального")
ax[0].set_xlabel("Теоретические квантили, σ"); ax[0].set_ylabel("Выборочные квантили, °C")
sns.histplot(test_sample, bins=50, stat="density", color=C.SAMPLE, ax=ax[1])
xx = np.linspace(test_sample.min(), test_sample.max(), 300)
ax[1].plot(xx, stats.norm.pdf(xx, test_sample.mean(), test_sample.std(ddof=1)), color=C.MODEL, lw=2,
           label=r"нормальное $\mathcal{N}(\bar{x}, s^2)$")
ax[1].set_xlabel(lab("T_degC")); ax[1].set_ylabel(dens_lab("T_degC")); ax[1].legend()
ax[1].set_title("Подвыборка и нормальное приближение")
comma_axes(ax[1])
plt.tight_layout(); plt.show()

# --- 19. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
surf = surface.values
print(f"Измерений на поверхности: {len(surf)}")
surf_test = rng.choice(surf, size=min(5000, len(surf)), replace=False)

sh_stat, sh_p = stats.shapiro(surf_test)
da_stat, da_p = stats.normaltest(surf_test)
print(f"Shapiro-Wilk:        W = {sh_stat:.4f}, p = {sh_p:.3g}")
print(f"D'Agostino-Pearson:  stat = {da_stat:.2f}, p = {da_p:.3g}")
print(f"skew = {stats.skew(surf):.3f}, excess kurtosis = {stats.kurtosis(surf):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
stats.probplot(surf_test, dist="norm", plot=ax[0])
ax[0].get_lines()[0].set(color=C.DATA, markersize=3); ax[0].get_lines()[1].set_color(C.MODEL)
ax[0].set_title("Q–Q график, поверхность (0 м)")
ax[0].set_xlabel("Теоретические квантили, σ"); ax[0].set_ylabel("Выборочные квантили, °C")
sns.histplot(surf_test, bins=40, stat="density", color=C.SAMPLE, ax=ax[1])
xx = np.linspace(surf_test.min(), surf_test.max(), 300)
ax[1].plot(xx, stats.norm.pdf(xx, surf_test.mean(), surf_test.std(ddof=1)), color=C.MODEL, lw=2,
           label=r"нормальное $\mathcal{N}(\bar{x}, s^2)$")
ax[1].set_xlabel(lab("T_degC")); ax[1].set_ylabel(dens_lab("T_degC")); ax[1].legend()
ax[1].set_title("Поверхность и нормальное приближение")
comma_axes(ax[1])
plt.tight_layout(); plt.show()

# --- 20. Код ячейки {display-mode: "form"} ---
#@title Код ячейки {display-mode: "form"}
deep = df.loc[df["Depthm"] == 500, "T_degC"].dropna().values
print(f"Измерений на 500 м: {len(deep)}; среднее = {deep.mean():.2f} °C; на поверхности: {surf.mean():.2f} °C")

t_stat, t_p = stats.ttest_ind(surf, deep, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(surf, deep, alternative="two-sided")
print(f"Welch t-test:  t = {t_stat:.2f}, p = {t_p:.3g}")
print(f"Mann-Whitney:  U = {u_stat:.0f}, p = {u_p:.3g}")

diff = surf.mean() - deep.mean()
se = np.sqrt(surf.var(ddof=1) / len(surf) + deep.var(ddof=1) / len(deep))
print(f"Разность средних = {diff:.2f} °C, 95% ДИ ≈ [{diff - 1.96*se:.2f}, {diff + 1.96*se:.2f}]")
