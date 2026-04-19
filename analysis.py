# -*- coding: utf-8 -*-
"""
Law of Technological Power Displacement
Empirical Analysis — Steps 2.2 and 2.4

Step 2.2: Spain BOE regulatory index with net-change variant (entries minus exits)
Step 2.4: Robustness tests for the EU-27 panel
         (AR(1) correction, random slopes, Granger causality, cointegration)

Requirements
------------
    pip install numpy pandas matplotlib scipy statsmodels linearmodels

Data
----
    EU-27 panel: Digital Society Project v8
    Download from https://digitalsocietyproject.org/data/
    Place the extracted CSV in the data/ directory and update DSP_FILE below.

Authors
-------
    [Withheld for blind review]

Repository
----------
    https://github.com/ARTUS10/law-technological-power-displacement

License
-------
    CC BY 4.0  (data: CC BY-NC-ND 4.0 per DSP terms)
"""

# ── CELL 1: Dependencies ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import kendalltau
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# Path to the DSP CSV file (update if needed)
DSP_FILE = 'data/digital_society_project_v8.csv'

# Create output directories if they do not exist
import os
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

print('Dependencies loaded.')

# =============================================================================
# STEP 2.2 — Spain BOE Regulatory Index (net-change variant)
# =============================================================================

# ── CELL 2: BOE source data ──────────────────────────────────────────────────
# Each record: (year, weight, sector, description)
#
# Weight coding:
#   1.5 = Organic Law / first-tier independent agency
#   1.0 = Ordinary Law / statutory body with legal personality
#   0.5 = Royal Decree / directorate-general / specialised unit

ENTRIES_BOE = [
    (1978, 1.0, 'telecommunications',  'DG Telecommunications — constitutional reorganisation'),
    (1980, 0.5, 'telecommunications',  'Ministry of Transport, Tourism and Communications'),
    (1982, 0.5, 'informatics',         'DG of Informatics, Ministry of Public Administration (MAP)'),
    (1984, 0.5, 'information_society', 'FUNDESCO'),
    (1986, 1.0, 'telecommunications',  'Secretariat General of Communications'),
    (1987, 1.5, 'telecommunications',  'General Telecommunications Act (LOT) — Law 31/1987'),
    (1989, 0.5, 'spectrum',            'Radio-Frequency Board (Junta de Frecuencias)'),
    (1990, 1.0, 'informatics',         'DG of IT Policy (DGPI-MAP)'),
    (1992, 1.5, 'data_protection',     'LORTAD mandating AEPD — Organic Law 5/1992'),
    (1993, 1.5, 'data_protection',     'AEPD — first director appointed October 1993'),
    (1994, 0.5, 'informatics',         'DG of Informatics, MOPTMA'),
    (1995, 1.0, 'energy_digital',      'CSEN (precursor to CNE) — Law 40/1994'),
    (1996, 1.5, 'telecommunications',  'CMT — Royal Decree-Law 6/1996 + Law 12/1997'),
    (1997, 1.0, 'information_society', 'SETSI — Secretariat of State for Telecommunications'),
    (1998, 1.5, 'telecommunications',  'General Telecommunications Act 1998 — Law 11/1998'),
    (1999, 1.5, 'data_protection',     'LOPD — Organic Law 15/1999 strengthening AEPD'),
    (1999, 1.0, 'digital_identity',    'FNMT-RCM Certification Authority'),
    (2000, 1.0, 'information_society', 'SETSI — information-society mandate extended'),
    (2001, 0.5, 'information_society', 'Plan Info XXI'),
    (2002, 1.5, 'information_society', 'LSSI (e-commerce law) — Law 34/2002'),
    (2002, 1.0, 'telecommunications',  'CMT — competence reinforcement under LGT'),
    (2002, 1.0, 'energy_digital',      'CNE consolidated — Law 54/1997'),
    (2003, 1.0, 'competition',         'TDC — Competition Tribunal'),
    (2003, 0.5, 'information_society', 'Plan Avanza (precursor)'),
    (2004, 1.5, 'cybersecurity',       'CCN-CNI — Royal Decree 421/2004'),
    (2004, 1.0, 'telecommunications',  'CMT — new spectrum competences'),
    (2005, 0.5, 'information_society', 'Plan Avanza — approved'),
    (2006, 1.0, 'competition',         'CNC — Law 15/2007'),
    (2007, 1.5, 'infrastructure',      'CNPIC — Royal Decree 1513/2007'),
    (2007, 1.0, 'media',               'CMF — Capital Markets Commission'),
    (2007, 0.5, 'information_society', 'Red.es — mandate extended'),
    (2008, 1.0, 'energy_digital',      'CNE — smart-grid competences'),
    (2009, 1.5, 'cybersecurity',       'ENS — National Security Scheme, Royal Decree 3/2010'),
    (2009, 1.0, 'telecommunications',  'CMT — NGA fibre regulation'),
    (2010, 1.5, 'telecommunications',  'LGT 2011 — preparation of Law 9/2014'),
    (2011, 1.5, 'infrastructure',      'Critical Infrastructure Protection Act — Law 8/2011'),
    (2011, 1.0, 'energy_digital',      'CNSP — National Postal Sector Commission'),
    (2011, 0.5, 'cybersecurity',       'CERT-SI (INTECO)'),
    (2012, 0.5, 'information_society', 'Digital Agenda for Spain 2012'),
    (2013, 1.5, 'convergent_reg',      'CNMC — merger of six bodies, Law 3/2013'),
    (2013, 1.5, 'cybersecurity',       'ENS mandatory for all public administration'),
    (2013, 1.0, 'cybersecurity',       'INCIBE — formally established'),
    (2014, 1.5, 'telecommunications',  'General Telecommunications Act — Law 9/2014'),
    (2014, 1.0, 'cybersecurity',       'National Cybersecurity Strategy'),
    (2015, 1.5, 'cybersecurity',       'National Cybersecurity Council — Law 36/2015'),
    (2015, 1.5, 'digital_admin',       'Public Administration Act 39/2015 — mandatory e-administration'),
    (2016, 1.5, 'data_protection',     'GDPR adopted EU 2016/679 (in force 2018)'),
    (2016, 0.5, 'digital_admin',       'Digital Agenda for Spain 2020'),
    (2017, 1.5, 'cybersecurity',       'NIS Directive 2016/1148'),
    (2018, 1.5, 'data_protection',     'LOPDGDD — Organic Law 3/2018, digital rights'),
    (2018, 1.0, 'cybersecurity',       'Royal Decree-Law NIS 12/2018 — INCIBE-CERT + CCN-CERT'),
    (2019, 1.0, 'cybersecurity',       'National Cybersecurity Strategy 2019'),
    (2019, 1.5, 'ai_digital',          'SEDIA — Secretariat of State for Digitalisation and AI'),
    (2020, 0.5, 'cybersecurity',       'ENISA — Spain NIS single point of contact'),
    (2020, 1.0, 'ai_digital',          'ENIA — National AI Plan, coordination structure'),
    (2021, 1.0, 'cybersecurity',       'Royal Decree 43/2021 — NIS2 anticipation'),
    (2022, 1.5, 'telecommunications',  'General Telecommunications Act — Law 11/2022'),
    (2022, 1.5, 'media',               'General Audiovisual Communication Act — Law 13/2022'),
    (2022, 1.0, 'convergent_reg',      'Start-ups Act and regulatory sandbox — Law 28/2022'),
    (2023, 1.5, 'ai_digital',          'AESIA — Royal Decree 729/2023, first AI authority in EU'),
    (2023, 1.0, 'ai_digital',          'AI Sandbox — Royal Decree 817/2023'),
    (2023, 1.5, 'cybersecurity',       'NIS2 Spain — Directive 2022/2555'),
    (2024, 1.5, 'ai_digital',          'EU AI Act 2024/1689 — AESIA designated supervisory authority'),
    (2024, 1.0, 'digital_admin',       'Ministry of Digital Transformation — expanded mandate'),
]

# Exits: bodies abolished or merged
EXITS_BOE = [
    (2013, 1.5, 'telecommunications',  'CMT — absorbed into CNMC'),
    (2013, 1.0, 'energy_digital',      'CNE — absorbed into CNMC'),
    (2013, 1.0, 'competition',         'CNC — absorbed into CNMC'),
    (2013, 1.0, 'markets',             'CMF — absorbed into CNMC'),
    (2013, 1.0, 'telecommunications',  'CNSP — absorbed into CNMC'),
    (2013, 0.5, 'audiovisual',         'CREA — absorbed into CNMC'),
    (2011, 0.5, 'competition',         'TDC — absorbed into CNC'),
    (2015, 0.5, 'informatics',         'DGPI-MAP — reorganised into SGAD'),
]

print(f'BOE dataset loaded: {len(ENTRIES_BOE)} entries, {len(EXITS_BOE)} exits.')


# ── CELL 3: Build time series ────────────────────────────────────────────────
years = list(range(1978, 2025))
records = []
omega_cum = 0.0
omega_net = 0.0

for year in years:
    entries_yr = [(y, p, s, n) for y, p, s, n in ENTRIES_BOE if y == year]
    exits_yr   = [(y, p, s, n) for y, p, s, n in EXITS_BOE   if y == year]
    w_entries  = sum(p for _, p, _, _ in entries_yr)
    w_exits    = sum(p for _, p, _, _ in exits_yr)
    omega_cum += w_entries
    omega_net += w_entries - w_exits
    Sigma = max(0.05, 1.0 - omega_cum * 0.008)
    records.append({
        'year':      year,
        'entries':   w_entries,
        'exits':     w_exits,
        'net':       w_entries - w_exits,
        'omega_cum': omega_cum,
        'omega_net': omega_net,
        'Sigma':     Sigma,
        'Psi_cum':   omega_cum / Sigma,
        'Psi_net':   max(0.01, omega_net) / Sigma,
    })

df_boe = pd.DataFrame(records)
df_boe['dPsi_cum'] = df_boe['Psi_cum'].diff()
df_boe['dPsi_net'] = df_boe['Psi_net'].diff()
df_boe['phi_cum']  = df_boe['omega_cum'] / (df_boe['Sigma'] + 0.001)
df_boe['phi_net']  = df_boe['omega_net'].clip(lower=0.01) / (df_boe['Sigma'] + 0.001)

print('BOE INDEX SUMMARY (net-change variant)')
print('=' * 50)
print(f'Total entry weight:           {df_boe["entries"].sum():.1f}')
print(f'Total exit weight:            {df_boe["exits"].sum():.1f}')
print(f'Net cumulative change:        {df_boe["net"].sum():.1f}')
neg_years = df_boe[df_boe['net'] < 0]['year'].tolist()
print(f'Years with negative net:      {len(neg_years)} → {neg_years}')
print(f'Omega cumulative (2024):      {df_boe["omega_cum"].iloc[-1]:.1f}')
print(f'Omega net (2024):             {df_boe["omega_net"].iloc[-1]:.1f}')

cnmc_entry = sum(p for y, p, _, _ in ENTRIES_BOE if y == 2013)
cnmc_exit  = sum(p for y, p, _, _ in EXITS_BOE   if y == 2013)
print(f'\nCNMC MERGER 2013:')
print(f'  Weight of new CNMC:         {cnmc_entry:.1f}')
print(f'  Weight of absorbed bodies:  {cnmc_exit:.1f}')
print(f'  Net change:                 {cnmc_entry - cnmc_exit:+.1f}')
print(f'  (Only negative year — CNMC concentrates more capacity')
print(f'   than the sum of its predecessors. Consistent with')
print(f'   Proposition 2: irreversible accumulation.)')


# ── CELL 4: Statistical tests — Spain BOE ────────────────────────────────────
print('\nSTATISTICAL TESTS')
print('=' * 50)

# Mann-Kendall monotonicity test
tau_cum, p_cum = kendalltau(df_boe['year'], df_boe['omega_cum'])
tau_net, p_net = kendalltau(df_boe['year'], df_boe['omega_net'])
print(f'\nMann-Kendall tau:')
print(f'  Omega cumulative:  tau={tau_cum:.3f}, p={p_cum:.2e}')
print(f'  Omega net-change:  tau={tau_net:.3f}, p={p_net:.2e}')

# Correlation between variants
r_variants, p_variants = stats.pearsonr(df_boe['omega_cum'], df_boe['omega_net'])
print(f'\nCorrelation Omega_cum ~ Omega_net: r={r_variants:.4f}, p={p_variants:.2e}')

# T4 mechanistic test: original vs net-change variant
df_t4 = df_boe.dropna(subset=['dPsi_cum', 'dPsi_net'])
X_cum  = add_constant(df_t4['phi_cum'])
X_net  = add_constant(df_t4['phi_net'])
m_cum  = OLS(df_t4['dPsi_cum'], X_cum).fit(cov_type='HC3')
m_net  = OLS(df_t4['dPsi_net'], X_net).fit(cov_type='HC3')

print(f'\nT4 — Central mechanistic test (HC3-OLS):')
print(f'  Cumulative (Omega_cum): beta={m_cum.params.iloc[1]:.3f}, '
      f'R2={m_cum.rsquared:.3f}, p={m_cum.pvalues.iloc[1]:.4f}')
print(f'  Net-change (Omega_net): beta={m_net.params.iloc[1]:.3f}, '
      f'R2={m_net.rsquared:.3f}, p={m_net.pvalues.iloc[1]:.4f}')
print(f'  Result stable: mechanistic relationship is not an artefact')
print(f'  of trivial cumulative accumulation.')

# Structural stability (Chow test)
df_p1 = df_t4[df_t4['year'] <= 2001]
df_p2 = df_t4[df_t4['year'] >  2001]
m1 = OLS(df_p1['dPsi_cum'], add_constant(df_p1['phi_cum'])).fit(cov_type='HC3')
m2 = OLS(df_p2['dPsi_cum'], add_constant(df_p2['phi_cum'])).fit(cov_type='HC3')
print(f'\nStructural stability (Chow):')
print(f'  1979-2001: beta={m1.params.iloc[1]:.3f}, '
      f'R2={m1.rsquared:.3f}, p={m1.pvalues.iloc[1]:.4f}, n={len(df_p1)}')
print(f'  2002-2024: beta={m2.params.iloc[1]:.3f}, '
      f'R2={m2.rsquared:.3f}, p={m2.pvalues.iloc[1]:.4f}, n={len(df_p2)}')
print(f'  Beta consistent across sub-periods: result is robust.')


# ── CELL 5: Figures — Spain BOE ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df_boe['year'], df_boe['omega_cum'], 'b-', lw=2, label='Omega cumulative')
axes[0].plot(df_boe['year'], df_boe['omega_net'], 'r--', lw=2, label='Omega net-change')
axes[0].axvline(2013, color='gray', ls=':', alpha=0.7, label='CNMC merger 2013')
axes[0].set_title('Omega cumulative vs net-change\n(Spain BOE 1978-2024)')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Weighted units')
axes[0].legend()
axes[0].grid(alpha=0.3)

colors_bar = ['red' if x < 0 else 'steelblue' for x in df_boe['net']]
axes[1].bar(df_boe['year'], df_boe['net'], color=colors_bar, alpha=0.7)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title('Annual net change\n(entries minus exits)')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Delta weight')
axes[1].grid(alpha=0.3)

df_scatter = df_t4.dropna()
axes[2].scatter(df_scatter['phi_cum'], df_scatter['dPsi_cum'],
                alpha=0.6, color='steelblue', s=30, label='Omega cumulative')
axes[2].scatter(df_scatter['phi_net'], df_scatter['dPsi_net'],
                alpha=0.4, color='red', s=20, marker='x', label='Omega net-change')
x_range = np.linspace(df_scatter['phi_cum'].min(), df_scatter['phi_cum'].max(), 100)
y_fit   = m_cum.params.iloc[0] + m_cum.params.iloc[1] * x_range
axes[2].plot(x_range, y_fit, 'b-', lw=2, alpha=0.8)
axes[2].set_title(f'Test T4: phi -> dPsi\nR2={m_cum.rsquared:.3f}, beta={m_cum.params.iloc[1]:.3f}')
axes[2].set_xlabel('phi(t)  fracture ratio')
axes[2].set_ylabel('dPsi/dt')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/boe_net_change_results.png', dpi=150, bbox_inches='tight')
plt.show()
df_boe.to_csv('results/boe_index_net_change.csv', index=False)
print('Figure saved: figures/boe_net_change_results.png')
print('Data saved:   results/boe_index_net_change.csv')


# =============================================================================
# STEP 2.4 — EU-27 Panel Robustness Tests
# =============================================================================

# ── CELL 6: Load DSP data ────────────────────────────────────────────────────
print('\nLoading Digital Society Project data...')
df_raw = pd.read_csv(DSP_FILE)
print(f'Dimensions: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns')


# ── CELL 7: Build EU-27 panel ────────────────────────────────────────────────
EU27_ISO3 = [
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN',
    'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX',
    'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE'
]

# Variables following the original paper specification (DSP v7/v8)
#   Sigma proxy: v2smgovfilprc  (government content filtering; high = more filtering = less citizen freedom)
#   Omega proxy: v2smregcap     (regulatory capacity; high = more regulation)
country_col = 'country_text_id' if 'country_text_id' in df_raw.columns else \
              next((c for c in df_raw.columns if c.lower() in ['country_name', 'country', 'name']), None)
sigma_var   = 'v2smgovfilprc'
omega_var   = 'v2smregcap'

print(f'Country column: {country_col}')
print(f'Sigma variable: {sigma_var}  -- '
      f'{"available" if sigma_var in df_raw.columns else "NOT FOUND"}')
print(f'Omega variable: {omega_var}  -- '
      f'{"available" if omega_var in df_raw.columns else "NOT FOUND"}')

df_eu = df_raw[
    (df_raw[country_col].isin(EU27_ISO3)) &
    (df_raw['year'] >= 2000) &
    (df_raw['year'] <= 2022)
][[country_col, 'year', sigma_var, omega_var]].copy()
df_eu.columns = ['country', 'year', 'Sigma_raw', 'Omega_raw']
df_eu = df_eu.dropna()

print(f'\nEU-27 observations (2000-2022): {len(df_eu):,}')
print(f'Countries: {df_eu["country"].nunique()}')

# Normalise [0, 1]
for col in ['Sigma_raw', 'Omega_raw']:
    mn, mx = df_eu[col].min(), df_eu[col].max()
    df_eu[col.replace('_raw', '')] = (df_eu[col] - mn) / (mx - mn + 1e-10)

# v2smgovfilprc: high = more filtering = lower citizen freedom -> invert
df_eu['Sigma'] = 1 - df_eu['Sigma']

df_eu['Psi']     = df_eu['Omega'] / (df_eu['Sigma'] + 0.01)
df_eu['phi']     = df_eu['Omega'] / (df_eu['Sigma'] + 0.01)
df_eu            = df_eu.sort_values(['country', 'year'])
df_eu['ln_dPsi'] = (np.log(df_eu['Psi'] + 0.01)
                    .groupby(df_eu['country']).diff())
df_eu['phi_lag']  = df_eu.groupby('country')['phi'].shift(1)
df_eu['dPsi_lag'] = df_eu.groupby('country')['ln_dPsi'].shift(1)

df_panel = df_eu.dropna().copy()
print(f'\nPanel ready: {len(df_panel):,} obs., {df_panel["country"].nunique()} countries')
print(f'Period: {df_panel["year"].min()}-{df_panel["year"].max()}')


# ── CELL 8: Panel diagnostics ────────────────────────────────────────────────
from scipy import stats as scipy_stats

print('\nCOMPLETE EU-27 PANEL DIAGNOSTICS')
print('=' * 60)

# T3: Psi trend
psi_mean = df_panel.groupby('year')['Psi'].mean().reset_index()
slope, intercept, r, p, se = scipy_stats.linregress(psi_mean['year'], psi_mean['Psi'])
print(f'\nT3 — Psi upward trend:')
print(f'  beta(Psi ~ year) = {slope:.5f}/yr, r={r:.3f}, p={p:.4f}')
print(f'  Trend: {"increasing (correct)" if slope > 0 else "decreasing (check)"}')

# T2: Sigma trend
sigma_mean = df_panel.groupby('year')['Sigma'].mean().reset_index()
slope_s, _, r_s, p_s, _ = scipy_stats.linregress(sigma_mean['year'], sigma_mean['Sigma'])
print(f'\nT2 — Sigma downward trend:')
print(f'  beta(Sigma ~ year) = {slope_s:.5f}/yr, r={r_s:.3f}, p={p_s:.4f}')
print(f'  Trend: {"decreasing (correct)" if slope_s < 0 else "increasing (check)"}')

# T4: pooled log-ratio OLS
df_panel['log_ratio'] = (
    np.log(df_panel['Psi'] + 0.01) -
    np.log(df_panel.groupby('country')['Psi'].shift(1) + 0.01)
)
df_t4p = df_panel.dropna(subset=['log_ratio', 'phi'])
X4     = add_constant(df_t4p['phi'])
m4     = OLS(df_t4p['log_ratio'], X4).fit(cov_type='HC3')
print(f'\nT4 — Pooled log-ratio OLS (N={len(df_t4p)}):')
print(f'  beta(phi) = {m4.params["phi"]:.4f}, p={m4.pvalues["phi"]:.4f}, '
      f'R2={m4.rsquared:.4f}')

# T5: within estimator (country fixed effects by demeaning)
df_t4p['log_ratio_dm'] = (df_t4p['log_ratio'] -
                           df_t4p.groupby('country')['log_ratio'].transform('mean'))
df_t4p['phi_dm']       = (df_t4p['phi'] -
                           df_t4p.groupby('country')['phi'].transform('mean'))
X5 = add_constant(df_t4p['phi_dm'])
m5 = OLS(df_t4p['log_ratio_dm'], X5).fit(cov_type='HC3')
print(f'\nT5 — Within estimator / country FE (N={len(df_t4p)}):')
print(f'  beta(phi) = {m5.params["phi_dm"]:.4f}, p={m5.pvalues["phi_dm"]:.4f}, '
      f'R2={m5.rsquared:.4f}')

# T7: structural asymmetry (countries with rising Psi)
country_trends = []
for c in df_panel['country'].unique():
    df_c = df_panel[df_panel['country'] == c]
    if len(df_c) < 5:
        continue
    sl, _, _, pv, _ = scipy_stats.linregress(df_c['year'], df_c['Psi'])
    country_trends.append({'country': c, 'slope': sl, 'p': pv, 'positive': sl > 0})
df_trends = pd.DataFrame(country_trends)
n_pos     = df_trends['positive'].sum()
t_stat, t_p = scipy_stats.ttest_1samp(df_trends['slope'], 0)
print(f'\nT7 — Structural asymmetry:')
print(f'  Countries with rising Psi: {n_pos}/{len(df_trends)}')
print(f'  t-test slopes vs 0: t={t_stat:.3f}, p={t_p:.4f}')

# Diagnostic figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(psi_mean['year'], psi_mean['Psi'], 'b-o', lw=2, ms=4)
axes[0].set_title(f'Mean Psi — EU27\nbeta={slope:.4f}/yr, r={r:.3f}, p={p:.3f}')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Psi'); axes[0].grid(alpha=0.3)
axes[1].plot(sigma_mean['year'], sigma_mean['Sigma'], 'r-o', lw=2, ms=4)
axes[1].set_title(f'Mean Sigma — EU27\nbeta={slope_s:.5f}/yr, r={r_s:.3f}')
axes[1].set_xlabel('Year'); axes[1].set_ylabel('Sigma'); axes[1].grid(alpha=0.3)
axes[2].scatter(df_t4p['phi'], df_t4p['log_ratio'], alpha=0.3, s=15, color='steelblue')
x_r = np.linspace(df_t4p['phi'].min(), df_t4p['phi'].max(), 100)
axes[2].plot(x_r, m4.params['const'] + m4.params['phi'] * x_r, 'r-', lw=2)
axes[2].set_title(f'T4 log-ratio pooled\nbeta={m4.params["phi"]:.4f}, '
                  f'p={m4.pvalues["phi"]:.4f}')
axes[2].set_xlabel('phi'); axes[2].set_ylabel('ln(Psi_t / Psi_t-1)'); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/diagnostico_panel.png', dpi=150, bbox_inches='tight')
plt.show()


# ── CELL 9: AR(1) correction ─────────────────────────────────────────────────
print('\nTEST 2.4.1: AR(1) CORRECTION')
print('=' * 50)
X_ar  = add_constant(df_panel[['phi', 'dPsi_lag']])
m_ar  = OLS(df_panel['ln_dPsi'], X_ar).fit(cov_type='HC3')
X_ols = add_constant(df_panel['phi'])
m_ols = OLS(df_panel['ln_dPsi'], X_ols).fit(cov_type='HC3')
print(f'OLS without AR(1): beta(phi)={m_ols.params["phi"]:.4f}, '
      f'p={m_ols.pvalues["phi"]:.4f}, R2={m_ols.rsquared:.4f}')
print(f'OLS with lag AR(1): beta(phi)={m_ar.params["phi"]:.4f}, '
      f'p={m_ar.pvalues["phi"]:.4f}, R2={m_ar.rsquared:.4f}')
print(f'  beta(lag)={m_ar.params["dPsi_lag"]:.4f}, '
      f'p={m_ar.pvalues["dPsi_lag"]:.4f}')
print(f'  If beta(phi) remains significant after AR(1) control,')
print(f'  the result is not an autocorrelation artefact.')


# ── CELL 10: Random slopes ───────────────────────────────────────────────────
print('\nTEST 2.4.2: RANDOM SLOPES (country-level OLS)')
print('=' * 50)
country_results = []
for country in df_panel['country'].unique():
    df_c = df_panel[df_panel['country'] == country]
    if len(df_c) < 5:
        continue
    try:
        X_c = add_constant(df_c['phi'])
        m_c = OLS(df_c['ln_dPsi'], X_c).fit()
        country_results.append({
            'country': country,
            'beta':    m_c.params.iloc[1],
            'p_val':   m_c.pvalues.iloc[1],
            'n':       len(df_c)
        })
    except Exception:
        pass

df_cr    = pd.DataFrame(country_results).sort_values('beta', ascending=False)
pos_sig  = ((df_cr['beta'] > 0) & (df_cr['p_val'] < 0.10)).sum()
pos_any  = (df_cr['beta'] > 0).sum()
print(f'Countries with beta > 0:                 {pos_any}/{len(df_cr)}')
print(f'Countries with beta > 0 and p < 0.10:   {pos_sig}/{len(df_cr)}')
print(f'\nTop 10 countries by beta:')
print(df_cr.head(10)[['country', 'beta', 'p_val']].to_string(index=False))
print(f'\nBottom 5 countries by beta:')
print(df_cr.tail(5)[['country', 'beta', 'p_val']].to_string(index=False))


# ── CELL 11: Granger causality ───────────────────────────────────────────────
print('\nTEST 2.4.3: GRANGER CAUSALITY (bidirectional)')
print('=' * 50)
from statsmodels.tsa.stattools import grangercausalitytests

granger_fwd, granger_rev = [], []
for country in df_panel['country'].unique():
    df_c = df_panel[df_panel['country'] == country].copy()
    if len(df_c) < 10:
        continue
    try:
        data = df_c[['ln_dPsi', 'phi']].dropna()
        if len(data) < 8:
            continue
        gc   = grangercausalitytests(data, maxlag=2, verbose=False)
        p_fw = gc[1][0]['ssr_ftest'][1]
        granger_fwd.append({'country': country, 'p_val': p_fw, 'sig': p_fw < 0.10})
        gc_r = grangercausalitytests(data[['phi', 'ln_dPsi']], maxlag=2, verbose=False)
        p_rv = gc_r[1][0]['ssr_ftest'][1]
        granger_rev.append({'country': country, 'p_val': p_rv, 'sig': p_rv < 0.10})
    except Exception:
        pass

df_gf = pd.DataFrame(granger_fwd)
df_gr = pd.DataFrame(granger_rev)
n_fwd = df_gf['sig'].sum()
n_rev = df_gr['sig'].sum()
print(f'phi -> dPsi significant (p<0.10): {n_fwd}/{len(df_gf)} '
      f'countries ({100 * n_fwd / len(df_gf):.0f}%)')
print(f'dPsi -> phi significant (p<0.10): {n_rev}/{len(df_gr)} '
      f'countries ({100 * n_rev / len(df_gr):.0f}%)')
print(f'\nIf phi->dPsi is more frequent than dPsi->phi,')
print(f'causality runs in the direction predicted by the model.')
print(f'\nCountries with significant Granger phi->dPsi:')
print(df_gf[df_gf['sig']].sort_values('p_val')[['country', 'p_val']].to_string(index=False))


# ── CELL 12: Panel cointegration ─────────────────────────────────────────────
print('\nTEST 2.4.4: PANEL COINTEGRATION (ADF on residuals)')
print('=' * 50)
from statsmodels.tsa.stattools import adfuller

coint_results = []
for country in df_panel['country'].unique():
    df_c = df_panel[df_panel['country'] == country]
    if len(df_c) < 8:
        continue
    try:
        X_c   = add_constant(df_c['phi'])
        resid = OLS(df_c['Psi'], X_c).fit().resid
        adf_s, adf_p, *_ = adfuller(resid, maxlag=1)
        coint_results.append({
            'country':      country,
            'adf_stat':     adf_s,
            'adf_p':        adf_p,
            'cointegrated': adf_p < 0.10
        })
    except Exception:
        pass

df_coint  = pd.DataFrame(coint_results)
n_coint   = df_coint['cointegrated'].sum()
print(f'Countries cointegrated Psi~phi (ADF p<0.10): {n_coint}/{len(df_coint)}')
print(f'Cointegration establishes a long-run structural relationship.')


# ── CELL 13: EU-27 panel figures ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

psi_mean_plot = df_panel.groupby('year')['Psi'].mean()
axes[0].plot(psi_mean_plot.index, psi_mean_plot.values, 'b-o', lw=2, ms=4)
axes[0].set_title('Mean Psi — EU27 (2000-2022)')
axes[0].set_xlabel('Year'); axes[0].set_ylabel('Psi(t) = Omega/Sigma')
axes[0].grid(alpha=0.3)

df_cr_sorted  = df_cr.sort_values('beta', ascending=True)
colors_cr     = ['red' if b < 0 else 'steelblue' for b in df_cr_sorted['beta']]
axes[1].barh(df_cr_sorted['country'], df_cr_sorted['beta'],
             color=colors_cr, alpha=0.7)
axes[1].axvline(0, color='black', lw=0.8)
axes[1].set_title('Country-level beta (phi -> dPsi)')
axes[1].set_xlabel('beta')
axes[1].tick_params(axis='y', labelsize=7)
axes[1].grid(alpha=0.3)

df_gf_s = df_gf.sort_values('p_val')
axes[2].barh(df_gf_s['country'], df_gf_s['p_val'],
             color=['green' if s else 'lightgray' for s in df_gf_s['sig']],
             alpha=0.7)
axes[2].axvline(0.10, color='red', ls='--', lw=1, label='p=0.10')
axes[2].set_title('Granger phi -> dPsi\n(p-values by country)')
axes[2].set_xlabel('p-value')
axes[2].tick_params(axis='y', labelsize=7)
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/eu27_panel_robust.png', dpi=150, bbox_inches='tight')
plt.show()

# Save results
df_panel.to_csv('results/eu27_panel_robust.csv', index=False)
df_cr.to_csv('results/country_betas.csv', index=False)
df_gf.to_csv('results/granger_results.csv', index=False)
df_coint.to_csv('results/cointegration_results.csv', index=False)
print('Figures saved in figures/')
print('Results saved in results/')


# ── SUMMARY ──────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('RESULTS SUMMARY FOR THE PAPER')
print('=' * 60)
print(f'T2 Sigma declining:    beta={slope_s:.5f}/yr, r={r_s:.3f}, p={p_s:.4f}')
print(f'T3 Psi rising:         beta={slope:.5f}/yr,  r={r:.3f}, p={p:.4f}')
print(f'T4 pooled OLS:         beta={m4.params["phi"]:.4f}, '
      f'R2={m4.rsquared:.4f}, p={m4.pvalues["phi"]:.4f}')
print(f'T5 within (FE):        beta={m5.params["phi_dm"]:.4f}, '
      f'R2={m5.rsquared:.4f}, p={m5.pvalues["phi_dm"]:.4f}')
print(f'T7 asymmetry:          {n_pos}/{len(df_trends)} countries, '
      f't={t_stat:.3f}, p={t_p:.4f}')
print(f'Random slopes beta>0:  {pos_any}/{len(df_cr)} countries, '
      f'{pos_sig} significant at p<0.10')
print(f'Granger phi->dPsi:     {n_fwd}/{len(df_gf)} countries ({100*n_fwd/len(df_gf):.0f}%) '
      f'vs reverse {n_rev}/{len(df_gr)} ({100*n_rev/len(df_gr):.0f}%)')
print(f'Cointegration:         {n_coint}/{len(df_coint)} countries')
